"""
Core implementation of the EnhancedRecommendationAgent focusing on a strict
planning → memory → reasoning pipeline where every module is injected
explicitly by each workflow.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List

from websocietysimulator.agent import RecommendationAgent

from base_agent_schema_validator import AgentSchemaValidator
from utils import (
    filter_item_info,
    parse_recommendation_result,
    validate_recommendations,
)


class EnhancedRecommendationAgentBase(RecommendationAgent):
    """
    Base class that wires together the modular planning → memory → reasoning workflow.
    Context gathering is built in so workflows only swap the three primary modules.
    """

    def __init__(
        self,
        llm,
        *,
        planning_module,
        memory_module,
        reasoning_module,
        info_orchestrator,
    ):
        super().__init__(llm=llm)
        for name, module in {
            "planning_module": planning_module,
            "memory_module": memory_module,
            "reasoning_module": reasoning_module,
            "info_orchestrator": info_orchestrator,
        }.items():
            if module is None:
                raise ValueError(f"{name} must not be None")

        self.planning = planning_module
        self.memory = memory_module
        self.reasoning = reasoning_module
        self.info_orchestrator = info_orchestrator

    def workflow(self):
        """
        Default workflow using the modules provided at initialization.
        """
        return self._execute_generic_workflow(
            workflow_name="Default Modular Workflow",
            planning_module=self.planning,
            memory_module=self.memory,
            reasoning_module=self.reasoning,
            info_orchestrator=self.info_orchestrator,
        )

    def _execute_generic_workflow(
        self,
        workflow_name: str,
        *,
        planning_module,
        memory_module,
        reasoning_module,
        info_orchestrator,
    ):
        """
        Shared execution pipeline. Every caller must pass the three modules.
        """
        logging.info("Executing %s", workflow_name)

        plan = self._generate_plan(planning_module)
        context = self._gather_context(plan)
        profiled_context = self._build_profile(context, info_orchestrator)
        enriched_context = self._integrate_memory(profiled_context, memory_module)
        recommendations = self._reason_and_rank(enriched_context, reasoning_module)

        return recommendations

    def _generate_plan(self, planning_module) -> List[Dict[str, str]]:
        if not self.task:
            raise RuntimeError("No active task. Did you call insert_task?")

        plan_task = (
            f"Create recommendations for user {self.task['user_id']} "
            f"from {len(self.task['candidate_list'])} items"
        )
        plan = planning_module(
            task_type="Recommendation",
            task_description=plan_task,
            feedback="",
            few_shot="",
        )
        AgentSchemaValidator.validate_plan(plan)
        
        return plan

    def _gather_context(
        self,
        plan: List[Dict[str, str]],
    ) -> Dict[str, List[Dict[str, str]]]:
        if not self.interaction_tool:
            raise RuntimeError("Interaction tool must be set before running the workflow.")

        user_id = self.task["user_id"]
        user_profile = self.interaction_tool.get_user(user_id=user_id)
        user_reviews = self.interaction_tool.get_reviews(user_id=user_id)

        candidate_items: List[Dict[str, str]] = []
        for item_id in self.task["candidate_list"]:
            try:
                item = self.interaction_tool.get_item(item_id=item_id)
                candidate_items.append(filter_item_info(item) if item else {"item_id": item_id})
            except Exception as exc:  # pylint: disable=broad-except
                candidate_items.append({"item_id": item_id, "error": str(exc)})
        
        AgentSchemaValidator.validate_context(
            user_profile=user_profile,
            user_reviews=user_reviews,
            candidate_items=candidate_items,
            plan=plan,
        )
        return {
            "plan": plan,
            "user_profile": user_profile,
            "user_reviews": user_reviews,
            "candidate_items": candidate_items,
        }

    def _integrate_memory(
        self,
        context: Dict[str, List[Dict[str, str]]],
        memory_module,
    ) -> Dict[str, List[Dict[str, str]]]:
        user_reviews = context.get("user_reviews") or []
        context["memory_context"] = ""

        for review in user_reviews[:20]:
            if "text" in review and review["text"]:
                review_summary = (
                    f"Stars: {review.get('stars', 'N/A')}, "
                    f"Text: {review['text'][:200]}"
                )
                memory_module(f"review: {review_summary}")

        if user_reviews:
            seed_text = user_reviews[0].get("text", "")[:200]
            if seed_text:
                retrieved = memory_module(seed_text)
                if retrieved:
                    context["memory_context"] = f"Memory Context:\n{retrieved}"

        AgentSchemaValidator.validate_memory_context(context.get("memory_context"))
        return context

    def _reason_and_rank(
        self,
        context: Dict[str, List[Dict[str, str]]],
        reasoning_module,
    ) -> List[str]:
        user_reviews = context.get("user_reviews") or []
        payload = {
            "user_profile": context.get("user_profile"),
            "user_reviews": user_reviews,
            "candidate_items": context.get("candidate_items") or [],
            "candidate_list": self.task["candidate_list"],
            "plan": context.get("plan") or [],
        }
        if context.get("memory_context"):
            payload["memory_context"] = context["memory_context"]
        if context.get("user_persona"):
            payload["user_persona"] = context["user_persona"]

        task_description = (
            "You are a recommendation system. Use the JSON context below to rank "
            "candidate items for the specified user. Return ONLY a Python list of "
            "item IDs drawn from candidate_list.\n"
            f"CONTEXT:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        result = reasoning_module(task_description, user_id=self.task["user_id"])
        ranked_list = parse_recommendation_result(result)
        return validate_recommendations(ranked_list, self.task["candidate_list"])

    def _build_profile(self, context, info_orchestrator):
        """
        Build user/book/item profiles using InfoOrchestrator.
        
        Workflow:
        1. InfoOrchestrator analyzes planner steps
        2. Retrieves parameters from memory
        3. Calls schema_fitter to build profiles
        4. Returns consolidated profiles
        """
        plan = context.get("plan", [])
        user_id = self.task.get("user_id")
        candidate_list = self.task.get("candidate_list", [])
        item_id = candidate_list[0] if candidate_list else None
        
        profile_results = info_orchestrator(
            planner_steps=plan,
            user_id=user_id,
            item_id=item_id,
            candidate_list=candidate_list
        )
        
        if profile_results.get("user_profile"):
            context["user_persona"] = profile_results["user_profile"]
        else:
            context.setdefault("user_persona", {})
        
        if profile_results.get("item_profiles"):
            context["item_profiles"] = profile_results["item_profiles"]
        elif profile_results.get("item_profile"):
            context["item_profiles"] = [profile_results["item_profile"]]
        
        return context

