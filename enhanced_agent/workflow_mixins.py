"""
Workflow mixin that houses all alternative module combinations for the enhanced
recommendation agent.
"""

from typing import List

from websocietysimulator.agent.modules.memory_modules_custom import (
    MemoryDILU,
    MemoryGenerative,
    MemoryTP,
    MemoryVoyager,
)
from websocietysimulator.agent.modules.planning_modules_custom import (
    PlanningDEPSCustom,
    PlanningHUGGINGGPTCustom,
    PlanningOPENAGICustom,
    PlanningTDCustom,
    PlanningVoyagerCustom,
)
from websocietysimulator.agent.modules.reasoning_modules import (
    ReasoningCOT,
    ReasoningCOTSC,
    ReasoningIO,
    ReasoningSelfRefine,
    ReasoningTOT,
)


class EnhancedWorkflowMixin:
    """
    Provides the various workflow combinations that were previously defined in
    the monolithic EnhancedRecommendationAgent.
    """

    def workflow_with_voyager_planning(self) -> List[str]:
        voyager_planning = PlanningVoyagerCustom(llm=self.llm)
        return self._execute_generic_workflow(
            workflow_name="Voyager Planning",
            planning_module=voyager_planning,
            memory_module=self.memory,
            reasoning_module=self.reasoning,
            info_orchestrator=self.info_orchestrator,
        )

    def workflow_with_self_refine(self) -> List[str]:
        self_refine_reasoning = ReasoningSelfRefine(
            profile_type_prompt="You are an intelligent recommendation system.",
            memory=None,
            llm=self.llm,
        )
        return self._execute_generic_workflow(
            workflow_name="Self-Refine",
            planning_module=self.planning,
            memory_module=self.memory,
            reasoning_module=self_refine_reasoning,
            info_orchestrator=self.info_orchestrator,
        )

    def workflow_with_cot_sc(self) -> List[str]:
        cot_sc_reasoning = ReasoningCOTSC(
            profile_type_prompt="You are an intelligent recommendation system.",
            memory=None,
            llm=self.llm,
        )
        return self._execute_generic_workflow(
            workflow_name="COT-SC (Self-Consistency)",
            planning_module=self.planning,
            memory_module=self.memory,
            reasoning_module=cot_sc_reasoning,
            info_orchestrator=self.info_orchestrator,
        )

    def workflow_with_voyager_memory(self) -> List[str]:
        voyager_memory = MemoryVoyager(llm=self.llm)
        return self._execute_generic_workflow(
            workflow_name="Voyager Memory",
            planning_module=self.planning,
            memory_module=voyager_memory,
            reasoning_module=self.reasoning,
            info_orchestrator=self.info_orchestrator,
        )

    def workflow_with_openagi_planning(self) -> List[str]:
        openagi_planning = PlanningOPENAGICustom(llm=self.llm)
        return self._execute_generic_workflow(
            workflow_name="OpenAGI Planning",
            planning_module=openagi_planning,
            memory_module=self.memory,
            reasoning_module=self.reasoning,
            info_orchestrator=self.info_orchestrator,
        )

    def workflow_hybrid_advanced(self) -> List[str]:
        huggingpt_planning = PlanningHUGGINGGPTCustom(llm=self.llm)
        tp_memory = MemoryTP(llm=self.llm)
        cot_reasoning = ReasoningCOT(
            profile_type_prompt="You are an intelligent recommendation system.",
            memory=tp_memory,
            llm=self.llm,
        )
        return self._execute_generic_workflow(
            workflow_name="Hybrid Advanced (HuggingGPT + TP Memory + COT)",
            planning_module=huggingpt_planning,
            memory_module=tp_memory,
            reasoning_module=cot_reasoning,
            info_orchestrator=self.info_orchestrator,
        )

    def workflow_with_tot_reasoning(self) -> List[str]:
        tot_reasoning = ReasoningTOT(
            profile_type_prompt="You are an intelligent recommendation system.",
            memory=None,
            llm=self.llm,
        )
        return self._execute_generic_workflow(
            workflow_name="Tree of Thoughts (TOT)",
            planning_module=self.planning,
            memory_module=self.memory,
            reasoning_module=tot_reasoning,
            info_orchestrator=self.info_orchestrator,
        )

    def workflow_with_td_planning(self) -> List[str]:
        td_planning = PlanningTDCustom(llm=self.llm)
        return self._execute_generic_workflow(
            workflow_name="Temporal Dependencies Planning",
            planning_module=td_planning,
            memory_module=self.memory,
            reasoning_module=self.reasoning,
            info_orchestrator=self.info_orchestrator,
        )

    def workflow_with_deps_planning(self) -> List[str]:
        deps_planning = PlanningDEPSCustom(llm=self.llm)
        generative_memory = MemoryGenerative(llm=self.llm)
        cot_reasoning = ReasoningCOT(
            profile_type_prompt="You are an intelligent recommendation system.",
            memory=generative_memory,
            llm=self.llm,
        )
        return self._execute_generic_workflow(
            workflow_name="Multi-Hop DEPS Planning",
            planning_module=deps_planning,
            memory_module=generative_memory,
            reasoning_module=cot_reasoning,
            info_orchestrator=self.info_orchestrator,
        )

    def workflow_all_voyager(self) -> List[str]:
        voyager_planning = PlanningVoyagerCustom(llm=self.llm)
        voyager_memory = MemoryVoyager(llm=self.llm)
        cot_reasoning = ReasoningCOT(
            profile_type_prompt="You are an intelligent recommendation system.",
            memory=voyager_memory,
            llm=self.llm,
        )
        return self._execute_generic_workflow(
            workflow_name="All Voyager Stack",
            planning_module=voyager_planning,
            memory_module=voyager_memory,
            reasoning_module=cot_reasoning,
            info_orchestrator=self.info_orchestrator,
        )

    def workflow_with_dilu_memory(self) -> List[str]:
        huggingpt_planning = PlanningHUGGINGGPTCustom(llm=self.llm)
        dilu_memory = MemoryDILU(llm=self.llm)
        return self._execute_generic_workflow(
            workflow_name="HuggingGPT + DILU Memory",
            planning_module=huggingpt_planning,
            memory_module=dilu_memory,
            reasoning_module=self.reasoning,
            info_orchestrator=self.info_orchestrator,
        )

    def workflow_simple_efficient(self) -> List[str]:
        io_reasoning = ReasoningIO(
            profile_type_prompt="You are an intelligent recommendation system.",
            memory=None,
            llm=self.llm,
        )
        return self._execute_generic_workflow(
            workflow_name="Simple Efficient (IO only)",
            planning_module=self.planning,
            memory_module=self.memory,
            reasoning_module=io_reasoning,
            info_orchestrator=self.info_orchestrator,
        )

    def workflow_tot_with_memory(self) -> List[str]:
        tp_memory = MemoryTP(llm=self.llm)
        tot_reasoning = ReasoningTOT(
            profile_type_prompt="You are an intelligent recommendation system.",
            memory=tp_memory,
            llm=self.llm,
        )
        return self._execute_generic_workflow(
            workflow_name="TOT + TP Memory (VERY EXPENSIVE)",
            planning_module=self.planning,
            memory_module=tp_memory,
            reasoning_module=tot_reasoning,
            info_orchestrator=self.info_orchestrator,
        )

    def workflow_deps_self_refine(self) -> List[str]:
        deps_planning = PlanningDEPSCustom(llm=self.llm)
        generative_memory = MemoryGenerative(llm=self.llm)
        self_refine_reasoning = ReasoningSelfRefine(
            profile_type_prompt="You are an intelligent recommendation system.",
            memory=generative_memory,
            llm=self.llm,
        )
        return self._execute_generic_workflow(
            workflow_name="DEPS + Self-Refine + Memory",
            planning_module=deps_planning,
            memory_module=generative_memory,
            reasoning_module=self_refine_reasoning,
            info_orchestrator=self.info_orchestrator,
        )

