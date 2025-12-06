from __future__ import annotations

from typing import Dict, List, Optional


class AgentSchemaValidator:
    """Centralized validation helpers for EnhancedRecommendationAgent."""

    PLAN_REQUIRED_KEYS = {"description", "reasoning instruction", "tool instruction"}
    REVIEW_REQUIRED_KEYS = {"review_id", "item_id", "text", "stars"}
    PERSONA_REQUIRED_KEYS = {"summary", "core_preferences", "avoidances", "recent_sentiment", "evidence"}
    PERSONA_SENTIMENT_VALUES = {"positive", "neutral", "negative"}

    @staticmethod
    def validate_plan(plan: List[Dict[str, str]]) -> None:
        if not isinstance(plan, list) or not plan:
            raise RuntimeError("Planning module must return a non-empty list of steps.")

        for idx, step in enumerate(plan):
            if not isinstance(step, dict):
                raise RuntimeError(f"Planning step {idx} must be a dict following planning schema.")

            missing = AgentSchemaValidator.PLAN_REQUIRED_KEYS - step.keys()
            if missing:
                raise RuntimeError(
                    f"Planning step {idx} missing required keys: {', '.join(sorted(missing))}."
                )

            for key in AgentSchemaValidator.PLAN_REQUIRED_KEYS:
                value = step.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise RuntimeError(
                        f"Planning step {idx} field '{key}' must be a non-empty string."
                    )

    @staticmethod
    def validate_context(
        *,
        user_profile: Optional[Dict],
        user_reviews: Optional[List[Dict]],
        candidate_items: Optional[List[Dict]],
        plan: Optional[List[Dict]],
    ) -> None:
        AgentSchemaValidator.validate_plan(plan)

        if not isinstance(user_profile, dict):
            raise RuntimeError("Context validation failed: user_profile must be a dict.")

        if not isinstance(user_reviews, list):
            raise RuntimeError("Context validation failed: user_reviews must be a list.")

        for idx, review in enumerate(user_reviews):
            if not isinstance(review, dict):
                raise RuntimeError(f"user_reviews[{idx}] must be a dict.")
            missing = AgentSchemaValidator.REVIEW_REQUIRED_KEYS - review.keys()
            if missing:
                raise RuntimeError(
                    f"user_reviews[{idx}] missing required keys: {', '.join(sorted(missing))}."
                )

        if not isinstance(candidate_items, list):
            raise RuntimeError("Context validation failed: candidate_items must be a list.")
        for idx, item in enumerate(candidate_items):
            if not isinstance(item, dict):
                raise RuntimeError(f"candidate_items[{idx}] must be a dict.")
            if "item_id" not in item:
                raise RuntimeError(f"candidate_items[{idx}] missing 'item_id'.")

    @staticmethod
    def validate_memory_context(memory_context) -> None:
        if memory_context is None:
            return
        if not isinstance(memory_context, str):
            raise RuntimeError("Memory module output must be a string per memory_modules contract.")

    @staticmethod
    def validate_persona(persona: Dict) -> None:
        if persona is None or not isinstance(persona, dict):
            raise RuntimeError("Profile module output must be a dict.")

        missing = AgentSchemaValidator.PERSONA_REQUIRED_KEYS - persona.keys()
        if missing:
            raise RuntimeError(
                f"Profile module output missing required keys: {', '.join(sorted(missing))}."
            )

        summary = persona.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise RuntimeError("Persona summary must be a non-empty string.")

        for key in ("core_preferences", "avoidances"):
            value = persona.get(key)
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise RuntimeError(f"Persona field '{key}' must be a list of strings.")

        sentiment = persona.get("recent_sentiment")
        if sentiment not in AgentSchemaValidator.PERSONA_SENTIMENT_VALUES:
            raise RuntimeError("Persona recent_sentiment must be one of positive|neutral|negative.")

        evidence = persona.get("evidence")
        if not isinstance(evidence, list):
            raise RuntimeError("Persona evidence must be a list.")
        for idx, entry in enumerate(evidence):
            if not isinstance(entry, dict):
                raise RuntimeError(f"Persona evidence[{idx}] must be a dict.")
            if "snippet" not in entry or not isinstance(entry["snippet"], str):
                raise RuntimeError(f"Persona evidence[{idx}] must include a text snippet string.")
            if "review_id" not in entry:
                raise RuntimeError(f"Persona evidence[{idx}] missing review_id.")


