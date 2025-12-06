"""
Utility helpers for the enhanced recommendation agent.
"""

import logging
import re
from typing import List


def filter_item_info(item):
    """Filter item information to keep only relevant fields."""
    keys_to_extract = [
        "item_id",
        "business_id",
        "asin",
        "name",
        "title",
        "title_without_series",
        "stars",
        "average_rating",
        "ratings_count",
        "review_count",
        "rating_number",
        "categories",
        "attributes",
        "description",
        "price",
        "authors",
        "publisher",
    ]

    filtered = {}
    for key in keys_to_extract:
        if key in item and item[key] is not None:
            value = item[key]
            if isinstance(value, str) and len(value) > 300:
                filtered[key] = value[:300] + "..."
            elif isinstance(value, dict):
                filtered[key] = str(value)[:200]
            else:
                filtered[key] = value
    return filtered


def parse_recommendation_result(result: str) -> List[str]:
    """Extract a ranked list of item IDs from an LLM response."""
    try:
        match = re.search(r"\[.*?\]", result, re.DOTALL)
        if match:
            list_str = match.group()
            try:
                ranked_list = eval(list_str)  # noqa: S307
                if isinstance(ranked_list, list):
                    return ranked_list
            except Exception:  # pylint: disable=broad-except
                pass

        items = re.findall(r'["\']([^"\']+)["\']', result)
        if items:
            return items

        return []
    except Exception:  # pylint: disable=broad-except
        return []


def validate_recommendations(ranked_list, candidate_list):
    """Ensure the final ranking only contains allowed IDs without duplicates."""
    seen = set()
    unique_list = []
    for item_id in ranked_list:
        if item_id not in seen and item_id in candidate_list:
            seen.add(item_id)
            unique_list.append(item_id)

    for item_id in candidate_list:
        if item_id not in seen:
            unique_list.append(item_id)
    return unique_list
