"""
search_reference.py — Load and inject Mathlib search references from search_results.json.

Provides high-confidence (score > threshold) Lean/Mathlib theorem matches
to enrich LLM prompts with authoritative type signatures and descriptions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from utils import logger


# ─────────────────────────────────────────────────────────────────────────────
# Search Reference Map
# ─────────────────────────────────────────────────────────────────────────────

_SEARCH_REF_MAP: dict[str, dict] = {}


def load_search_references(
    search_results_path: str,
    score_threshold: float = 0.92,
) -> dict[str, dict]:
    """
    Load search_results.json and build a map of id → search_info
    for samples with search_score > threshold.

    Returns:
        dict mapping sample id (e.g. "formalrx_0002") to search info dict
    """
    global _SEARCH_REF_MAP

    gt_path = Path(search_results_path)
    if not gt_path.exists():
        logger.warning(f"Ground truth file not found: {gt_path}")
        return {}

    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Error loading ground truth: {e}")
        return {}

    ref_map = {}
    for entry in data:
        score = entry.get("search_score", 0)
        if score and score > score_threshold:
            sample_id = entry.get("id", "")
            if not sample_id:
                continue
            ref_map[sample_id] = {
                "search_score": score,
                "search_formal_name": entry.get("search_formal_name", ""),
                "search_type": entry.get("search_type", ""),
                "search_informal_name": entry.get("search_informal_name", ""),
                "search_informal_description": entry.get("search_informal_description", ""),
                "search_path": entry.get("search_path", ""),
                "search_kind": entry.get("search_kind", ""),
            }

    _SEARCH_REF_MAP = ref_map
    logger.info(
        f"Loaded {len(ref_map)} high-confidence search references "
        f"(score > {score_threshold}) from {gt_path}"
    )
    return ref_map


def get_search_reference(sample_id: str) -> Optional[dict]:
    """Get search reference for a sample by its id. Returns None if not found."""
    return _SEARCH_REF_MAP.get(sample_id)


def format_search_reference_block(search_ref: dict) -> str:
    """
    Format a search reference dict into a prompt block.

    Returns a markdown-formatted block to inject into LLM prompts.
    """
    if not search_ref:
        return ""

    score = search_ref.get("search_score", 0)
    formal_name = search_ref.get("search_formal_name", "")
    search_type = search_ref.get("search_type", "")
    description = search_ref.get("search_informal_description", "")
    path = search_ref.get("search_path", "")
    kind = search_ref.get("search_kind", "theorem")

    lines = [f"## Mathlib Reference (high-confidence match, score={score:.3f})"]
    
    if description:
        lines.append(f"**Description**: {description}")

    if formal_name:
        lines.append(f"**Name**: `{formal_name}`")
    if kind:
        lines.append(f"**Kind**: {kind}")
    if search_type:
        lines.append(f"```lean")
        lines.append(search_type)
        lines.append(f"```")
    
    if path:
        lines.append(f"Mathlib path: `{path}`")

    lines.append("")
    lines.append(
        "⚠ Use this reference to compare with the formal statement. "
    )

    return "\n".join(lines)
