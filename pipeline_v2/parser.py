"""
parser.py — Parse and validate LLM output for FormalRx diagnoses.

Pipeline V2 additions:
  - parse_verdict_only(): lightweight verdict-only parser for Stage 1a
  - parse_correction_only(): correction-only parser for Stage 1b
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from taxonomy import normalize_category, GROUP_NAMES, CATEGORY_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Prediction:
    idx: str
    aligned: str            # internal: "Aligned" | "Misaligned"
    error_category: str     # canonical SCI category name | "N/A"
    error_segment: str      # minimal erroneous code snippet | "N/A"
    corrected_statement: str  # full corrected Lean 4 | "N/A"
    reasoning: str = ""     # chain-of-thought (not submitted)
    raw_response: str = ""  # original LLM text (not submitted)
    parse_error: str = ""   # non-empty if parsing failed

    def to_submission_dict(self) -> dict:
        """
        Return the dict in the exact flat format required by Codabench.

        Required schema:
          {"idx": str, "verdict": "aligned"|"misaligned",
           "error_category": str|null, "error_segment": str|null,
           "corrected_statement": str|null}
        """
        is_aligned = self.aligned.lower() == "aligned"
        return {
            "idx": self.idx,
            "verdict": "aligned" if is_aligned else "misaligned",
            "error_category": None if is_aligned else (
                None if self.error_category in ("N/A", "", None) else self.error_category
            ),
            "error_segment": None if is_aligned else (
                None if self.error_segment in ("N/A", "", None) else self.error_segment
            ),
            "corrected_statement": None if is_aligned else (
                None if self.corrected_statement in ("N/A", "", None) else self.corrected_statement
            ),
        }

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "aligned": self.aligned,
            "error_category": self.error_category,
            "error_segment": self.error_segment,
            "corrected_statement": self.corrected_statement,
            "reasoning": self.reasoning,
            "raw_response": self.raw_response,
            "parse_error": self.parse_error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Prediction:
        return cls(
            idx=d["idx"],
            aligned=d.get("aligned", "Aligned" if d.get("verdict") == "aligned" else "Misaligned"),
            error_category=d.get("error_category") or "N/A",
            error_segment=d.get("error_segment") or "N/A",
            corrected_statement=d.get("corrected_statement") or "N/A",
            reasoning=d.get("reasoning", ""),
            raw_response=d.get("raw_response", ""),
            parse_error=d.get("parse_error", ""),
        )


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_json_text(text: str) -> str:
    """
    Try to extract a JSON object from an LLM response.
    Handles:
    - Bare JSON
    - JSON wrapped in ```json ... ``` fences
    - JSON embedded in surrounding prose
    """
    text = text.strip()

    # 1. Try bare parse
    if text.startswith("{"):
        return text

    # 2. Try markdown fence
    match = _JSON_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()

    # 3. Find first { and last } and extract
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text


def _clean_json_text(text: str) -> str:
    """
    Sanitize raw JSON text before parsing:
    - Escapes unescaped backslashes (e.g. \\forall -> \\\\forall)
    """
    text = text.strip()
    pattern = r'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})|(\\)'
    return re.sub(pattern, lambda m: r'\\' if m.group(1) else m.group(0), text)


def _unescape_json_string(s: str) -> str:
    """Unescape backslash escapes in a raw JSON string value."""
    parts = s.split('\\\\')
    for i in range(len(parts)):
        parts[i] = parts[i].replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
    return '\\'.join(parts)


def parse_loose_json(text: str) -> dict:
    """
    Fallback parser for malformed JSON where values might contain unescaped quotes.
    Identifies keys and extracts string values between key boundaries.
    """
    known_keys = [
        "reasoning", "verdict", "error_location", "corrected_statement",
        "error_group", "error_type"
    ]
    
    matches = []
    for key in known_keys:
        # Match starting at a boundary or after a comma/bracket
        pattern = r'(?:^|[\{\s,])"' + key + r'"\s*:\s*"'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            key_start = match.start()
            while key_start < len(text) and text[key_start] in ('{', ',', ' ', '\n', '\r', '\t'):
                key_start += 1
            matches.append((key, key_start, match.end()))
            
    if not matches:
        return {}
        
    matches.sort(key=lambda x: x[1])
    
    result = {}
    for i in range(len(matches)):
        key, start_idx, val_start = matches[i]
        
        if i + 1 < len(matches):
            val_end = matches[i+1][1]
        else:
            val_end = text.rfind("}")
            if val_end == -1:
                val_end = len(text)
                
        val_text = text[val_start:val_end].strip()
        
        if val_text.endswith("}"):
            val_text = val_text[:-1].strip()
        if val_text.endswith(","):
            val_text = val_text[:-1].strip()
        if val_text.endswith('"'):
            val_text = val_text[:-1]
            
        result[key] = _unescape_json_string(val_text)
        
    return result


def strip_thinking(text: str) -> tuple[str, str]:
    """
    Remove thinking process from the model output.
    Returns (clean_text, thinking_process).
    """
    thinking = ""
    # Standard <think>...</think>
    match = re.search(r"<think>([\s\S]*?)</think>", text)
    if match:
        thinking = match.group(1).strip()
        text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    elif "</think>" in text:
        parts = text.split("</think>", 1)
        thinking = parts[0].replace("Thinking Process:", "").strip()
        text = parts[1]
    elif text.strip().startswith("Thinking Process:"):
        # If there is no closing tag but it starts with Thinking Process:
        parts = text.split("\n\n", 1)
        if len(parts) > 1:
            thinking = parts[0].replace("Thinking Process:", "").strip()
            text = parts[1]

    return text.strip(), thinking


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1a (V2): Verdict-only parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_verdict_only(
    idx: str,
    raw_response: str,
) -> tuple[str, str]:
    """
    Parse a Stage-1a (verdict-only) LLM response.

    Returns:
        (verdict, reasoning)
        verdict: "Aligned" | "Misaligned"
        reasoning: chain-of-thought text
    """
    clean_text, thinking = strip_thinking(raw_response)
    json_text = _clean_json_text(_extract_json_text(clean_text))
    try:
        data = json.loads(json_text, strict=False)
    except json.JSONDecodeError:
        # Try loose JSON parser fallback
        data = parse_loose_json(clean_text)
        if not data:
            # Fallback: try to find verdict keyword
            lower = clean_text.lower()
            if "aligned" in lower and "misaligned" not in lower:
                return "Aligned", thinking
            return "Misaligned", thinking

    reasoning = str(data.get("reasoning", ""))
    if thinking:
        reasoning = f"[Thinking]\n{thinking}\n\n[Reasoning]\n{reasoning}".strip()
    verdict_raw = str(data.get("verdict", "")).strip().lower()

    if verdict_raw == "aligned":
        return "Aligned", reasoning
    return "Misaligned", reasoning



# ─────────────────────────────────────────────────────────────────────────────
# Stage 1b (V2): Correction + Localization parser (merged)
# ─────────────────────────────────────────────────────────────────────────────

def parse_correction_localization(
    idx: str,
    raw_response: str,
) -> tuple[str, str, str]:
    """
    Parse a Stage-1b (correction + localization) LLM response.

    Expected JSON: {"reasoning": "...", "error_location": "...", "corrected_statement": "..."}

    Returns:
        (corrected_statement, error_location, reasoning)
    """
    clean_text, thinking = strip_thinking(raw_response)
    json_text = _clean_json_text(_extract_json_text(clean_text))
    try:
        data = json.loads(json_text, strict=False)
    except json.JSONDecodeError:
        data = parse_loose_json(clean_text)
        if not data:
            # Try to extract code from markdown fence
            code_match = re.search(r"```(?:lean)?\s*([\s\S]*?)```", clean_text)
            if code_match:
                return code_match.group(1).strip(), "N/A", thinking
            return "N/A", "N/A", thinking

    corrected = str(data.get("corrected_statement", "")).strip()
    error_location = str(data.get("error_location", "")).strip()
    reasoning = str(data.get("reasoning", ""))
    if thinking:
        reasoning = f"[Thinking]\n{thinking}\n\n[Reasoning]\n{reasoning}".strip()

    if not corrected or corrected.lower() in ("n/a", "null", "none", ""):
        corrected = "N/A"

    if not error_location or error_location.lower() in ("n/a", "null", "none", ""):
        error_location = "N/A"

    return corrected, error_location, reasoning


# Backward compatible alias
def parse_correction_only(
    idx: str,
    raw_response: str,
) -> tuple[str, str]:
    """Legacy wrapper: returns (corrected_statement, reasoning) only."""
    corrected, _, reasoning = parse_correction_localization(idx, raw_response)
    return corrected, reasoning


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 (legacy compatible): Verdict + Correction parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_verdict_correction(
    idx: str,
    raw_response: str,
) -> Prediction:
    """
    Parse a Stage-1 (verdict + correction) LLM response.

    Returns:
        Prediction with verdict and corrected_statement.
        If aligned: error fields are "N/A".
        If misaligned: corrected_statement is set, error_category/segment = "N/A" (filled later).
    """
    clean_text, thinking = strip_thinking(raw_response)
    json_text = _clean_json_text(_extract_json_text(clean_text))
    try:
        data = json.loads(json_text, strict=False)
    except json.JSONDecodeError:
        data = parse_loose_json(clean_text)
        if not data:
            # Fallback: try to find verdict keyword
            lower = clean_text.lower()
            if "aligned" in lower and "misaligned" not in lower:
                return Prediction(
                    idx=idx, aligned="Aligned",
                    error_category="N/A", error_segment="N/A",
                    corrected_statement="N/A",
                    reasoning=thinking,
                    raw_response=raw_response,
                    parse_error="Stage1 JSONDecodeError (fallback: aligned)",
                )
            return Prediction(
                idx=idx, aligned="Misaligned",
                error_category="N/A", error_segment="N/A",
                corrected_statement="N/A",
                reasoning=thinking,
                raw_response=raw_response,
                parse_error="Stage1 JSONDecodeError (fallback: misaligned)",
            )

    reasoning = str(data.get("reasoning", ""))
    if thinking:
        reasoning = f"[Thinking]\n{thinking}\n\n[Reasoning]\n{reasoning}".strip()
    verdict_raw = str(data.get("verdict", "")).strip().lower()
    corrected = str(data.get("corrected_statement", "")).strip()

    if verdict_raw == "aligned":
        return Prediction(
            idx=idx, aligned="Aligned",
            error_category="N/A", error_segment="N/A",
            corrected_statement="N/A",
            reasoning=reasoning,
            raw_response=raw_response,
        )

    # Misaligned
    if not corrected or corrected.lower() in ("n/a", "null", "none", ""):
        corrected = "N/A"

    return Prediction(
        idx=idx, aligned="Misaligned",
        error_category="N/A",  # filled by Stage 2
        error_segment="N/A",   # filled by Stage 2
        corrected_statement=corrected,
        reasoning=reasoning,
        raw_response=raw_response,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2b: Localization-only parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_localization_only(
    idx: str,
    raw_response: str,
) -> tuple[str, str]:
    """
    Parse a Stage-2b (localization-only) LLM response.

    Returns:
        (error_location, reasoning)
    """
    clean_text, thinking = strip_thinking(raw_response)
    json_text = _clean_json_text(_extract_json_text(clean_text))
    try:
        data = json.loads(json_text, strict=False)
    except json.JSONDecodeError:
        data = parse_loose_json(clean_text)
        if not data:
            return "N/A", thinking

    error_location = str(data.get("error_location", "N/A")).strip()
    reasoning = str(data.get("reasoning", ""))
    if thinking:
        reasoning = f"[Thinking]\n{thinking}\n\n[Reasoning]\n{reasoning}".strip()

    if not error_location or error_location.lower() in ("n/a", "null", "none"):
        error_location = "N/A"

    return error_location, reasoning


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2c Call 1: Group classification parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_group_classify(
    idx: str,
    raw_response: str,
) -> tuple[str, str]:
    """
    Parse a Stage-2c Call 1 (group classification) LLM response.

    Returns:
        (error_group, reasoning)
    """
    clean_text, thinking = strip_thinking(raw_response)
    json_text = _clean_json_text(_extract_json_text(clean_text))
    try:
        data = json.loads(json_text, strict=False)
    except json.JSONDecodeError as e:
        data = parse_loose_json(clean_text)
        if not data:
            raise ValueError(f"Failed to decode group classification JSON: {e}. Raw: {raw_response!r}") from e

    reasoning = str(data.get("reasoning", ""))
    if thinking:
        reasoning = f"[Thinking]\n{thinking}\n\n[Reasoning]\n{reasoning}".strip()
    error_group = str(data.get("error_group", "")).strip()

    if not error_group or error_group.lower() in ("n/a", "null", "none"):
        raise ValueError("Group classification returned empty or N/A group")

    # Match group case-insensitively
    matched_group = None
    for g in GROUP_NAMES:
        if g.lower() == error_group.lower():
            matched_group = g
            break

    if not matched_group:
        raise ValueError(f"Unrecognized error group: '{error_group}'")

    return matched_group, reasoning


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2c Call 2: Category classification parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_category_classify(
    idx: str,
    raw_response: str,
) -> tuple[str, str, str, str]:
    """
    Parse a Stage-2c Call 2 (category classification) LLM response.

    Returns:
        (error_category, error_location, reasoning, corrected_statement)
    """
    clean_text, thinking = strip_thinking(raw_response)
    json_text = _clean_json_text(_extract_json_text(clean_text))
    try:
        data = json.loads(json_text, strict=False)
    except json.JSONDecodeError as e:
        data = parse_loose_json(clean_text)
        if not data or "error_type" not in data:
            raise ValueError(f"Failed to decode category classification JSON: {e}. Raw: {raw_response!r}") from e

    reasoning = str(data.get("reasoning", ""))
    if thinking:
        reasoning = f"[Thinking]\n{thinking}\n\n[Reasoning]\n{reasoning}".strip()
    error_cat_raw = str(data.get("error_type", "")).strip()
    error_location = str(data.get("error_location", "N/A")).strip()
    corrected = str(data.get("corrected_statement", "N/A")).strip()

    if not error_cat_raw or error_cat_raw.lower() in ("n/a", "null", "none"):
        raise ValueError("Category classification returned empty or N/A category")

    normalized = normalize_category(error_cat_raw)
    if not normalized:
        raise ValueError(f"Unrecognized error category: '{error_cat_raw}'")

    if not error_location or error_location.lower() in ("n/a", "null", "none"):
        error_location = "N/A"

    if not corrected or corrected.lower() in ("n/a", "null", "none"):
        corrected = "N/A"

    return normalized, error_location, reasoning, corrected


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 (1-call): Direct classification parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_direct_classify(
    idx: str,
    raw_response: str,
) -> tuple[str, str]:
    """
    Parse a 1-call direct classification LLM response.

    Expected JSON: {"reasoning": "...", "error_type": "..."}

    Returns:
        (error_category, reasoning)
    """
    clean_text, thinking = strip_thinking(raw_response)
    json_text = _clean_json_text(_extract_json_text(clean_text))
    try:
        data = json.loads(json_text, strict=False)
    except json.JSONDecodeError as e:
        data = parse_loose_json(clean_text)
        if not data or "error_type" not in data:
            raise ValueError(f"Failed to decode direct classification JSON: {e}. Raw: {raw_response!r}") from e

    reasoning = str(data.get("reasoning", ""))
    if thinking:
        reasoning = f"[Thinking]\n{thinking}\n\n[Reasoning]\n{reasoning}".strip()
    error_cat_raw = str(data.get("error_type", "")).strip()

    if not error_cat_raw or error_cat_raw.lower() in ("n/a", "null", "none"):
        raise ValueError("Direct classification returned empty or N/A category")

    normalized = normalize_category(error_cat_raw)
    if not normalized:
        raise ValueError(f"Unrecognized error category: '{error_cat_raw}'")

    return normalized, reasoning
