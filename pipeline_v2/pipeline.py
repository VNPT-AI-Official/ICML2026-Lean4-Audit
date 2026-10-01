"""
pipeline.py — V2 async FormalRx diagnostic pipeline.

Key improvements over V1:
  - Stage 1a: Verdict-only with 3× self-consistency majority vote
  - Stage 1b: Correction-only (runs only if Misaligned), uses Stage 1a reasoning
  - Dynamic few-shot: domain-based example selection from header imports

Usage:
    python pipeline.py [--config config.yaml] [--provider openai] [--limit N] [--dry-run]
"""

import argparse
import asyncio
import json
import os
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

# pyrefly: ignore [missing-import]
from openai import AsyncOpenAI, AsyncAzureOpenAI
from tqdm.asyncio import tqdm as atqdm

from parser import (
    Prediction,
    parse_verdict_only,
    parse_correction_only,
    parse_correction_localization,
    parse_verdict_correction,
    parse_localization_only,
    parse_group_classify,
    parse_category_classify,
    parse_direct_classify,
)
from prompt import (
    build_verdict_only_message,
    build_correction_only_message,
    build_correction_localization_message,
    build_localization_message,
    build_group_classify_message,
    build_category_classify_message,
    build_direct_classify_message,
    build_compiler_feedback_message,
)
from search_reference import load_search_references, get_search_reference
from taxonomy import SINGLE_CATEGORY_GROUPS
from lean_verifier import verify_lean_code
from postprocess import postprocess_prediction
from rule_classifier import classify_by_diff
from utils import (
    AsyncRateLimiter,
    append_checkpoint,
    get_provider_config,
    load_checkpoint,
    load_config,
    logger,
    setup_logging,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(config: dict) -> list[dict]:
    """Load FormalRx test samples from HuggingFace or local JSONL."""
    ds_config = config["dataset"]
    source = ds_config.get("source", "huggingface")

    if source == "local":
        local_path = ds_config.get("local_path")
        if not local_path or not Path(local_path).exists():
            raise FileNotFoundError(f"Local dataset file not found: {local_path}")
        samples = []
        with open(local_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        logger.info(f"Loaded {len(samples)} samples from local file: {local_path}")
    else:
        try:
            from datasets import load_dataset as hf_load_dataset
        except ImportError:
            raise ImportError("Install 'datasets' package: pip install datasets")

        repo = ds_config.get("hf_repo", "LARK-Lab/FormalRx-Test")
        split = ds_config.get("hf_split", "test")
        logger.info(f"Loading dataset from HuggingFace: {repo} / {split}")
        ds = hf_load_dataset(repo, split=split)
        samples = [dict(row) for row in ds]
        logger.info(f"Loaded {len(samples)} samples from HuggingFace")

    # Merge or parse lean_structure info
    lean_struct_cfg = config.get("lean_structure", {})
    if lean_struct_cfg.get("enabled", True):
        # 1. Try to load pre-processed enriched file
        enriched_path = lean_struct_cfg.get("enriched_file", "enriched_dataset.jsonl")
        enriched_map = {}
        if enriched_path and Path(enriched_path).exists():
            try:
                with open(enriched_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            if "idx" in data and "lean_structure" in data:
                                enriched_map[data["idx"]] = data["lean_structure"]
                logger.info(f"Loaded parsed Lean structures for {len(enriched_map)} samples from {enriched_path}")
            except Exception as e:
                logger.warning(f"Error loading enriched Lean structures from {enriched_path}: {e}")

        # 2. Populate lean_structure for each sample
        use_fallback = lean_struct_cfg.get("use_regex_fallback", True)
        fallback_count = 0
        from lean_parser import parse_lean_statement_regex

        for sample in samples:
            idx = sample.get("idx", "")
            if idx in enriched_map:
                sample["lean_structure"] = enriched_map[idx]
            elif use_fallback:
                try:
                    struct = parse_lean_statement_regex(sample.get("formal_statement", ""))
                    sample["lean_structure"] = struct.to_dict()
                    fallback_count += 1
                except Exception as e:
                    logger.warning(f"Failed fallback parsing for {idx}: {e}")
                    sample["lean_structure"] = {}
            else:
                sample["lean_structure"] = {}

        if fallback_count > 0:
            logger.info(f"Dynamically parsed Lean structures using regex fallback for {fallback_count} samples")

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# API client factory
# ─────────────────────────────────────────────────────────────────────────────

def make_client(provider_config: dict):
    name = provider_config.get("provider_name", "openai")
    if name == "azure":
        return AsyncAzureOpenAI(
            azure_endpoint=provider_config["base_url"],
            api_key=provider_config["api_key"],
            api_version=provider_config.get("api_version", "2024-02-01"),
        )
    else:
        return AsyncOpenAI(
            base_url=provider_config.get("base_url"),
            api_key=provider_config["api_key"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# API call helper
# ─────────────────────────────────────────────────────────────────────────────

async def call_api(
    client,
    messages: list[dict],
    model: str,
    gen_config: dict,
    max_tokens_override: int | None = None,
    temperature_override: float | None = None,
) -> str:
    """Make one API call and return the raw response text."""
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature_override if temperature_override is not None else gen_config.get("temperature", 0.0),
        "max_tokens": max_tokens_override or gen_config.get("max_tokens", 2048),
    }

    if gen_config.get("response_format") == "json_object":
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
    except Exception as e:
        error_str = str(e).lower()
        if "structured-outputs" in error_str or "response_format" in error_str:
            if "response_format" in kwargs:
                del kwargs["response_format"]
                response = await client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
        raise e


# ─────────────────────────────────────────────────────────────────────────────
# Self-consistency majority vote
# ─────────────────────────────────────────────────────────────────────────────

def majority_vote(
    votes: list[tuple[str, str]],
    threshold: int = 2,
    strategy: str = "majority",
) -> tuple[str, str, list[str]]:
    """
    Perform majority vote on verdict results.

    Args:
        votes: list of (verdict, reasoning) tuples
        threshold: minimum votes to win (used when strategy="majority")
        strategy: "majority" = need threshold votes to win,
                  "any_misaligned" = 1 Misaligned vote → Misaligned

    Returns:
        (final_verdict, combined_reasoning, all_reasonings)
    """
    verdict_counts = Counter(v for v, _ in votes)
    all_reasonings = [r for _, r in votes]

    # Determine winner
    misaligned_count = verdict_counts.get("Misaligned", 0)
    aligned_count = verdict_counts.get("Aligned", 0)

    if strategy == "any_misaligned":
        # Any single Misaligned vote → Misaligned (max recall)
        if misaligned_count >= 1:
            final_verdict = "Misaligned"
            misaligned_votes = [r for v, r in votes if v == "Misaligned"]
            combined = misaligned_votes[0] if misaligned_votes else ""
        else:
            final_verdict = "Aligned"
            combined = all_reasonings[0] if all_reasonings else ""
    else:
        # Default: majority threshold
        if misaligned_count >= threshold:
            final_verdict = "Misaligned"
            misaligned_votes = [r for v, r in votes if v == "Misaligned"]
            combined = misaligned_votes[0] if misaligned_votes else ""
        elif aligned_count >= threshold:
            final_verdict = "Aligned"
            aligned_votes = [r for v, r in votes if v == "Aligned"]
            combined = aligned_votes[0] if aligned_votes else ""
        else:
            # No clear majority — default to Misaligned (higher recall)
            final_verdict = "Misaligned"
            combined = all_reasonings[0] if all_reasonings else ""

    vote_summary = f"[SC {misaligned_count}M/{aligned_count}A/{len(votes)}T]"
    combined = f"{vote_summary} {combined}"

    return final_verdict, combined, all_reasonings


def majority_vote_category(
    votes: list[tuple[str, str]],
) -> tuple[str, str]:
    """
    Perform majority vote on category classification results.

    Args:
        votes: list of (category, reasoning) tuples

    Returns:
        (winning_category, reasoning_from_winner)
    """
    category_counts: Counter = Counter(cat for cat, _ in votes)
    # Pick the most common category; on ties take the first one encountered
    winning_cat = category_counts.most_common(1)[0][0]
    winning_reasoning = next((r for c, r in votes if c == winning_cat), "")
    n = len(votes)
    summary = f"[SC2 {winning_cat} {category_counts[winning_cat]}/{n}T]"
    return winning_cat, f"{summary} {winning_reasoning}"


# ─────────────────────────────────────────────────────────────────────────────
# Two-stage sample processor (V2: self-consistency + split stages)
# ─────────────────────────────────────────────────────────────────────────────

async def process_sample_v2(
    sample: dict,
    clients: dict,
    models: dict,
    gen_config: dict,
    sc_config: dict,
    semaphore: asyncio.Semaphore,
    rate_limiter: AsyncRateLimiter,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    retry_backoff: float = 2.0,
    use_compiler: bool = True,
    use_rule_classifier: bool = True,
    few_shot_config: dict = None,
    postprocess_config: dict = None,
    category_stages: int = 1,
) -> Prediction:
    """
    V2 two-stage processing with self-consistency + search reference:
      Stage 1a: Verdict × N votes (with search ref) → majority vote
      Stage 1b: Correction + Localization merged (with search ref)
      Stage 2: Error categorization (with search ref)
        - 2a: Rule-based classify
        - 2b: (skipped — localization from Stage 1b)
        - 2c: Group → Category (with search ref)
    """
    idx = sample.get("idx", "unknown")
    n_votes = sc_config.get("stage1a_n_votes", 3)
    sc_temperature = sc_config.get("stage1a_temperature", 0.7)
    vote_threshold = sc_config.get("verdict_threshold", 2)

    # Get search reference for this sample
    search_ref = get_search_reference(idx)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1a: Verdict (N× self-consistency, with search ref)
    # ══════════════════════════════════════════════════════════════════════════
    verdict_messages = build_verdict_only_message(
        sample,
        use_few_shot=gen_config.get("use_few_shot", True),
        few_shot_count=gen_config.get("few_shot_count", 2),
        search_ref=search_ref,
    )
    stage1_max_tokens = gen_config.get("stage1_max_tokens", 1024)
    stage2_max_tokens = gen_config.get("stage2_max_tokens", 1024)

    async def _single_verdict_call() -> tuple[str, str]:
        """Make a single verdict API call with retries."""
        delay = retry_delay
        for attempt in range(1, max_retries + 1):
            try:
                await rate_limiter.acquire()
                raw = await call_api(
                    clients["verdict"], verdict_messages, models["verdict"], gen_config,
                    max_tokens_override=stage1_max_tokens,
                    temperature_override=sc_temperature,
                )
                verdict, reasoning = parse_verdict_only(idx, raw)
                return verdict, reasoning
            except Exception as e:
                logger.warning(f"[{idx}] Stage1a vote API error attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(delay)
                    delay *= retry_backoff
        return "Misaligned", ""  # Fallback: default to misaligned

    # Run N votes concurrently within semaphore
    async with semaphore:
        vote_tasks = [_single_verdict_call() for _ in range(n_votes)]
        votes = await asyncio.gather(*vote_tasks)

    verdict_strategy = sc_config.get("verdict_strategy", "majority")
    final_verdict, combined_reasoning, all_reasonings = majority_vote(
        list(votes), threshold=vote_threshold, strategy=verdict_strategy,
    )
    logger.debug(f"[{idx}] Stage1a → {final_verdict} (votes: {[v for v, _ in votes]})")

    # ── Aligned → done ────────────────────────────────────────────────────────
    if final_verdict == "Aligned":
        return Prediction(
            idx=idx, aligned="Aligned",
            error_category="N/A", error_segment="N/A",
            corrected_statement="N/A",
            reasoning=combined_reasoning,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1b: Correction + Localization (merged, with search ref)
    # ══════════════════════════════════════════════════════════════════════════
    correction_messages = build_correction_localization_message(
        sample, search_ref=search_ref,
    )
    stage1b_max_tokens = gen_config.get("stage1b_max_tokens", 2048)

    corrected = "N/A"
    error_segment = "N/A"
    delay = retry_delay
    async with semaphore:
        for attempt in range(1, max_retries + 1):
            try:
                # Bump temperature slightly on retries to get varied responses
                temp_override = 0.0 if attempt == 1 else min(0.2 * attempt, 0.7)
                await rate_limiter.acquire()
                raw_corr = await call_api(
                    clients["correction"], correction_messages, models["correction"], gen_config,
                    max_tokens_override=stage1b_max_tokens,
                    temperature_override=temp_override,
                )
                parsed_corr, loc_from_1b, corr_reasoning = parse_correction_localization(idx, raw_corr)

                # Accumulate reasoning from each attempt
                if corr_reasoning:
                    combined_reasoning += f" [Correction+Loc] {corr_reasoning}"

                # Update best results (keep best non-N/A values across attempts)
                if parsed_corr and parsed_corr not in ("N/A", "", None):
                    corrected = parsed_corr
                if loc_from_1b and loc_from_1b not in ("N/A", "", None):
                    error_segment = loc_from_1b

                # Validate: misaligned must have a valid correction
                if corrected in ("N/A", "", None):
                    logger.warning(
                        f"[{idx}] Stage1b attempt {attempt}/{max_retries}: "
                        f"correction is N/A (parse returned empty), retrying..."
                    )
                    if attempt < max_retries:
                        await asyncio.sleep(delay)
                        delay *= retry_backoff
                    continue  # retry — don't break on N/A

                break  # success — got a valid correction

            except Exception as e:
                logger.warning(f"[{idx}] Stage1b API error attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(delay)
                    delay *= retry_backoff

    # Post-loop validation: warn loudly if correction/segment still N/A
    if corrected in ("N/A", "", None):
        logger.warning(
            f"[{idx}] Stage1b FAILED after {max_retries} attempts: "
            f"no valid correction produced for Misaligned sample"
        )
    if error_segment in ("N/A", "", None):
        logger.warning(
            f"[{idx}] Stage1b: error_segment is N/A after {max_retries} attempts"
        )

    logger.debug(f"[{idx}] Stage1b → correction {'OK' if corrected != 'N/A' else 'FAILED'}, loc={'OK' if error_segment != 'N/A' else 'FAILED'}")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2: Error Categorization (with search ref)
    # ══════════════════════════════════════════════════════════════════════════
    error_category = "N/A"
    stage2_reasoning = ""
    rule_detected = False

    # ── Stage 2a: Rule-based diff classification ──────────────────────────────
    if use_rule_classifier and corrected not in ("N/A", "", None):
        try:
            rule_result = classify_by_diff(
                informal_statement=sample.get("informal_statement", ""),
                formal_statement=sample.get("formal_statement", ""),
                corrected_statement=corrected,
            )
            if rule_result and rule_result.category:
                error_category = rule_result.category
                rule_detected = True
                stage2_reasoning = (
                    f"[Rule {rule_result.rule_id}] [{rule_result.confidence}] "
                    f"{rule_result.evidence}"
                )
                logger.debug(f"[{idx}] Stage2a rule: {error_category} via {rule_result.rule_id}")
        except Exception as e:
            logger.debug(f"[{idx}] Stage2a rule error (non-fatal): {e}")

    if rule_detected:
        # Stage 2b is SKIPPED — localization already obtained from Stage 1b
        logger.debug(f"[{idx}] Stage2 rule detected, using localization from Stage 1b")
    else:
        # ── Stage 2c: LLM classification ──────────────────────────────────────
        logger.debug(f"[{idx}] Stage2a no rule → LLM classify (stages={category_stages})")

        if category_stages == 1:
            # ── 1-call direct classification (with optional self-consistency) ──
            exclude_cats = (postprocess_config or {}).get("exclude_categories", [])
            direct_messages = build_direct_classify_message(
                sample,
                use_few_shot=gen_config.get("use_few_shot", True),
                few_shot_count=gen_config.get("few_shot_count", 2),
                exclude_categories=exclude_cats,
                search_ref=search_ref,
            )
            n_cat_votes = sc_config.get("stage2_n_votes", 1)
            cat_sc_temperature = sc_config.get("stage2_temperature", 0.5)

            async def _single_category_call() -> tuple[str, str]:
                """Make a single category API call with retries."""
                _delay = retry_delay
                for attempt in range(1, max_retries + 1):
                    try:
                        await rate_limiter.acquire()
                        raw_direct = await call_api(
                            clients["category"], direct_messages, models["category"], gen_config,
                            max_tokens_override=stage2_max_tokens,
                            temperature_override=cat_sc_temperature if n_cat_votes > 1 else None,
                        )
                        cat, cat_reasoning = parse_direct_classify(idx, raw_direct)
                        return cat, cat_reasoning
                    except Exception as e:
                        logger.warning(f"[{idx}] Stage2c direct API error attempt {attempt}/{max_retries}: {e}")
                        if attempt < max_retries:
                            await asyncio.sleep(_delay)
                            _delay *= retry_backoff
                return "N/A", ""

            if n_cat_votes > 1:
                async with semaphore:
                    cat_vote_tasks = [_single_category_call() for _ in range(n_cat_votes)]
                    cat_votes = await asyncio.gather(*cat_vote_tasks)
                error_category, stage2_reasoning = majority_vote_category(list(cat_votes))
                stage2_reasoning = f"[Direct-SC] {stage2_reasoning}"
                logger.debug(f"[{idx}] Stage2c SC votes: {[c for c, _ in cat_votes]} → {error_category}")
            else:
                delay = retry_delay
                async with semaphore:
                    for attempt in range(1, max_retries + 1):
                        try:
                            await rate_limiter.acquire()
                            raw_direct = await call_api(
                                clients["category"], direct_messages, models["category"], gen_config,
                                max_tokens_override=stage2_max_tokens,
                            )
                            cat, cat_reasoning = parse_direct_classify(idx, raw_direct)
                            error_category = cat
                            stage2_reasoning = f"[Direct] {cat_reasoning}"
                            break
                        except Exception as e:
                            logger.warning(f"[{idx}] Stage2c direct API error attempt {attempt}/{max_retries}: {e}")
                            if attempt < max_retries:
                                await asyncio.sleep(delay)
                                delay *= retry_backoff
        else:
            # Call 1: Group Classification
            exclude_cats = (postprocess_config or {}).get("exclude_categories", [])
            group_messages = build_group_classify_message(
                sample,
                use_few_shot=gen_config.get("use_few_shot", True),
                few_shot_count=gen_config.get("few_shot_count", 2),
                exclude_categories=exclude_cats,
                search_ref=search_ref,
            )
            delay = retry_delay
            group_name = "N/A"
            async with semaphore:
                for attempt in range(1, max_retries + 1):
                    try:
                        await rate_limiter.acquire()
                        raw_group = await call_api(
                            clients["category"], group_messages, models["category"], gen_config,
                            max_tokens_override=stage2_max_tokens,
                        )
                        group_name, grp_reasoning = parse_group_classify(idx, raw_group)
                        stage2_reasoning = f"[Group] {group_name} {grp_reasoning}"
                        break
                    except Exception as e:
                        logger.warning(f"[{idx}] Stage2c-1 API error attempt {attempt}/{max_retries}: {e}")
                        if attempt < max_retries:
                            await asyncio.sleep(delay)
                            delay *= retry_backoff

            # Check single-category groups
            if group_name in SINGLE_CATEGORY_GROUPS:
                error_category = SINGLE_CATEGORY_GROUPS[group_name]
                stage2_reasoning += f" [Single-cat → {error_category}]"
                # Localization already from Stage 1b — no extra call needed
            elif group_name != "N/A":
                # Call 2: Category Classification (with search ref)
                cat_messages = build_category_classify_message(
                    sample, group_name,
                    use_few_shot=gen_config.get("use_few_shot", True),
                    few_shot_count=gen_config.get("few_shot_count", 2),
                    exclude_categories=exclude_cats,
                    search_ref=search_ref,
                )
                delay = retry_delay
                async with semaphore:
                    for attempt in range(1, max_retries + 1):
                        try:
                            await rate_limiter.acquire()
                            raw_cat = await call_api(
                                clients["category"], cat_messages, models["category"], gen_config,
                                max_tokens_override=stage2_max_tokens,
                            )
                            cat, loc, cat_reasoning, cat_corr = parse_category_classify(idx, raw_cat)
                            error_category = cat
                            if loc and loc != "N/A":
                                error_segment = loc  # Override with category-stage localization
                            if corrected in ("N/A", "", None) and cat_corr not in ("N/A", "", None):
                                corrected = cat_corr
                            stage2_reasoning += f" [Cat] {cat_reasoning}"
                            break
                        except Exception as e:
                            logger.warning(f"[{idx}] Stage2c-2 API error attempt {attempt}/{max_retries}: {e}")
                            if attempt < max_retries:
                                await asyncio.sleep(delay)
                                delay *= retry_backoff

    # ── Final validation: misaligned must have all fields ───────────────────────
    _na_fields = []
    if error_category in ("N/A", "", None):
        _na_fields.append("error_category")
    if error_segment in ("N/A", "", None):
        _na_fields.append("error_segment")
    if corrected in ("N/A", "", None):
        _na_fields.append("corrected_statement")
    if _na_fields:
        logger.warning(
            f"[{idx}] Misaligned sample has N/A fields after all stages: "
            f"{', '.join(_na_fields)}"
        )

    # ── Assemble final prediction ─────────────────────────────────────────────
    if stage2_reasoning:
        combined_reasoning += f" {stage2_reasoning}"

    pred = Prediction(
        idx=idx,
        aligned="Misaligned",
        error_category=error_category,
        error_segment=error_segment,
        corrected_statement=corrected,
        reasoning=combined_reasoning,
    )

    # ── Lean 4 Compiler Feedback Loop ─────────────────────────────────────────
    if use_compiler and pred.corrected_statement not in ("N/A", "", None):
        max_comp_retries = 3
        compiler_msgs = correction_messages.copy()
        compiler_msgs.append({"role": "assistant", "content": json.dumps({"corrected_statement": pred.corrected_statement})})

        for comp_attempt in range(1, max_comp_retries + 1):
            is_valid, err_output = await verify_lean_code(pred.corrected_statement)
            if is_valid:
                pred.reasoning += " [Compiler] Passed."
                break
            pred.reasoning += f" [Compiler] Failed ({comp_attempt})."
            compiler_msgs = build_compiler_feedback_message(compiler_msgs, err_output)
            try:
                await rate_limiter.acquire()
                raw_fix = await call_api(clients["correction"], compiler_msgs, models["correction"], gen_config)
                fix_corr, _ = parse_correction_only(idx, raw_fix)
                if fix_corr not in ("N/A", "", None):
                    pred.corrected_statement = fix_corr
                compiler_msgs.append({"role": "assistant", "content": raw_fix})
            except Exception as e:
                logger.warning(f"[{idx}] Compiler feedback API error: {e}")
                break

    # ── Post-Processing Layer ─────────────────────────────────────────────────
    pred = postprocess_prediction(pred, sample, postprocess_config)

    return pred


# ─────────────────────────────────────────────────────────────────────────────
# Stats tracking
# ─────────────────────────────────────────────────────────────────────────────

class Stats:
    def __init__(self):
        self.total = 0
        self.aligned = 0
        self.misaligned = 0
        self.parse_errors = 0
        self.api_failures = 0
        self.start_time = time.time()

    def update(self, pred: Prediction):
        self.total += 1
        if pred.aligned == "Aligned":
            self.aligned += 1
        else:
            self.misaligned += 1
        if pred.parse_error:
            self.parse_errors += 1
        if "API failure" in (pred.parse_error or ""):
            self.api_failures += 1

    def summary(self) -> str:
        elapsed = time.time() - self.start_time
        rate = self.total / elapsed if elapsed > 0 else 0
        return (
            f"Processed: {self.total} | "
            f"Aligned: {self.aligned} ({self.aligned/max(1,self.total)*100:.1f}%) | "
            f"Misaligned: {self.misaligned} | "
            f"Parse errors: {self.parse_errors} | "
            f"API failures: {self.api_failures} | "
            f"Speed: {rate:.2f} samples/s | "
            f"Elapsed: {elapsed:.0f}s"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_predictions(predictions: list[Prediction], output_file: str, save_reasoning: bool = True):
    sub_path = Path(output_file)
    debug_path = sub_path.with_suffix(".debug.jsonl")
    with open(sub_path, "w", encoding="utf-8") as f_sub, \
         open(debug_path, "w", encoding="utf-8") as f_dbg:
        for pred in predictions:
            f_sub.write(json.dumps(pred.to_submission_dict(), ensure_ascii=False) + "\n")
            if save_reasoning:
                dbg_dict = pred.to_submission_dict()
                dbg_dict["reasoning"] = pred.reasoning
                dbg_dict["parse_error"] = pred.parse_error
                f_dbg.write(json.dumps(dbg_dict, ensure_ascii=False) + "\n")
    logger.info(f"Predictions saved to: {sub_path} ({len(predictions)} lines)")


def create_submission_zip(predictions_file: str, zip_file: str):
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(predictions_file, arcname=Path(predictions_file).name)
    logger.info(f"Submission zip created: {zip_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main(args):
    config = load_config(args.config)
    setup_logging(config["output"].get("log_file", "pipeline.log"))

    if args.provider:
        config["active_provider"] = args.provider

    active_prov_cfg = get_provider_config(config)
    gen_config = config["generation"]
    pipeline_cfg = config["pipeline"]
    output_cfg = config["output"]
    arch_cfg = config.get("architecture", {})
    sc_config = config.get("self_consistency", {
        "stage1a_n_votes": 3,
        "stage1a_temperature": 0.7,
        "verdict_threshold": 2,
    })
    stage_providers = config.get("stage_providers", {})

    verdict_prov_name = stage_providers.get("verdict")
    verdict_prov_cfg = get_provider_config(config, verdict_prov_name) if verdict_prov_name else active_prov_cfg

    correction_prov_name = stage_providers.get("correction")
    correction_prov_cfg = get_provider_config(config, correction_prov_name) if correction_prov_name else active_prov_cfg

    category_prov_name = stage_providers.get("category")
    category_prov_cfg = get_provider_config(config, category_prov_name) if category_prov_name else active_prov_cfg

    client_cache = {}
    for p_cfg in [verdict_prov_cfg, correction_prov_cfg, category_prov_cfg]:
        p_name = p_cfg["provider_name"]
        if p_name not in client_cache:
            client_cache[p_name] = make_client(p_cfg)

    clients = {
        "verdict": client_cache[verdict_prov_cfg["provider_name"]],
        "correction": client_cache[correction_prov_cfg["provider_name"]],
        "category": client_cache[category_prov_cfg["provider_name"]],
    }

    models = {
        "verdict": verdict_prov_cfg["model"],
        "correction": correction_prov_cfg["model"],
        "category": category_prov_cfg["model"],
    }

    use_rule_classifier = arch_cfg.get("use_rule_classifier", True)
    category_stages = arch_cfg.get("category_stages", 1)
    n_votes = sc_config.get("stage1a_n_votes", 3)

    logger.info(
        f"Starting V2 pipeline | "
        f"verdict={verdict_prov_cfg['provider_name']}:{models['verdict']} | "
        f"correction={correction_prov_cfg['provider_name']}:{models['correction']} | "
        f"category={category_prov_cfg['provider_name']}:{models['category']} | "
        f"self_consistency={n_votes}× | rule_classifier={use_rule_classifier} | "
        f"category_stages={category_stages}"
    )

    # ── Load dataset ─────────────────────────────────────────────────────────
    samples = load_dataset(config)
    if args.limit:
        samples = samples[:args.limit]
        logger.info(f"Limited to {len(samples)} samples")

    # ── Load search references ────────────────────────────────────────────────
    search_ref_cfg = config.get("search_reference", {})
    if search_ref_cfg.get("enabled", True):
        search_file = search_ref_cfg.get("ground_truth_file", "../datasets/search_results.json")
        score_threshold = search_ref_cfg.get("score_threshold", 0.92)
        load_search_references(search_file, score_threshold=score_threshold)

    checkpoint_file = output_cfg.get("checkpoint_file", ".checkpoint.jsonl")
    done_ids: set[str] = set()
    if args.resume:
        done_ids = load_checkpoint(checkpoint_file)
    else:
        open(checkpoint_file, 'w').close()

    pending = [s for s in samples if s.get("idx", "") not in done_ids]
    logger.info(f"Total: {len(samples)} | Done: {len(done_ids)} | Pending: {len(pending)}")

    if not pending:
        logger.info("All samples already processed! Rebuilding from checkpoint.")
    else:
        semaphore = asyncio.Semaphore(pipeline_cfg.get("max_concurrency", 8))
        rate_limiter = AsyncRateLimiter(max_requests_per_minute=300)
        stats = Stats()

        logger.info("Using V2 architecture (Stage1a: verdict×N + Stage1b: correction + Stage2: classify)")

        tasks = [
            process_sample_v2(
                s, clients=clients, models=models, gen_config=gen_config,
                sc_config=sc_config, semaphore=semaphore, rate_limiter=rate_limiter,
                max_retries=pipeline_cfg.get("max_retries", 3),
                retry_delay=pipeline_cfg.get("retry_delay", 5.0),
                retry_backoff=pipeline_cfg.get("retry_backoff", 2.0),
                use_compiler=arch_cfg.get("use_lean_compiler", True),
                use_rule_classifier=use_rule_classifier,
                few_shot_config=config.get("few_shot", {}),
                postprocess_config=config.get("postprocess", {}),
                category_stages=category_stages,
            )
            for s in pending
        ]

        completed = 0
        for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Diagnosing V2", unit="sample"):
            pred = await coro
            stats.update(pred)
            append_checkpoint(checkpoint_file, pred.to_dict())
            completed += 1
            if completed % 100 == 0:
                logger.info(stats.summary())

        logger.info(f"\nFinal stats: {stats.summary()}")

    # ── Rebuild full predictions from checkpoint ─────────────────────────────
    pred_dict: dict[str, Prediction] = {}
    idx_to_order = {s.get("idx", ""): i for i, s in enumerate(samples)}

    if Path(checkpoint_file).exists():
        with open(checkpoint_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                idx = obj.get("idx", "")
                pred_dict[idx] = Prediction.from_dict(obj)

    all_predictions = list(pred_dict.values())
    all_predictions.sort(key=lambda p: idx_to_order.get(p.idx, 99999))

    # ── Save outputs ──────────────────────────────────────────────────────────
    predictions_file = args.output or output_cfg.get("predictions_file", "predictions.jsonl")
    save_predictions(all_predictions, predictions_file, save_reasoning=output_cfg.get("save_reasoning", True))
    create_submission_zip(predictions_file, output_cfg.get("submission_zip", "submission.zip"))
    logger.info("Pipeline V2 complete. Upload submission.zip to Codabench.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FormalRx V2 pipeline (self-consistency)")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--provider", default=None, help="Override provider")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N samples")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt and exit")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--output", default=None, help="Override output file")
    args = parser.parse_args()

    asyncio.run(main(args))
