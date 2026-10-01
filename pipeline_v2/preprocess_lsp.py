"""
preprocess_lsp.py — Batch pre-processor to enrich FormalRx dataset with
parsed Lean structure information.

Usage:
    python preprocess_lsp.py [--input INPUT_JSONL] [--output OUTPUT_JSONL] [--source huggingface|local]

Reads the dataset (from HuggingFace or local JSONL), parses each formal_statement
using the regex-based Lean parser, and writes an enriched JSONL with a
'lean_structure' field added to each sample.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from lean_parser import parse_lean_statement_regex


def load_dataset_samples(source: str, local_path: str | None = None) -> list[dict]:
    """Load dataset samples from HuggingFace or local JSONL."""
    if source == "huggingface":
        from datasets import load_dataset
        ds = load_dataset("LARK-Lab/FormalRx-Test", split="test")
        return [dict(ds[i]) for i in range(len(ds))]
    elif source == "local" and local_path:
        samples = []
        with open(local_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples
    else:
        raise ValueError(f"Unknown source: {source}")


def enrich_sample(sample: dict) -> dict:
    """Parse the formal_statement and add lean_structure field."""
    formal = sample.get("formal_statement", "")
    if not formal:
        sample["lean_structure"] = {}
        return sample

    structure = parse_lean_statement_regex(formal)
    sample["lean_structure"] = structure.to_dict()
    return sample


def main():
    parser = argparse.ArgumentParser(description="Enrich FormalRx dataset with Lean structure")
    parser.add_argument("--source", default="local", choices=["huggingface", "local"],
                        help="Dataset source")
    parser.add_argument("--input", default="../datasets/FormalRx_Test.jsonl",
                        help="Local input JSONL path (for source=local)")
    parser.add_argument("--output", default="../datasets/enriched_dataset.jsonl",
                        help="Output enriched JSONL path")
    parser.add_argument("--stats", action="store_true", help="Print parsing statistics")
    args = parser.parse_args()

    print(f"Loading dataset from {args.source}...")
    samples = load_dataset_samples(args.source, args.input)
    print(f"Loaded {len(samples)} samples")

    t0 = time.time()
    enriched = []
    stats = {
        "total": len(samples),
        "has_name": 0,
        "has_goal": 0,
        "has_hypotheses": 0,
        "has_quantifiers": 0,
    }

    for sample in samples:
        enriched_sample = enrich_sample(sample)
        enriched.append(enriched_sample)

        ls = enriched_sample.get("lean_structure", {})
        if ls.get("theorem_name"):
            stats["has_name"] += 1
        if ls.get("goal"):
            stats["has_goal"] += 1
        if ls.get("hypotheses"):
            stats["has_hypotheses"] += 1
        if ls.get("quantifiers"):
            stats["has_quantifiers"] += 1

    elapsed = time.time() - t0

    # Write output
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        for sample in enriched:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(enriched)} enriched samples to {output_path}")
    print(f"Time: {elapsed:.2f}s ({elapsed/len(samples)*1000:.2f}ms/sample)")

    if args.stats:
        print(f"\n--- Parsing Statistics ---")
        for key, val in stats.items():
            if key == "total":
                print(f"  Total: {val}")
            else:
                pct = 100 * val / stats["total"] if stats["total"] > 0 else 0
                print(f"  {key}: {val}/{stats['total']} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
