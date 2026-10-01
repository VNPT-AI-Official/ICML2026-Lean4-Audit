#!/usr/bin/env python3

import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from client_example import search

NUM_WORKERS = 32
VERSIONS = ["v4.19.0", "v4.24.0", "v4.28.0"]

results = []
with open("../datasets/FormalRx_Test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            results.append(json.loads(line))


def process_sample(sample):
    best_hit = None
    best_version = None

    for version in VERSIONS:
        try:
            hits = search(
                sample["formal_statement"],
                top_k=1,
                version=version,
            )
            hit = hits[0] if hits else None
        except Exception as e:
            print(f"Error for {sample['idx']} version {version}: {e}")
            hit = None

        if hit and (best_hit is None or hit["score"] > best_hit["score"]):
            best_hit = hit
            best_version = version

    return {
        "id": sample["idx"],
        "header": sample["header"],
        "informal_statement": sample["informal_statement"],
        "formal_statement": sample["formal_statement"],
        "search_version": best_version,
        "search_score": best_hit["score"] if best_hit else None,
        "search_formal_name": best_hit["formal_name"] if best_hit else None,
        "search_informal_name": best_hit.get("informal_name") if best_hit else None,
        "search_kind": best_hit.get("kind") if best_hit else None,
        "search_type": best_hit.get("type") if best_hit else None,
        "search_informal_description": best_hit.get("informal_description") if best_hit else None,
        "search_path": best_hit.get("path") if best_hit else None,
    }


output = [None] * len(results)

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(process_sample, sample): i for i, sample in enumerate(results)}
    with tqdm(total=len(results), desc="Searching") as pbar:
        for future in as_completed(futures):
            i = futures[future]
            output[i] = future.result()
            pbar.update(1)

with open("../datasets/search_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Done. {len(output)} records written to search_results.json")