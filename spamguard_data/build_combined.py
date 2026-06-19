from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl, template_hash, text_hash, write_csv
from .manifest import load_manifest


OUT_DIR = Path("data/combined")


def load_allowed_rows(mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = load_manifest()
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for source in manifest.get("sources", []):
        allowed = source.get("allowed_use", {})
        if mode == "training" and not allowed.get("training"):
            skipped.append({"source_name": source.get("source_name"), "reason": "allow_training_false"})
            continue
        if mode == "evaluation" and not allowed.get("evaluation"):
            skipped.append({"source_name": source.get("source_name"), "reason": "allow_evaluation_false"})
            continue
        source_rows = read_jsonl(Path(source.get("processed_path", "")))
        for row in source_rows:
            if mode == "training" and not row.get("allow_training"):
                continue
            if mode == "evaluation" and not row.get("allow_evaluation"):
                continue
            if row.get("label") not in {"HAM", "SPAM", "PHISHING"}:
                continue
            rows.append(row)
    return rows, skipped


def dedup_and_cluster(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[int]], dict[str, Any]]:
    seen_texts: set[str] = set()
    clean_rows: list[dict[str, Any]] = []
    removed_exact = 0
    for row in rows:
        h = text_hash(row.get("text", ""))
        if h in seen_texts:
            removed_exact += 1
            continue
        seen_texts.add(h)
        clean_rows.append(row)

    clusters: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(clean_rows):
        clusters[template_hash(row.get("text", ""))].append(idx)

    report = {
        "input_rows": len(rows),
        "rows_after_exact_dedup": len(clean_rows),
        "removed_exact_duplicates": removed_exact,
        "template_clusters": len(clusters),
        "repeated_template_clusters": sum(1 for members in clusters.values() if len(members) > 1),
        "rows_in_repeated_template_clusters": sum(len(members) for members in clusters.values() if len(members) > 1),
    }
    return clean_rows, clusters, report


def cluster_signature(rows: list[dict[str, Any]], members: list[int]) -> tuple[str, str]:
    labels = Counter(rows[idx].get("label") for idx in members)
    sources = Counter(rows[idx].get("source") for idx in members)
    return labels.most_common(1)[0][0], sources.most_common(1)[0][0]


def split_clusters(rows: list[dict[str, Any]], clusters: dict[str, list[int]], seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[list[int]]] = defaultdict(list)
    for members in clusters.values():
        grouped[cluster_signature(rows, members)].append(members)

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for cluster_list in grouped.values():
        rng.shuffle(cluster_list)
        n = len(cluster_list)
        val_cut = max(1, int(n * 0.15)) if n >= 7 else 0
        test_cut = max(1, int(n * 0.15)) if n >= 7 else 0
        for idx, members in enumerate(cluster_list):
            if idx < val_cut:
                split = "validation"
            elif idx < val_cut + test_cut:
                split = "test"
            else:
                split = "train"
            for member in members:
                row = dict(rows[member])
                row["clean_split"] = split
                split_rows[split].append(row)
    return split_rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "labels": dict(Counter(row.get("label") for row in rows)),
        "sources": dict(Counter(row.get("source") for row in rows)),
        "languages": dict(Counter(row.get("language") for row in rows)),
        "hard_ham": sum(1 for row in rows if row.get("label") == "HAM" and row.get("category_tags")),
        "links": sum(1 for row in rows if row.get("contains_link")),
        "phones": sum(1 for row in rows if row.get("contains_phone")),
    }


def write_split_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "text",
        "label",
        "source",
        "clean_split",
        "split_origin",
        "language",
        "sample_origin",
        "allow_training",
        "allow_evaluation",
        "license_status",
        "contains_link",
        "contains_phone",
        "message_length",
        "category_tags",
        "metadata",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field), ensure_ascii=False) if isinstance(row.get(field), (list, dict)) else row.get(field)
                    for field in fields
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean combined SpamGuard data files without training.")
    parser.add_argument("--seed", type=int, default=20260619)
    args = parser.parse_args()

    training_rows, skipped_training = load_allowed_rows("training")
    train_clean_rows, train_clusters, train_dedup_report = dedup_and_cluster(training_rows)
    split_rows = split_clusters(train_clean_rows, train_clusters, args.seed)

    evaluation_rows, skipped_eval = load_allowed_rows("evaluation")
    eval_clean_rows, eval_clusters, eval_dedup_report = dedup_and_cluster(evaluation_rows)
    hebrew_eval = [row for row in eval_clean_rows if row.get("source") == "hebrew_eval"]
    auxiliary_eval = [row for row in eval_clean_rows if row.get("source") != "hebrew_eval" and not row.get("allow_training")]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        write_split_csv(OUT_DIR / f"{split}.csv", rows)
    write_split_csv(OUT_DIR / "hebrew_eval.csv", hebrew_eval)
    write_split_csv(OUT_DIR / "auxiliary_evaluation_only.csv", auxiliary_eval)

    report = {
        "do_not_train_yet": True,
        "training_candidate_inputs": train_dedup_report,
        "evaluation_candidate_inputs": eval_dedup_report,
        "split_summary": {split: summarize(rows) for split, rows in split_rows.items()},
        "hebrew_eval_summary": summarize(hebrew_eval),
        "auxiliary_evaluation_only_summary": summarize(auxiliary_eval),
        "skipped_training_sources": skipped_training,
        "skipped_evaluation_sources": skipped_eval,
        "split_policy": "Exact duplicate removal, then template-hash clusters split as whole clusters. Synthetic Hebrew remains separate evaluation-only.",
        "recommendation": "Review manifest/license flags and quality report before enabling any conditional source for training.",
    }
    with (OUT_DIR / "data_report.json").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_csv(OUT_DIR / "split_summary.csv", [{"split": split, **summarize(rows)} for split, rows in split_rows.items()])
    print(json.dumps({"combined_dir": str(OUT_DIR), "report": str(OUT_DIR / "data_report.json")}, indent=2))


if __name__ == "__main__":
    main()
