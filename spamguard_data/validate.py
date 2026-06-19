from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import VALID_LABELS, read_jsonl, template_hash, text_hash, write_csv
from .manifest import load_manifest


REPORT_DIR = Path("data/reports")


def load_rows(include_empty_sources: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = load_manifest()
    rows: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    for source in manifest.get("sources", []):
        path = Path(source.get("processed_path", ""))
        source_rows = read_jsonl(path)
        if source_rows or include_empty_sources:
            source_summaries.append(
                {
                    "source_name": source.get("source_name"),
                    "processed_path": str(path),
                    "rows": len(source_rows),
                    "allowed_use": source.get("allowed_use"),
                    "license_status": source.get("license_status"),
                }
            )
        rows.extend(source_rows)
    return rows, source_summaries


def length_bucket(length: int) -> str:
    if length <= 0:
        return "empty"
    if length <= 20:
        return "0-20w"
    if length <= 80:
        return "21-80w"
    if length <= 200:
        return "81-200w"
    if length <= 800:
        return "201-800w"
    return "800w+"


def summarize_counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "UNKNOWN")) for row in rows).items()))


def nested_count(rows: list[dict[str, Any]], keys: tuple[str, str]) -> dict[str, dict[str, int]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        out[str(row.get(keys[0], "UNKNOWN"))][str(row.get(keys[1], "UNKNOWN"))] += 1
    return {key: dict(counter) for key, counter in sorted(out.items())}


def validate_rows(rows: list[dict[str, Any]], source_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    hashes = [text_hash(row.get("text", "")) for row in rows]
    templates = [template_hash(row.get("text", "")) for row in rows]
    hash_counts = Counter(hashes)
    template_counts = Counter(templates)

    invalid_rows = []
    for idx, row in enumerate(rows):
        issues = []
        if not str(row.get("text", "")).strip():
            issues.append("missing_text")
        if row.get("label") not in VALID_LABELS:
            issues.append("invalid_label")
        if row.get("label") == "UNKNOWN":
            issues.append("unknown_label")
        if issues:
            invalid_rows.append({"row_index": idx, "source": row.get("source"), "label": row.get("label"), "issues": issues})

    duplicate_examples = []
    seen: dict[str, int] = {}
    for idx, h in enumerate(hashes):
        if h in seen and len(duplicate_examples) < 100:
            duplicate_examples.append(
                {
                    "first_row": seen[h],
                    "duplicate_row": idx,
                    "first_source": rows[seen[h]].get("source"),
                    "duplicate_source": rows[idx].get("source"),
                    "label": rows[idx].get("label"),
                }
            )
        seen.setdefault(h, idx)

    repeated_templates = [
        {"template_hash": h, "count": c}
        for h, c in template_counts.most_common(100)
        if c > 1
    ]

    hard_ham_rows = [
        row
        for row in rows
        if row.get("label") == "HAM" and row.get("category_tags")
    ]

    report = {
        "total_rows": len(rows),
        "source_summaries": source_summaries,
        "label_distribution": summarize_counter(rows, "label"),
        "language_distribution": summarize_counter(rows, "language"),
        "source_distribution": summarize_counter(rows, "source"),
        "source_label_distribution": nested_count(rows, ("source", "label")),
        "language_label_distribution": nested_count(rows, ("language", "label")),
        "length_bucket_distribution": dict(Counter(length_bucket(int(row.get("message_length", 0) or 0)) for row in rows)),
        "missing_text_count": sum(1 for row in rows if not str(row.get("text", "")).strip()),
        "invalid_or_unknown_label_count": len(invalid_rows),
        "exact_duplicate_rows": sum(c - 1 for c in hash_counts.values() if c > 1),
        "exact_duplicate_groups": sum(1 for c in hash_counts.values() if c > 1),
        "repeated_template_rows": sum(c - 1 for c in template_counts.values() if c > 1),
        "repeated_template_groups": sum(1 for c in template_counts.values() if c > 1),
        "link_coverage": {
            "total": sum(1 for row in rows if row.get("contains_link")),
            "ham": sum(1 for row in rows if row.get("contains_link") and row.get("label") == "HAM"),
            "spam_or_phishing": sum(1 for row in rows if row.get("contains_link") and row.get("label") in {"SPAM", "PHISHING"}),
        },
        "phone_coverage": {
            "total": sum(1 for row in rows if row.get("contains_phone")),
            "ham": sum(1 for row in rows if row.get("contains_phone") and row.get("label") == "HAM"),
            "spam_or_phishing": sum(1 for row in rows if row.get("contains_phone") and row.get("label") in {"SPAM", "PHISHING"}),
        },
        "hard_ham_heuristic_count": len(hard_ham_rows),
        "hard_ham_by_source": dict(Counter(row.get("source", "UNKNOWN") for row in hard_ham_rows)),
        "invalid_row_examples": invalid_rows[:100],
        "duplicate_examples": duplicate_examples,
        "repeated_template_examples": repeated_templates,
        "extreme_length": {
            "short_0_5_words": sum(1 for row in rows if int(row.get("message_length", 0) or 0) <= 5),
            "very_long_2000w_plus": sum(1 for row in rows if int(row.get("message_length", 0) or 0) >= 2000),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate processed SpamGuard datasets.")
    parser.add_argument("--include-empty-sources", action="store_true")
    args = parser.parse_args()
    rows, source_summaries = load_rows(include_empty_sources=args.include_empty_sources)
    report = validate_rows(rows, source_summaries)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / "data_quality_report.json").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_csv(REPORT_DIR / "source_summaries.csv", source_summaries)
    write_csv(REPORT_DIR / "duplicate_examples.csv", report["duplicate_examples"])
    write_csv(REPORT_DIR / "invalid_row_examples.csv", report["invalid_row_examples"])
    print(json.dumps({"report": str(REPORT_DIR / "data_quality_report.json"), "total_rows": report["total_rows"]}, indent=2))


if __name__ == "__main__":
    main()
