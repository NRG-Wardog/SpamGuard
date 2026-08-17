from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spamguard_data.build_combined import (
    dedup_and_cluster,
    split_clusters,
    verify_split_isolation,
)
from spamguard_data.common import enrich_record, template_hash
from spamguard_data.ingest import email_records_from_folder
from spamguard_data.manifest import load_manifest, save_manifest, upsert_source
from spamguard_data.validate import validate_rows


def record(text: str, label: str, source: str = "fixture") -> dict[str, object]:
    return enrich_record(
        text=text,
        label=label,
        source=source,
        allow_training=True,
        allow_evaluation=True,
        license_status="test_only",
    ).to_dict()


class IngestionTests(unittest.TestCase):
    def test_email_folder_ingestion_parses_body_label_and_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spam_dir = root / "spam"
            spam_dir.mkdir()
            (spam_dir / "message.eml").write_bytes(
                b"From: sender@example.test\nContent-Type: text/html; charset=utf-8\n\n<p>Claim prize now</p>"
            )

            rows = email_records_from_folder(
                root,
                source="fixture",
                allow_training=False,
                allow_evaluation=True,
                license_status="review_required",
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "SPAM")
        self.assertEqual(rows[0]["text"], "Claim prize now")
        self.assertFalse(rows[0]["allow_training"])
        self.assertEqual(rows[0]["license_status"], "review_required")


class ManifestTests(unittest.TestCase):
    def test_upsert_preserves_existing_fields_and_round_trips(self) -> None:
        manifest = {"version": 1, "sources": [{"source_name": "a", "license_status": "review", "count": 1}]}
        upsert_source(manifest, {"source_name": "a", "count": 2})
        upsert_source(manifest, {"source_name": "b", "allowed_use": {"training": False}})

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            save_manifest(manifest, path)
            loaded = load_manifest(path)

        self.assertEqual(loaded["sources"][0]["license_status"], "review")
        self.assertEqual(loaded["sources"][0]["count"], 2)
        self.assertEqual([item["source_name"] for item in loaded["sources"]], ["a", "b"])


class GovernanceTests(unittest.TestCase):
    def test_dedup_excludes_and_reports_conflicting_labels(self) -> None:
        rows = [
            record("same message", "HAM", "one"),
            record(" SAME  message ", "SPAM", "two"),
            record("safe unique", "HAM"),
            record("safe unique", "HAM"),
        ]

        clean, _clusters, report = dedup_and_cluster(rows)

        self.assertEqual([row["text"] for row in clean], ["safe unique"])
        self.assertEqual(report["removed_exact_duplicates"], 1)
        self.assertEqual(report["conflicting_label_groups"], 1)
        self.assertEqual(report["conflicting_label_rows_excluded"], 2)
        self.assertEqual(report["conflicting_label_examples"][0]["labels"], ["HAM", "SPAM"])

    def test_template_variants_are_clustered_together(self) -> None:
        rows = [record("Call +1 212 555 1234 for $50", "SPAM"), record("Call +1 646 555 9876 for $99", "SPAM")]
        clean, clusters, report = dedup_and_cluster(rows)

        self.assertEqual(len(clean), 2)
        self.assertEqual(template_hash(rows[0]["text"]), template_hash(rows[1]["text"]))
        self.assertEqual(sorted(len(members) for members in clusters.values()), [2])
        self.assertEqual(report["repeated_template_clusters"], 1)

    def test_split_is_reproducible_and_has_no_exact_or_template_leakage(self) -> None:
        rows = []
        for label in ("HAM", "SPAM"):
            for index in range(20):
                unique_word = chr(ord("a") + index) * 4
                rows.append(record(f"{label} campaign {unique_word}", label, f"source-{index % 2}"))
        clean, clusters, _report = dedup_and_cluster(rows)

        first = split_clusters(clean, clusters, seed=1234)
        second = split_clusters(clean, clusters, seed=1234)

        signature = lambda splits: {
            name: [(row["text"], row["clean_split"]) for row in values]
            for name, values in splits.items()
        }
        self.assertEqual(signature(first), signature(second))
        self.assertTrue(verify_split_isolation(first)["passed"])
        self.assertTrue(all(first[name] for name in ("train", "validation", "test")))

    def test_leakage_check_detects_template_overlap(self) -> None:
        splits = {
            "train": [record("Pay 100 at https://example.test/a", "SPAM")],
            "validation": [record("Pay 200 at https://example.test/b", "SPAM")],
            "test": [],
        }
        check = verify_split_isolation(splits)

        self.assertFalse(check["passed"])
        self.assertEqual(check["pairwise"]["train_vs_validation"]["template_overlap"], 1)

    def test_quality_report_surfaces_label_conflicts(self) -> None:
        report = validate_rows([record("same", "HAM"), record("same", "SPAM")], [])
        self.assertEqual(report["conflicting_label_groups"], 1)
        self.assertEqual(report["conflicting_label_rows"], 2)


if __name__ == "__main__":
    unittest.main()
