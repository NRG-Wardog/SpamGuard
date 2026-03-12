from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from spamguard_demo.inference import (
    _decode_part_bytes,
    classify_mail_folder,
    extract_eml_metadata,
    parse_eml_bytes,
    predict_eml_file,
)


class InferenceSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.model_dir = self.repo_root / "globalSpm.model"

    def test_parse_eml_bytes_handles_hebrew_style_headers(self) -> None:
        raw_bytes = (self.repo_root / "test" / "ham_he.eml").read_bytes()
        parsed_text = parse_eml_bytes(raw_bytes)
        metadata = extract_eml_metadata(raw_bytes)

        self.assertIn("Subject: סיכום פגישה", parsed_text)
        self.assertEqual(metadata["from"], "Alex@example.com")

    def test_predict_eml_file_smoke_samples(self) -> None:
        expected_predictions = {
            "spam_en.eml": "SPAM",
            "spam_he.eml": "SPAM",
            "ham_en.eml": "HAM",
            "ham_he.eml": "HAM",
        }

        for file_name, expected_prediction in expected_predictions.items():
            with self.subTest(file_name=file_name):
                result = predict_eml_file(
                    self.repo_root / "test" / file_name,
                    model_dir=self.model_dir,
                )
                self.assertEqual(result["prediction"], expected_prediction)
                self.assertGreaterEqual(result["spam_probability"], 0.0)
                self.assertLessEqual(result["spam_probability"], 1.0)

    def test_decode_part_bytes_handles_iso_8859_8_directional_aliases(self) -> None:
        hebrew_text = "שלום עולם"
        raw_bytes = hebrew_text.encode("iso-8859-8")

        decoded = _decode_part_bytes(raw_bytes, "iso-8859-8-i")

        self.assertEqual(decoded, hebrew_text)


class BatchClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.model_dir = self.repo_root / "globalSpm.model"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="spamguard_batch_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_classify_mail_folder_copies_inputs_and_writes_manifest(self) -> None:
        input_dir = self.temp_dir / "mails"
        output_dir = self.temp_dir / "result"
        input_dir.mkdir(parents=True, exist_ok=True)

        for sample_path in (self.repo_root / "test").glob("*.eml"):
            shutil.copy2(sample_path, input_dir / sample_path.name)

        results = classify_mail_folder(
            input_dir=input_dir,
            output_dir=output_dir,
            copy_mode="copy",
            model_dir=self.model_dir,
            source_name="local",
        )

        self.assertEqual(len(results), 5)
        self.assertTrue((output_dir / "spam").exists())
        self.assertTrue((output_dir / "ham").exists())
        self.assertTrue((output_dir / "results.csv").exists())
        self.assertEqual(len(list(input_dir.glob("*.eml"))), 5)

        with (output_dir / "results.csv").open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertEqual(row["provider"], "local")
            self.assertEqual(row["remote_message_id"], "")
            self.assertEqual(row["thread_id"], "")
            self.assertTrue(Path(self.repo_root / row["stored_result_path"]).exists())


if __name__ == "__main__":
    unittest.main()
