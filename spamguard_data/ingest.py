from __future__ import annotations

import argparse
import csv
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .common import (
    enrich_record,
    infer_label_from_path,
    iter_mbox_texts,
    normalize_label,
    parse_email_bytes,
    read_jsonl,
    write_csv,
    write_jsonl,
)
from .manifest import MANIFEST_PATH, load_manifest, save_manifest, upsert_source


ROOT = Path(".")
RAW = Path("data/raw")
PROCESSED = Path("data/processed")

SPAMASSASSIN_URL = "https://spamassassin.apache.org/old/publiccorpus/"
UCI_SMS_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


SOURCE_METADATA: dict[str, dict[str, Any]] = {
    "spamassassin": {
        "source_name": "spamassassin",
        "display_name": "SpamAssassin Public Corpus",
        "source_url": "https://spamassassin.apache.org/old/publiccorpus/readme.html",
        "download_method": "Local archives already present; public Apache mirror supports direct archive download.",
        "parser_name": "spamassassin_email_dirs",
        "license_status": "public_corpus_sender_copyright_do_not_use_live_system",
        "allowed_use": {"training": False, "evaluation": True, "research_only": True},
        "label_mapping": {"easy_ham": "HAM", "easy_ham_2": "HAM", "hard_ham": "HAM", "spam": "SPAM", "spam_2": "SPAM"},
        "known_risks": ["old data", "sender copyright ambiguity", "newsletter/list bias", "template duplicates"],
        "notes": "Hard-ham is useful for evaluation, but training remains disabled pending sender-copyright review. Do not inject these messages into live email systems.",
    },
    "csdmc2010": {
        "source_name": "csdmc2010",
        "display_name": "CSDMC2010 formatted local corpus",
        "source_url": "https://github.com/zrz1996/Spam-Email-Classifier-DataSet",
        "download_method": "Local ZIPs already present in this project.",
        "parser_name": "csdmc_email_dirs",
        "license_status": "conditional_underlying_competition_terms_need_review",
        "allowed_use": {"training": False, "evaluation": True, "research_only": True},
        "label_mapping": {"ham": "HAM", "spam": "SPAM"},
        "known_risks": ["underlying corpus rights unclear", "HTML artifacts", "template duplicates", "unlabeled original test set not used"],
        "notes": "Keep evaluation/research-only until the original CSDMC terms are accepted.",
    },
    "uci_sms": {
        "source_name": "uci_sms",
        "display_name": "UCI SMS Spam Collection",
        "source_url": "https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection",
        "download_method": f"Direct download from {UCI_SMS_URL}",
        "parser_name": "uci_sms_tsv",
        "license_status": "CC_BY_4_0",
        "allowed_use": {"training": True, "evaluation": True, "research_only": False},
        "label_mapping": {"ham": "HAM", "spam": "SPAM"},
        "known_risks": ["SMS domain differs from email", "short-message sampling bias", "possible duplicate templates"],
        "notes": "Use as auxiliary data for short-message robustness, not as a replacement for email hard-HAM.",
    },
    "hebrew_eval": {
        "source_name": "hebrew_eval",
        "display_name": "Synthetic/manual Hebrew evaluation probe",
        "source_url": "local:analysis_outputs/data_building/hebrew_eval_200_synthetic_probe.csv",
        "download_method": "Generated locally by previous data-building pass.",
        "parser_name": "hebrew_probe_csv",
        "license_status": "local_synthetic_manual_eval_only",
        "allowed_use": {"training": False, "evaluation": True, "research_only": False},
        "label_mapping": {"HAM": "HAM", "SPAM": "SPAM"},
        "known_risks": ["synthetic/manual probes", "needs native-speaker audit", "not representative enough for training"],
        "notes": "Do not mix into training automatically.",
    },
    "enron_manual": {
        "source_name": "enron_manual",
        "display_name": "Enron / Enron-Spam manual ingest",
        "source_url": "https://www.cs.cmu.edu/~enron/ and https://www2.aueb.gr/users/ion/data/enron-spam/",
        "download_method": "Manual or optional large download; place maildir/CSV/mbox files under data/raw/enron_manual/.",
        "parser_name": "manual_folder",
        "license_status": "public_record_privacy_review_required",
        "allowed_use": {"training": False, "evaluation": True, "research_only": True},
        "label_mapping": {"ham": "HAM", "spam": "SPAM"},
        "known_risks": ["privacy/ethics concerns", "very stale corporate data", "large duplicates", "threads and templates"],
        "notes": "Enable training only after privacy/license review and strict deduplication.",
    },
    "trec_manual": {
        "source_name": "trec_manual",
        "display_name": "TREC Spam Corpora manual ingest",
        "source_url": "https://trec.nist.gov/data/spam.html",
        "download_method": "Manual download after reviewing/accepting corpus terms; place extracted corpus under data/raw/trec_manual/.",
        "parser_name": "trec_folder",
        "license_status": "agreement_required",
        "allowed_use": {"training": False, "evaluation": False, "research_only": True},
        "label_mapping": {"ham": "HAM", "spam": "SPAM"},
        "known_risks": ["agreement/access ambiguity", "campaign duplicates", "chronological leakage if split incorrectly"],
        "notes": "Do not auto-download or train until agreement is accepted.",
    },
    "dcinbox_manual": {
        "source_name": "dcinbox_manual",
        "display_name": "DCinbox newsletter-style HAM manual ingest",
        "source_url": "https://www.dcinbox.com/",
        "download_method": "Manual export/download if usage terms permit; place CSV files under data/raw/dcinbox_manual/.",
        "parser_name": "newsletter_csv",
        "license_status": "terms_need_review",
        "allowed_use": {"training": False, "evaluation": True, "research_only": True},
        "label_mapping": {"newsletter": "HAM", "ham": "HAM"},
        "known_risks": ["political/newsletter domain bias", "high template repetition", "terms unclear"],
        "notes": "Useful for hard-HAM evaluation/mining if license allows.",
    },
    "avocado_manual": {
        "source_name": "avocado_manual",
        "display_name": "Avocado Research Email Collection",
        "source_url": "https://catalog.ldc.upenn.edu/LDC2015T03",
        "download_method": "Manual LDC licensed download only; place approved files under data/raw/avocado_manual/.",
        "parser_name": "manual_folder",
        "license_status": "LDC_license_required",
        "allowed_use": {"training": False, "evaluation": False, "research_only": True},
        "label_mapping": {"email": "HAM"},
        "known_risks": ["license required", "no redistribution", "privacy sensitivity", "possible malware/attachments", "thread duplicates"],
        "notes": "Excellent business/system HAM candidate after LDC approval.",
    },
    "sms_phishing_manual": {
        "source_name": "sms_phishing_manual",
        "display_name": "SMS Phishing Dataset for ML and Pattern Recognition",
        "source_url": "https://data.mendeley.com/datasets/f45bkkt8pr/1",
        "download_method": "Manual Mendeley download after verifying page license; place CSV/XLSX under data/raw/sms_phishing_manual/.",
        "parser_name": "generic_sms_csv",
        "license_status": "license_needs_review",
        "allowed_use": {"training": False, "evaluation": True, "research_only": True},
        "label_mapping": {"legitimate": "HAM", "ham": "HAM", "spam": "SPAM", "smishing": "PHISHING"},
        "known_risks": ["SMS domain mismatch", "license must be confirmed", "duplicate templates", "label noise"],
        "notes": "Auxiliary short-message and phishing robustness source after license check.",
    },
}


MANUAL_READMES = {
    "data/raw/enron_manual/README.md": """# Enron / Enron-Spam manual source

Place approved Enron maildir, mbox, CSV, or raw email files here.

Suggested sources:
- CMU Enron Email Dataset: https://www.cs.cmu.edu/~enron/
- Enron-Spam: https://www2.aueb.gr/users/ion/data/enron-spam/

Keep raw files unchanged. The parser infers labels from folder names containing `ham`, `spam`,
`phishing`, or from CSV label columns. This source remains evaluation/research-only in the
manifest until privacy/license review is complete.
""",
    "data/raw/trec_manual/README.md": """# TREC Spam Corpora manual source

Do not bypass TREC/Waterloo/NIST access terms. After reviewing and accepting the applicable
agreement, place the extracted TREC corpus here.

Expected formats supported:
- `full/index` or `index` containing lines like `spam ../data/inmail.1`
- referenced raw message files under the same extracted tree

This source is marked agreement-required and is not training-enabled by default.
""",
    "data/raw/dcinbox_manual/README.md": """# DCinbox newsletter-style HAM manual source

Place permitted DCinbox CSV exports here only after confirming usage terms.

Supported CSV columns include `text`, `body`, `content`, `message`, `newsletter_text`, or
`subject` plus body fields. Rows default to HAM/newsletter if no label column is present.
This source is useful for hard-HAM evaluation/mining but is not training-enabled by default.
""",
    "data/raw/avocado_manual/README.md": """# Avocado Research Email Collection manual source

This dataset requires LDC access and signed license agreements:
https://catalog.ldc.upenn.edu/LDC2015T03

Place approved local copies here only if your organization has access. Do not redistribute
raw files. Avoid processing attachments unless they have been scanned and approved.
This source is not training-enabled by default.
""",
    "data/raw/sms_phishing_manual/README.md": """# SMS Phishing Dataset manual source

Dataset page: https://data.mendeley.com/datasets/f45bkkt8pr/1

Manually download only after verifying the displayed license and your intended use. Place
CSV/XLSX exports here. The generic SMS parser supports columns like `text`, `message`,
`label`, `type`, or `category`, and maps Legitimate/Ham to HAM, Spam to SPAM, and
Smishing/Phishing to PHISHING.
""",
}


def copy_existing_raw() -> None:
    mappings = [
        (Path("data/spamassassin_corpus"), RAW / "spamassassin"),
        (Path("data/csdmc2010"), RAW / "csdmc2010"),
        (Path("analysis_outputs/data_building/hebrew_eval_200_synthetic_probe.csv"), RAW / "hebrew_eval" / "hebrew_eval_200_synthetic_probe.csv"),
    ]
    for src, dst in mappings:
        if not src.exists():
            continue
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            for path in src.iterdir():
                target = dst / path.name
                if path.is_file() and not target.exists():
                    shutil.copy2(path, target)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)


def download_uci_sms() -> None:
    raw_dir = RAW / "uci_sms"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "sms_spam_collection.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(UCI_SMS_URL, zip_path)
    extract_dir = raw_dir / "extracted"
    if not (extract_dir / "SMSSpamCollection").exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)


def setup_manual_readmes() -> None:
    for path_text, content in MANUAL_READMES.items():
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("w", encoding="utf-8", newline="\n") as fh:
                fh.write(content)


def email_records_from_folder(
    root: Path,
    *,
    source: str,
    allow_training: bool,
    allow_evaluation: bool,
    license_status: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith(".") or path.suffix.lower() in {".zip", ".bz2", ".gz", ".md"}:
            continue
        label = infer_label_from_path(path)
        if label == "UNKNOWN":
            continue
        try:
            text = parse_email_bytes(path.read_bytes())
        except Exception:
            continue
        if not text:
            continue
        rows.append(
            enrich_record(
                text=text,
                label=label,
                source=source,
                allow_training=allow_training,
                allow_evaluation=allow_evaluation,
                license_status=license_status,
                metadata={"raw_path": str(path)},
            ).to_dict()
        )
    return rows


def process_spamassassin() -> dict[str, Any]:
    meta = SOURCE_METADATA["spamassassin"]
    root = Path("data/spamassassin_corpus")
    rows = email_records_from_folder(
        root,
        source="spamassassin",
        allow_training=bool(meta["allowed_use"]["training"]),
        allow_evaluation=True,
        license_status=meta["license_status"],
    )
    out = PROCESSED / "spamassassin" / "spamassassin.jsonl"
    count = write_jsonl(out, rows)
    write_csv(PROCESSED / "spamassassin" / "spamassassin_preview.csv", rows[:1000])
    return {**meta, "local_raw_path": str(RAW / "spamassassin"), "processed_path": str(out), "sample_counts": summarize(rows), "dedup_status": "not_deduplicated"}


def process_csdmc() -> dict[str, Any]:
    meta = SOURCE_METADATA["csdmc2010"]
    root = Path("data/csdmc2010")
    rows = email_records_from_folder(
        root,
        source="csdmc2010",
        allow_training=False,
        allow_evaluation=True,
        license_status=meta["license_status"],
    )
    out = PROCESSED / "csdmc2010" / "csdmc2010.jsonl"
    count = write_jsonl(out, rows)
    write_csv(PROCESSED / "csdmc2010" / "csdmc2010_preview.csv", rows[:1000])
    return {**meta, "local_raw_path": str(RAW / "csdmc2010"), "processed_path": str(out), "sample_counts": summarize(rows), "dedup_status": "not_deduplicated"}


def process_uci_sms() -> dict[str, Any]:
    meta = SOURCE_METADATA["uci_sms"]
    path = RAW / "uci_sms" / "extracted" / "SMSSpamCollection"
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for idx, line in enumerate(fh):
                label, sep, text = line.partition("\t")
                if not sep:
                    continue
                rows.append(
                    enrich_record(
                        text=text,
                        label=label,
                        source="uci_sms",
                        split_origin="raw_sms",
                        sample_origin="real_sms",
                        allow_training=True,
                        allow_evaluation=True,
                        license_status=meta["license_status"],
                        metadata={"row": idx},
                    ).to_dict()
                )
    out = PROCESSED / "uci_sms" / "uci_sms.jsonl"
    write_jsonl(out, rows)
    write_csv(PROCESSED / "uci_sms" / "uci_sms_preview.csv", rows[:1000])
    return {**meta, "local_raw_path": str(RAW / "uci_sms"), "processed_path": str(out), "sample_counts": summarize(rows), "dedup_status": "not_deduplicated"}


def process_hebrew_eval() -> dict[str, Any]:
    meta = SOURCE_METADATA["hebrew_eval"]
    path = RAW / "hebrew_eval" / "hebrew_eval_200_synthetic_probe.csv"
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for idx, row in enumerate(reader):
                text = row.get("text", "")
                label = row.get("label", row.get("true_label", "UNKNOWN"))
                sample_origin = row.get("sample_origin") or "synthetic_manual_probe"
                record = enrich_record(
                    text=text,
                    label=label,
                    source="hebrew_eval",
                    split_origin="hebrew_probe",
                    sample_origin=sample_origin,
                    allow_training=False,
                    allow_evaluation=True,
                    license_status=meta["license_status"],
                    metadata={k: v for k, v in row.items() if k not in {"text", "label"}},
                ).to_dict()
                record["allow_training"] = False
                rows.append(record)
    out = PROCESSED / "hebrew_eval" / "hebrew_eval.jsonl"
    write_jsonl(out, rows)
    write_csv(PROCESSED / "hebrew_eval" / "hebrew_eval.csv", rows)
    return {**meta, "local_raw_path": str(RAW / "hebrew_eval"), "processed_path": str(out), "sample_counts": summarize(rows), "dedup_status": "deduplicated_by_generation_script"}


def detect_csv_columns(fieldnames: list[str]) -> tuple[str | None, str | None]:
    lower = {name.lower().strip(): name for name in fieldnames}
    text_col = next((lower[key] for key in ["text", "message", "body", "content", "email", "newsletter_text", "raw_text"] if key in lower), None)
    label_col = next((lower[key] for key in ["label", "type", "category", "class", "target", "spam"] if key in lower), None)
    return text_col, label_col


def process_generic_manual(source_key: str) -> dict[str, Any]:
    meta = SOURCE_METADATA[source_key]
    root = RAW / source_key
    rows: list[dict[str, Any]] = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name.lower() == "readme.md":
                continue
            suffix = path.suffix.lower()
            if suffix == ".csv":
                with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
                    reader = csv.DictReader(fh)
                    text_col, label_col = detect_csv_columns(reader.fieldnames or [])
                    if text_col is None:
                        continue
                    for idx, row in enumerate(reader):
                        label = normalize_label(row.get(label_col)) if label_col else infer_label_from_path(path)
                        if source_key == "dcinbox_manual" and label == "UNKNOWN":
                            label = "HAM"
                        rows.append(
                            enrich_record(
                                text=row.get(text_col, ""),
                                label=label,
                                source=source_key,
                                split_origin="manual_csv",
                                sample_origin="real_manual",
                                allow_training=bool(meta["allowed_use"]["training"]),
                                allow_evaluation=bool(meta["allowed_use"]["evaluation"]),
                                license_status=meta["license_status"],
                                metadata={"raw_path": str(path), "row": idx},
                            ).to_dict()
                        )
            elif suffix in {".mbox", ".mbx"}:
                label = infer_label_from_path(path)
                for text, message_meta in iter_mbox_texts(path):
                    rows.append(
                        enrich_record(
                            text=text,
                            label=label,
                            source=source_key,
                            split_origin="manual_mbox",
                            sample_origin="real_manual",
                            allow_training=bool(meta["allowed_use"]["training"]),
                            allow_evaluation=bool(meta["allowed_use"]["evaluation"]),
                            license_status=meta["license_status"],
                            metadata=message_meta,
                        ).to_dict()
                    )
            elif suffix in {".eml", ".txt", ""}:
                label = infer_label_from_path(path)
                if source_key == "avocado_manual" and label == "UNKNOWN":
                    label = "HAM"
                if label == "UNKNOWN":
                    continue
                rows.append(
                    enrich_record(
                        text=parse_email_bytes(path.read_bytes()),
                        label=label,
                        source=source_key,
                        split_origin="manual_email",
                        sample_origin="real_manual",
                        allow_training=bool(meta["allowed_use"]["training"]),
                        allow_evaluation=bool(meta["allowed_use"]["evaluation"]),
                        license_status=meta["license_status"],
                        metadata={"raw_path": str(path)},
                    ).to_dict()
                )
    out = PROCESSED / source_key / f"{source_key}.jsonl"
    write_jsonl(out, rows)
    if rows:
        write_csv(PROCESSED / source_key / f"{source_key}_preview.csv", rows[:1000])
    return {**meta, "local_raw_path": str(root), "processed_path": str(out), "sample_counts": summarize(rows), "dedup_status": "not_deduplicated"}


def process_trec_manual() -> dict[str, Any]:
    meta = SOURCE_METADATA["trec_manual"]
    root = RAW / "trec_manual"
    rows: list[dict[str, Any]] = []
    index_files = [p for p in [root / "full" / "index", root / "index"] if p.exists()]
    for index_path in index_files:
        with index_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh):
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                label = normalize_label(parts[0])
                msg_path = (index_path.parent / parts[1]).resolve()
                if not msg_path.exists():
                    msg_path = (root / parts[1]).resolve()
                if not msg_path.exists():
                    continue
                rows.append(
                    enrich_record(
                        text=parse_email_bytes(msg_path.read_bytes()),
                        label=label,
                        source="trec_manual",
                        split_origin="manual_trec",
                        sample_origin="real_manual",
                        allow_training=False,
                        allow_evaluation=False,
                        license_status=meta["license_status"],
                        metadata={"index_path": str(index_path), "line": line_no, "raw_path": str(msg_path)},
                    ).to_dict()
                )
    out = PROCESSED / "trec_manual" / "trec_manual.jsonl"
    write_jsonl(out, rows)
    return {**meta, "local_raw_path": str(root), "processed_path": str(out), "sample_counts": summarize(rows), "dedup_status": "not_deduplicated"}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels: dict[str, int] = {}
    languages: dict[str, int] = {}
    hard_ham = 0
    for row in rows:
        labels[row["label"]] = labels.get(row["label"], 0) + 1
        languages[row["language"]] = languages.get(row["language"], 0) + 1
        if row["label"] == "HAM" and row.get("category_tags"):
            hard_ham += 1
    return {"total": len(rows), "labels": labels, "languages": languages, "hard_ham_heuristic": hard_ham}


def update_all_manifest(processed_sources: Iterable[dict[str, Any]]) -> None:
    manifest = load_manifest()
    manifest["version"] = 1
    manifest["notes"] = "Central SpamGuard dataset manifest. Training flags are conservative until license review is complete."
    for source in SOURCE_METADATA.values():
        local = source.get("local_raw_path") or str(RAW / source["source_name"])
        processed = source.get("processed_path") or str(PROCESSED / source["source_name"] / f'{source["source_name"]}.jsonl')
        upsert_source(manifest, {**source, "local_raw_path": local, "processed_path": processed, "sample_counts": {"total": 0}, "dedup_status": "not_processed"})
    for source in processed_sources:
        upsert_source(manifest, source)
    save_manifest(manifest, MANIFEST_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and ingest SpamGuard datasets without training.")
    parser.add_argument("--download-auto", action="store_true", help="Download legally direct small sources such as UCI SMS.")
    parser.add_argument("--process", action="store_true", help="Process available raw/manual datasets into unified JSONL.")
    parser.add_argument("--setup-manual", action="store_true", help="Create manual source folders and README files.")
    args = parser.parse_args()

    if not any([args.download_auto, args.process, args.setup_manual]):
        args.download_auto = args.process = args.setup_manual = True

    processed: list[dict[str, Any]] = []
    copy_existing_raw()
    if args.setup_manual:
        setup_manual_readmes()
    if args.download_auto:
        try:
            download_uci_sms()
        except Exception as exc:
            print(f"WARNING: UCI SMS download failed: {exc}")
    if args.process:
        processed.extend([process_spamassassin(), process_csdmc(), process_uci_sms(), process_hebrew_eval()])
        for key in ["enron_manual", "dcinbox_manual", "avocado_manual", "sms_phishing_manual"]:
            processed.append(process_generic_manual(key))
        processed.append(process_trec_manual())
    update_all_manifest(processed)
    print(json.dumps({"manifest": str(MANIFEST_PATH), "processed_sources": [p["source_name"] for p in processed]}, indent=2))


if __name__ == "__main__":
    main()
