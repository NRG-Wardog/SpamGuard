# SpamGuard Data Ingestion Next Steps

This branch adds a non-training dataset acquisition and ingestion layer for SpamGuard. No model training, RL, fine-tuning, or threshold re-selection was performed in this branch.

## What Was Added

- `spamguard_data/`: reusable ingestion, parsing, manifest, quality-check, and combined-dataset builder code.
- `data/dataset_manifest.json`: central metadata manifest for supported sources, license status, allowed use, paths, parser names, risks, and counts.
- `data/reports/`: safe summary reports from the processed sources.
- Manual dataset placeholder folders under `data/raw/*_manual/` with README files explaining where approved files should be placed.
- Parser support and manifest/report entries for a synthetic Hebrew evaluation probe marked evaluation-only and `allow_training=False`. The generated probe rows are not committed.

## Supported Datasets

| Source | Parser | Current status | Default use |
|---|---|---|---|
| SpamAssassin Public Corpus | raw email folders | local archives processed | training + evaluation |
| UCI SMS Spam Collection | TSV SMS file | auto-downloaded | auxiliary training + evaluation |
| CSDMC2010 | raw email folders | local ZIPs processed | evaluation/research only pending terms review |
| Hebrew synthetic/manual probe | CSV | generated locally, not committed | evaluation only |
| Enron / Enron-Spam | manual folder / CSV / mbox / raw email | parser ready, files not committed | research/evaluation only until approval |
| TREC Spam Corpora | TREC index + raw email files | parser ready, files not committed | agreement required |
| DCinbox | newsletter CSV | parser ready, files not committed | evaluation/research only until terms review |
| Avocado | manual folder / mbox / raw email | parser ready, files not committed | LDC license required |
| SMS Phishing | generic SMS CSV | parser ready, files not committed | evaluation/research only until license review |

## Automatic Downloads

The ingestion script can automatically download the UCI SMS Spam Collection from the UCI repository. It also copies existing local SpamAssassin, CSDMC2010, and Hebrew probe inputs into the `data/raw/` layout when present.

Raw downloaded/copied corpora and generated message-text datasets are intentionally ignored by git.

## Manual Downloads And License Approval

Do not bypass license gates, access agreements, or account approval. Place approved files in these folders:

- Enron / Enron-Spam: `data/raw/enron_manual/`
- TREC Spam Corpora: `data/raw/trec_manual/`
- DCinbox newsletter-style HAM: `data/raw/dcinbox_manual/`
- Avocado Research Email Collection: `data/raw/avocado_manual/`
- SMS Phishing Dataset: `data/raw/sms_phishing_manual/`

Each folder contains a README with expected formats and source-specific notes.

## Training Eligibility

Currently safe to consider later, after final review:

- SpamAssassin Public Corpus: useful for hard-HAM; public corpus but sender copyright caution remains.
- UCI SMS Spam Collection: CC BY 4.0; use as auxiliary short-message data, not primary email data.

Evaluation/research only by default:

- CSDMC2010: underlying competition terms still need review.
- Hebrew synthetic/manual probe: synthetic, evaluation-only, `allow_training=False`; generated rows are kept local and ignored.
- Enron / Enron-Spam: privacy/license review required.
- DCinbox: usage terms need review.
- SMS Phishing: page license must be verified.

Blocked until explicit approval:

- TREC Spam Corpora: agreement/access terms required.
- Avocado: LDC license required.

## How To Run Ingestion

From the repository root:

```bash
.venv/Scripts/python.exe -m spamguard_data.ingest
```

Useful variants:

```bash
.venv/Scripts/python.exe -m spamguard_data.ingest --download-auto
.venv/Scripts/python.exe -m spamguard_data.ingest --setup-manual
.venv/Scripts/python.exe -m spamguard_data.ingest --process
```

Outputs:

- Manifest: `data/dataset_manifest.json`
- Processed JSONL files: `data/processed/<source_name>/`
- Manual README folders: `data/raw/*_manual/`

## How To Run Quality Checks

```bash
.venv/Scripts/python.exe -m spamguard_data.validate --include-empty-sources
```

Outputs:

- `data/reports/data_quality_report.json`
- `data/reports/source_summaries.csv`
- `data/reports/duplicate_examples.csv`
- `data/reports/invalid_row_examples.csv`

Checks include missing text, invalid labels, exact duplicates, repeated template fingerprints, language distribution, HAM/SPAM balance, hard-HAM heuristics, link/phone coverage, and extreme lengths.

## How To Build The Combined Dataset

```bash
.venv/Scripts/python.exe -m spamguard_data.build_combined
```

Outputs:

- `data/combined/train.csv`
- `data/combined/validation.csv`
- `data/combined/test.csv`
- `data/combined/hebrew_eval.csv`
- `data/combined/auxiliary_evaluation_only.csv`
- `data/combined/data_report.json`

The builder respects source-level and row-level `allow_training` / `allow_evaluation` flags, removes exact duplicates, clusters by normalized template hash, and splits by cluster rather than by row.

## Current Quality Snapshot

Latest local run:

- Total processed rows: `16,134`
- Labels: `12,029 HAM`, `4,105 SPAM`
- Missing text: `0`
- Invalid or unknown labels: `0`
- Exact duplicate rows detected: `999`
- Repeated template rows detected: `1,198`
- Hard-HAM heuristic rows: `10,567`
- Hebrew evaluation rows: `200`, all evaluation-only

The training-candidate clean split currently includes only training-enabled sources:

- Train: `7,646`
- Validation: `1,642`
- Test: `1,637`

## Remaining Risks

- License and usage terms must be approved before enabling conditional sources for training.
- Raw corpora may contain sender-copyrighted content, stale examples, or private/personal data.
- Spam/newsletter datasets contain duplicate campaigns and repeated templates.
- Exact deduplication is implemented; deeper near-duplicate clustering should be added before training.
- Labels may be noisy, especially in legacy public corpora.
- SMS data helps short-message robustness but is domain-shifted from email.
- Synthetic Hebrew probes are useful for evaluation but are not representative enough for training.

## Future Options

- Deeper near-duplicate cleaning with MinHash, SimHash, or embedding clustering.
- Stronger template clustering for newsletters, spam campaigns, and mailer-daemon patterns.
- Source-specific sampling and loss weighting.
- Hard-HAM mining from current false positives and high-score HAM.
- Real, permissioned Hebrew HAM/SPAM collection.
- Hebrew evaluation expansion with native-speaker review.
- License approval for TREC, Avocado, DCinbox, and Enron.
- Cost-sensitive supervised fine-tuning after the data inventory is approved.
- Threshold re-search after new approved validation/test data is built.

## No Training Performed

This branch is data acquisition, parsing, validation, and preparation only. It does not train, fine-tune, run RL, change thresholds, or deploy a new model.
