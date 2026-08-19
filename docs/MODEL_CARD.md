# SpamGuard Model & Evaluation Card

This document defines the public evaluation boundary for SpamGuard. It is intentionally conservative: it distinguishes the evidence captured by a completed notebook experiment from the stricter data-governance pipeline now present in the repository.

## System Purpose

SpamGuard is an email-classification system and ML-engineering project for distinguishing legitimate mail from spam/phishing-oriented content. The repository combines a Transformer classifier with governed dataset ingestion, reproducible split construction, leakage checks, `.eml` inference, read-only Gmail import, and a review UI.

The primary engineering objective is not a single headline score. It is a traceable path from source data to a defensible evaluation and usable inference workflow.

## Intended Use

- experimentation with email spam/phishing classification;
- reproducible ML/data-engineering demonstrations;
- local or review-oriented classification of `.eml` files;
- read-only Gmail import followed by local review/inference;
- studying dataset provenance, deduplication, template leakage, and evaluation controls.

## Out of Scope

SpamGuard should not be interpreted as:

- a universal claim of real-world spam/phishing detection accuracy;
- a substitute for a production mail-security gateway;
- a guarantee against novel, adversarial, or distribution-shifted campaigns;
- evidence that the recorded notebook score will reproduce on every future dataset version;
- an authorization to use a source for training or evaluation when its licensing/policy metadata does not permit that use.

## Recorded Evaluation Evidence

A completed two-epoch Transformer experiment preserved in `SpamGuard_Transformer_Email_Spam_Classifier.ipynb` records:

| Split | Accuracy | Precision | Recall | F1 | ROC AUC | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 99.32% | 99.45% | 99.02% | 99.23% | 99.97% | 192.16 samples/s |
| Test | 99.33% | 99.51% | 98.98% | 99.24% | 99.97% | 192.54 samples/s |

Additional recorded values include validation loss `0.03736`, test loss `0.03612`, training loss `0.06774`, and training runtime `4034.18 s`.

These numbers are evidence from that specific committed experiment. They are not presented as external, temporal, adversarial, or production-generalization results. See [`RESULTS.md`](RESULTS.md) for the full provenance statement.

## Evaluation Boundary

The current repository separates three layers:

1. **Data validity**: records are structurally usable and their source/allowed-use metadata is retained.
2. **Split validity**: exact duplicates, conflicting labels, and repeated normalized templates are controlled before deterministic train/validation/test construction.
3. **Model quality**: metrics are meaningful only after the first two layers are satisfied.

The current governed pipeline is stricter than the workflow used for the historical recorded notebook run. Therefore the project does not silently relabel the historical score as a result from the newer governed split.

## Data Governance Controls

The reusable data pipeline implements and tests the following invariants:

- source provenance and allowed-use metadata are preserved;
- identical normalized text with conflicting labels is excluded;
- exact duplicate content is removed before split construction;
- repeated normalized templates are clustered;
- a template cluster is assigned to only one split;
- exact-text and template overlap are checked across train, validation, and test;
- split construction is deterministic for a fixed seed;
- leakage verification fails rather than merely warning.

See [`ML_ENGINEERING.md`](ML_ENGINEERING.md) for the engineering contract and `spamguard_data/` for the implementation.

## Inference Surface

The repository supports:

- local `.eml` parsing and classification;
- folder/batch inference;
- read-only Gmail OAuth import;
- a Streamlit review interface.

Gmail access is explicitly read-only. OAuth client/token files are local runtime material and are not part of the model artifact or public dataset.

## Known Limitations

- The committed headline metrics come from one recorded experiment rather than an independently reproduced external benchmark.
- No temporal holdout or continuously refreshed production traffic benchmark is committed.
- No claim is made for robustness to adversarially crafted messages, unseen campaign families, or future distribution shift.
- Public CI validates deterministic data-engineering invariants; it does not retrain the model on every pull request.
- Throughput values are copied from the recorded evaluation run and should not be treated as a controlled hardware benchmark without the original runtime context.
- Source licensing and allowed-use status can constrain which data may participate in future training/evaluation runs.

## Reproducibility Requirements for Future Results

A future result should be treated as comparable only when it records at least:

- source-manifest/data-policy state;
- split seed and split artifact identity;
- leakage-check result;
- model/config identifier;
- per-class precision, recall, and F1;
- confusion matrix;
- appropriate ROC/PR metrics;
- inference latency/throughput with hardware context;
- explicit statement of whether the evaluation set is internal, external, temporal, or adversarial.

Until such a run is committed, the notebook experiment above remains the recorded model-performance evidence and the governed pipeline remains the stronger evidence for ML-engineering methodology.
