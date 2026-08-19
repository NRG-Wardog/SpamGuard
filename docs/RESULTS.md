# Evaluation Results

This document records metrics that are already captured in the executed training notebook committed to this repository. It exists to make the evidence easy to review without requiring a reader to inspect notebook output cells.

## Captured Transformer Run

The committed notebook contains a completed two-epoch training/evaluation run. The final recorded metrics are:

| Split | Accuracy | Precision | Recall | F1 | ROC AUC | Eval throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 99.32% | 99.45% | 99.02% | 99.23% | 99.97% | 192.16 samples/s |
| Test | 99.33% | 99.51% | 98.98% | 99.24% | 99.97% | 192.54 samples/s |

Additional recorded values:

- Validation loss: `0.03736`
- Test loss: `0.03612`
- Training runtime: `4034.18 s`
- Training throughput: `12.75 samples/s`
- Training loss: `0.06774`

The notebook also records the first-epoch validation F1 at approximately **98.59%**, followed by the final validation F1 of approximately **99.23%** after the second epoch.

## Metric Provenance

These numbers are **not hand-entered estimates**. They are transcribed from the output of `SpamGuard_Transformer_Email_Spam_Classifier.ipynb` committed in the repository.

The captured run used the notebook's configured Transformer workflow, deterministic seed, and the datasets enabled in that run. The notebook configuration records `SEED=42`, two training epochs, a maximum sequence length of 256, and the enabled multi-source email corpora.

## How to Interpret the Numbers

The metrics demonstrate that the trained classifier performed strongly on the validation/test data used by that recorded experiment. They should not be interpreted as a claim of universal real-world spam/phishing detection accuracy.

The repository has since added a stricter data-engineering layer around the model, including source provenance, exact-text deduplication, conflicting-label removal, template clustering, deterministic split generation, and explicit leakage checks. Those controls are documented separately in [`ML_ENGINEERING.md`](ML_ENGINEERING.md).

For that reason, the project treats **evaluation methodology and leakage control as first-class evidence**, rather than presenting a single accuracy number as the entire result.

## Reproducing Evaluation

The original model workflow remains in:

`SpamGuard_Transformer_Email_Spam_Classifier.ipynb`

The governed dataset preparation flow is implemented under `spamguard_data/`. Before comparing future model runs, rebuild and validate the dataset through that pipeline so the reported result is tied to a reproducible split and an explicit data-policy state.

## Recommended Future Result Reporting

Future runs should preserve, at minimum:

- dataset/source manifest version;
- split seed and split artifact hashes;
- per-class precision, recall, and F1;
- confusion matrix;
- ROC/PR curves where appropriate;
- inference throughput/latency and hardware context;
- explicit leakage-check result;
- model/config identifier.
