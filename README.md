# SpamGuard - Transformer Email Spam Classifier

Notebook project for training English and multilingual spam/ham classifiers with Transformer encoders and validating predictions from `.eml` files.

## What This Project Does
- Trains a Transformer text classifier for spam detection (`HAM=0`, `SPAM=1`)
- Supports an English model and a multilingual model
- Parses real `.eml` files and runs inference
- Shows evaluation metrics and graphs (confusion matrix, ROC, PR, learning curves)

## Project Structure
- `SpamGuard_Transformer_Email_Spam_Classifier.ipynb`: End-to-end notebook pipeline
- `test/`: Example `.eml` files for inference tests
- `spmEN.model/`: English model output
- `globalSpm.model/`: Multilingual model output
- `.hf_cache/`: Hugging Face cache (ASCII-safe path for Windows)
- `data/`: Downloaded datasets (not committed)

## Quick Start
1. Open `SpamGuard_Transformer_Email_Spam_Classifier.ipynb`.
2. Run cells top-to-bottom.
3. Train and save models:
- English output directory: `spmEN.model/`
- Multilingual output directory: `globalSpm.model/`

## EML Test Files
- `test/spam_en.eml`
- `test/spam_he.eml`
- `test/ham_en.eml`
- `test/ham_he.eml`

## Notes
- English-only training performs worse on Hebrew or other languages.
- For multilingual support, use `MODEL_NAME = "xlm-roberta-base"` and train again.
- Cache path is redirected to `./.hf_cache` to avoid non-ASCII Windows path issues.

## Output Metrics
The notebook reports:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

