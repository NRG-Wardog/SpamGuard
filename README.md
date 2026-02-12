# LLM Spam Classifier

Notebook project for training a spam/ham email classifier with Transformers and running EML-based tests.

## Project Structure
- `LLM_Spam.ipynb` Notebook pipeline (setup, data loading, training, evaluation, inference)
- `test/` Sample .eml files for quick validation
- `spmEN.model/` Trained English model output (DeBERTa)
- `globalSpm.model/` Trained multilingual model output (XLM-R)
- `.hf_cache/` Hugging Face cache (ASCII-safe path)
- `data/` Downloaded datasets (not committed)

## Quick Start
1. Open `LLM_Spam.ipynb`.
2. Run the notebook from top to bottom.
3. If you trained previously:
   - English model output is saved in `spmEN.model/`.
   - Multilingual model output is saved in `globalSpm.model/`.

## EML Tests
The notebook includes an EML parsing helper and a test cell that runs these files:
- `test/spam_en.eml`
- `test/spam_he.eml`
- `test/ham_en.eml`
- `test/ham_he.eml`

## Notes
- Hebrew and other non-English messages are less accurate with English-only training.
- For multilingual support, use `MODEL_NAME = "xlm-roberta-base"` and re-train.
- The Hugging Face cache is redirected to `./.hf_cache` to avoid non-ASCII Windows path issues.

## Outputs
- Model artifacts are saved to the folder in `cfg.OUTPUT_DIR`.
- Metrics (accuracy, precision, recall, F1, ROC-AUC) are printed during training.
