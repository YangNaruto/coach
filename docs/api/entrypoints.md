## CLI entry points

### Slot Filling (Coach)
- File: `slu_main.py`
- Function: `main(params)`
  - Builds `BinarySLUTagger`, `SlotNamePredictor` (and `SentRepreGenerator` if `--tr`), trains for up to `--epoch` epochs with early stopping.
  - Uses `src.slu.dataloader.get_dataloader(--tgt_dm, --batch_size, --tr, --n_samples)`
  - Saves best model at `dump_path/best_model.pth` with keys `binary_slu_tagger` and `slotname_predictor`.

### Slot Filling baselines
- File: `slu_baseline.py`
- Function: `run_baseline(params)`
  - Trains `ConceptTagger` baseline (CT/RZT). Evaluates bin-F1 and final slot F1 on dev/test.

### Slot Filling test runner
- File: `slu_test.py`
- Functions:
  - `test_coach(params)`: Load Coach checkpoint and evaluate testset.
  - `test_baseline(params)`: Load baseline checkpoint and evaluate testset.
  - `test_coach_on_seen_and_unseen(params)`: Evaluate Coach on seen/unseen subsets for `--tgt_dm`.
  - `test_baseline_on_seen_and_unseen(params)`: Evaluate baseline on seen/unseen subsets.

### NER (Coach)
- File: `ner_main.py`
- Function: `main(params)`
  - Trains `BinaryNERagger` and `EntityNamePredictor` (and `SentRepreGenerator` if `--tr`). Saves best model by dev F1.

### NER baselines
- File: `ner_baseline.py`
- Function: `run_baseline(params)`
  - If `--bilstmcrf`, trains `BiLSTMCRFTagger` via `BiLSTMCRFTrainer`. Else trains CT/RZT `ConceptTagger` baseline.

All entry points use `config.get_params()` and `src.utils.init_experiment()` for consistent logging and output directories.