## SLU baselines (CT/RZT) (`src/slu/baseline_*.py`)

Data loader
- `baseline_loader.Dataset(X, y, domains)` and `collate_fn`
- `convert_labels_to_slot_based_labels(label_list, dm_name)` → `[num_slot_in_domain, seq_len]` with BIO in `y1_set`
- `get_dataloader(tgt_domain, batch_size, n_samples)` → CT/RZT training uses slot-based labels per domain

Model (`baseline_model.py`)
- `ConceptTagger(params, vocab)`
  - Encodes with `Lstm`; for each domain, concatenates token features with per-slot embeddings and optionally example-attention features (`--use_example`) to predict BIO per slot.
- `LstmBasedSlotPredictor(params)`
  - LSTM over per-slot sequence to generate slot-BIO predictions.

Trainer (`baseline_trainer.py`)
- `BaselineTrainer(params, slu_tagger)`
  - `train_step(X, lengths, y, y_dm)` computes cross-entropy over per-slot sequences.
  - `evaluate(e, dataloader, istestset=False)` converts per-slot predictions into final tags and computes F1 (also outputs binary F1). Saves best model on dev.

Baseline entry point (`slu_baseline.py`)
- `run_baseline(params)` wires loader, model, trainer, and training loop.