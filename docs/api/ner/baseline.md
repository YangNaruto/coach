## NER baselines (CT/RZT/BiLSTM-CRF)

Data loader (`src/ner/baseline_loader.py`)
- `Dataset(X, y)` and `collate_fn`
- `convert_labels_to_slot_based_labels(label_list)` → `[num_entity_types, seq_len]` BIO labels (`y1_set`), used by CT/RZT
- `get_dataloader(batch_size, bilstmcrf=False, n_samples=0)` chooses label format based on `bilstmcrf`

Models (`src/ner/baseline_model.py`)
- `ConceptTagger(params, vocab)`: CT/RZT; concatenates token features with entity embeddings and optional example attention
- `BiLSTMCRFTagger(params, vocab)`: classic sequence tagger over `y2_set`

Trainers (`src/ner/baseline_trainer.py`)
- `BaselineTrainer(params, ner_tagger)`: trains CT/RZT, `evaluate` turns per-entity predictions into final tags
- `BiLSTMCRFTrainer(params, ner_tagger)`: CRF loss and Viterbi decode for evaluation