## NER trainers (`src/ner/trainer.py`, `src/ner/baseline_trainer.py`)

### NERTrainer(params, binary_ner_tagger, entityname_predictor, sent_repre_generator=None)
- Two-step trainer for Coach.
- `train_step(X, lengths, y_bin, y_final, templates=None, tem_lengths=None, epoch=None)` → losses tuple
- `evaluate(dataloader, istestset=False)` → `(bin_f1, final_f1, stop_training_flag)`
- `combine_binary_and_entityname_preds(binary_preds_batch, entityname_preds_batch)` → final tag indices

### BaselineTrainer (CT/RZT)
- `train_step(X, lengths, y)` trains CT/RZT predictions per entity; `evaluate(dataloader, istestset=False)` reports final F1.

### BiLSTMCRFTrainer
- Trains `BiLSTMCRFTagger` using CRF NLL; `evaluate` decodes with Viterbi and computes F1.