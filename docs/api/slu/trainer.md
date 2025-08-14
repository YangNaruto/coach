## SLU trainer (`src/slu/trainer.py`)

### SLUTrainer(params, binary_slu_tagger, slotname_predictor, sent_repre_generator=None)
- Optimizes the two-step SLU model. When `params.tr=True`, also optimizes TR losses via `SentRepreGenerator`.

Methods
- `train_step(X, lengths, y_bin, y_final, y_dm, templates=None, tem_lengths=None, epoch=None)`
  - Returns `(loss_bin, loss_slotname)` or with TR `(loss_bin, loss_slotname, loss_tem0, loss_tem1)`
- `evaluate(dataloader, istestset=False)`
  - Builds binary predictions via CRF decode, predicts slot names, combines to final tags, computes F1 using CoNLL-2002.
  - Returns `(bin_f1, final_f1, stop_training_flag)`; saves best model on dev.
- `combine_binary_and_slotname_preds(dm_id_batch, binary_preds_batch, slotname_preds_batch)`
  - Combines BIO and span predictions into final label indices.
- `save_model()`
  - Saves dict with `binary_slu_tagger` and `slotname_predictor` to `best_model.pth` in `params.dump_path`.