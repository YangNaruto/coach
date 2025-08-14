## SLU models (`src/slu/model.py`)

### BinarySLUTagger(params, vocab)
- Predicts BIO tags (O/B/I) per token, plus exposes CRF utilities.
- forward(X, lengths=None) → `(logits, lstm_hidden)`
  - `logits`: `[bsz, seq_len, 3]`
  - `lstm_hidden`: `[bsz, seq_len, hidden]`
- `crf_decode(inputs, lengths)` → list of length-`len` numpy arrays of predicted BIO indices
- `crf_loss(inputs, lengths, y)` → scalar loss

### SlotNamePredictor(params)
- Given a sequence’s hidden states and BIO predictions, extracts spans and predicts slot names using slot embeddings per domain.
- Supports span encoder: Transformer (`enc_type=trs`), LSTM (`lstm`), or sum (`none`).
- forward(domains, hidden_layers, binary_predictions=None, binary_golds=None, final_golds=None)
  - Train mode: pass `binary_golds` and `final_golds` to return `(pred_slotname_list, gold_slotname_list)`
  - Eval mode: pass `binary_predictions` to return `pred_slotname_list`

### SentRepreGenerator(params, vocab)
- Encodes three templates per sample and the input sequence, returning `[bsz, 3, hidden]` template reps and `[bsz, hidden]` input rep. Used by TR.

### SLUTagger(params, vocab)
- Full slot tagger with CRF over final slot space `y2_set`.
- forward(X) → logits `[bsz, seq_len, num_slot]`
- `crf_decode`/`crf_loss` like above.