## NER models (`src/ner/model.py`)

### BinaryNERagger(params, vocab)
- Predicts BIO entity tags (O/B-Entity/I-Entity) per token.
- forward(X) → `(logits, lstm_hidden)`
- `crf_decode` and `crf_loss` mirror SLU equivalents.

### EntityNamePredictor(params)
- Extract spans via BIO and classify into entity types using learned entity type embeddings (`ner_entity_type_emb_file`).
- Supports span encoder: Transformer/LSTM/sum; train/eval modes similar to SLU’s slot predictor.

### SentRepreGenerator(params, vocab)
- Same as SLU version; encodes templates and input for TR losses.

### ConceptTagger baseline (CT/RZT)
- `ConceptTagger(params, vocab)` in `src/ner/baseline_model.py`: encodes tokens and concatenates entity/type embeddings (and example attention if `--use_example`) then predicts BIO per entity type.
- `BiLSTMCRFTagger(params, vocab)`: classic BiLSTM-CRF sequence tagger over `y2_set`.