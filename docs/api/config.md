## Configuration (CLI flags)

The project centralizes configuration via `config.get_params()`. Flags are available to all entry points (e.g., `slu_main.py`, `ner_main.py`, baselines, and tests).

- **--exp_name**: Experiment name. Default: `default`
- **--logger_filename**: Log file name. Default: `cross-domain-slu.log`
- **--dump_path**: Root directory for experiment outputs. Default: `experiments`
- **--exp_id**: Experiment ID (subfolder inside `dump_path/exp_name`). Default: `1`

Adaptation and training
- **--epoch**: Max training epochs. Default: `300`
- **--tgt_dm**: Target domain for SLU. One of AddToPlaylist, BookRestaurant, GetWeather, PlayMusic, RateBook, SearchCreativeWork, SearchScreeningEvent. Default: empty
- **--emb_file**: Embeddings file path. For SLU: `./data/snips/emb/slu_word_char_embs.npy` (or with slot embeddings). For NER: `./data/ner/emb/ner_embs.npy`. Default: `./data/snips/emb/slu_word_char_embs.npy`
- **--emb_dim**: Word embedding dimension used by encoders. Default: `400` (SLU) / set to `300` for NER examples
- **--batch_size**: Training batch size. Default: `32`
- **--num_binslot**: Number of binary labels for sequence tagging (O, B, I). Default: `3`
- **--num_slot**: Number of slot types (SLU final labels). Default: `72`
- **--num_domain**: Number of domains. Default: `7`
- **--freeze_emb**: Freeze embedding layer (flag). Default: `False`

Slot/entity embeddings and model basics
- **--slot_emb_file**: Pickle dict mapping domain→slot embeddings. Default: `./data/snips/emb/slot_word_char_embs_based_on_each_domain.dict`
- **--bidirection**: Use bidirectional LSTM encoders (flag). Default: `False`
- **--lr**: Learning rate. Default: `5e-4`
- **--dropout**: Dropout rate in encoders/modules. Default: `0.3`
- **--hidden_dim**: LSTM hidden size (per direction). Default: `200`
- **--n_layer**: Number of LSTM layers. Default: `2`
- **--early_stop**: Stop after this many dev epochs without improvement. Default: `5`
- **--binary**: Train binary-only models (internal flag). Default: `False`

Template regularization (TR)
- **--tr**: Enable template regularization. Default: `False`

Few-shot
- **--n_samples**: Few-shot sample count for target domain (SLU) or target dataset (NER). Default: `0`

Entity-slot encoder for step two
- **--enc_type**: Encoder for entity/slot token spans: `trs` (Transformer), `lstm`, or `none` (sum). Default: `lstm`

Transformer parameters (used when `enc_type=trs`)
- **--num_heads**: Attention heads. Default: `4`
- **--trs_hidden_dim**: Hidden size after span encoder. Default: `400`
- **--filter_size**: FFN hidden size. Default: `64`
- **--dim_key**: Key dimension (0 means equals hidden size). Default: `0`
- **--dim_value**: Value dimension (0 means equals hidden size). Default: `0`
- **--trs_layers**: Number of encoder layers. Default: `1`

Baselines
- **--use_example**: Use example-based features (RZT). Default: `False`
- **--example_emb_file**: Example embeddings dict for SLU RZT baseline. Default: `./data/snips/emb/example_embs_based_on_each_domain.dict`

Testing
- **--model_path**: Path to a saved model `.pth` file. Default: empty
- **--model_type**: Saved model type (`coach`, `ct`, `rzt`, etc.). Default: empty
- **--test_mode**: Test mode (`testset` or `seen_unseen` for SLU). Default: `testset`

NER-specific
- **--ner_entity_type_emb_file**: NER entity type embedding `.npy`. Default: `./data/ner/emb/entity_type_embs.npy`
- **--ner_example_emb_file**: NER example embeddings `.npy`. Default: `./data/ner/emb/example_embs.npy`
- **--bilstmcrf**: Use BiLSTM-CRF baseline (NER). Default: `False`
- **--num_entity_label**: Number of entity labels for BiLSTM-CRF. Default: `9`

Notes
- All scripts use the same flag parser; irrelevant flags are simply ignored by some entry points.
- `--dump_path/--exp_name/--exp_id` decide where models and logs are stored, e.g., `experiments/coach_lstmenc/atp_50/`.