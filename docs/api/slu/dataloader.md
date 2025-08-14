## SLU dataloader (`src/slu/dataloader.py`)

Classes
- `Dataset(X, y1, y2, domains, template_list=None)`

Functions
- `collate_fn(data)` → `(padded_seqs, lengths, y1, y2, domains)`
  - `padded_seqs`: LongTensor `[bsz, max_len]` padded with `PAD_INDEX`
  - `lengths`: LongTensor `[bsz]`
  - `y1`, `y2`: python lists (per-sample label sequences)
  - `domains`: LongTensor `[bsz]`
- `collate_fn_for_label_encoder(data)` → `(padded_seqs, lengths, y1, y2, domains, padded_templates, tem_lengths)` when TR is enabled
  - `padded_templates`: LongTensor `[bsz, 3, max_template_len]`
  - `tem_lengths`: LongTensor `[bsz]`
- `get_dataloader(tgt_domain, batch_size, use_label_encoder, n_samples)` → `(dataloader_tr, dataloader_val, dataloader_test, vocab)`
  - For zero-shot, source domains form train; target domain: first 500 → val, rest → test
  - For few-shot, first `n_samples` in target are added to train; target `n_samples:500` → val; rest → test