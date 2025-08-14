## NER dataloader (`src/ner/dataloader.py`)

Classes
- `Dataset(X, y1, y2, template_list=None)`

Functions
- `collate_fn(data)` → `(padded_seqs, lengths, y1, y2)`
- `collate_fn_for_label_encoder(data)` → `(padded_seqs, lengths, y1, y2, padded_templates, tem_lengths)`
- `get_dataloader(batch_size, use_label_encoder=False, n_samples=0)` → `(dataloader_tr, dataloader_val, dataloader_test, vocab)`
  - If `n_samples>0`, pulls fixed random indices from target test set into training (few-shot) and removes them from test.