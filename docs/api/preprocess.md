## Preprocessing scripts (`preprocess/`)

### slu_preprocess.py
- `preprocess_slu(datatype)`: Convert dataset split files into SNIPS-style unified files under `data/snips/<Domain>/<Domain>.txt`.
- `generate_seen_and_unseen_data()`: Create `seen_slots.txt` and `unseen_slots.txt` for each domain based on cross-domain seen slot coverage; used by `slu_test.py --test_mode seen_unseen`.

### gen_embeddings_for_slu.py
- Provides `slot_list`, `domain2slot`, and helper functions to construct slot embeddings per domain and combine with character n-gram embeddings.
- `gen_embs_for_vocab()`: Build word embeddings for SLU vocabulary.
- `gen_slot_embs_based_on_each_domain(emb_file)`: Produce domain→slot embeddings dict.
- `combine_word_with_char_embs_for_slot(slotembs_dict_file)` and `combine_word_with_char_embs_for_vocab(wordembs_file)`
- `add_slot_embs_to_slu_embs(slot_embs_file, slu_embs_file)`: Merge slot embeddings into word+char matrix.

### gen_example_emb_for_slu_baseline.py
- `gen_example_embs_based_on_each_domain(emb_file)`: Build per-slot example embeddings dict for RZT baseline.

### gen_embeddings_for_ner.py
- `gen_embs_for_vocab()`: Build NER word embeddings.
- `gen_embs_for_entity_types(emb_file, emb_dim)`: Build entity type embeddings by summing word vectors.
- `gen_example_embs_for_entity_types(emb_file, emb_dim)`: Build example embeddings for NER baselines.

Run as scripts where a `__main__` is present; otherwise, import and call functions from a setup notebook/script.