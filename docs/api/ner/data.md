## NER data and constants (`src/ner/datareader.py`)

Constants
- `y1_set = ["O", "B-Entity", "I-Entity"]`
- `y2_set = ["O", "B-LOC", "I-LOC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]`
- `ENTITY_PAD=0`, `PAD_INDEX=0`, `UNK_INDEX=1`
- Entity types/descriptions and label→description mapping are provided for template generation.

Classes
- `Vocab`: simple incremental vocabulary with `index_words` and maps.

Functions
- `read_file(filepath, vocab, use_label_encoder=False)` → reads CoNLL-style files to token/label sequences. When TR is on, generates 3 template variants per sample.
- `binarize_data(data, vocab, use_label_encoder=False)` → converts to index sequences; includes `template_list` if provided.
- `datareader(use_label_encoder=False)` → returns `(train_bin, val_bin, test_bin, vocab)` from `data/ner/conll2003` and `data/ner/tech`.