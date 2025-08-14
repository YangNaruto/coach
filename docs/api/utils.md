## Core utilities (`src/utils.py`)

### init_experiment(params, logger_filename)
- **Purpose**: Create experiment output directory, persist params, and instantiate a configured logger.
- **Side effects**: Updates `params.dump_path` to the final run directory and writes `params.pkl`.
- **Returns**: `logging.Logger`

Example
```python
from config import get_params
from src.utils import init_experiment

params = get_params()
logger = init_experiment(params, logger_filename=params.logger_filename)
logger.info("experiment ready at %s", params.dump_path)
```

### create_logger(filepath)
- **Purpose**: Configure a `logging.Logger` with console and optional file handlers.
- **Returns**: `logging.Logger`

### get_saved_path(params)
- **Purpose**: Ensure `dump_path/exp_name/exp_id` directories exist and update `params.dump_path` to that leaf.

### load_embedding(vocab, emb_dim, emb_file, oov_emb_file="")
- **Purpose**: Load text embeddings (e.g., fastText) into an array matching `vocab.n_words`.
- **Returns**: `np.ndarray` shaped `[vocab_size, emb_dim]`

### load_embedding_from_npy(emb_file)
- **Purpose**: Load `.npy` embedding matrix.
- **Returns**: `np.ndarray`

### load_embedding_from_pkl(emb_file)
- **Purpose**: Load a pickled Python object (e.g., dict of domain→embeddings).
- **Returns**: deserialized object