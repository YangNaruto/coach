## CoNLL-2002 metrics (`src/conll2002_metrics.py`)

Public helpers
- `evaluate(lines, options=None)` → internal counts
- `metrics(counts)` → overall/by-type precision/recall/F1
- `report(counts, out=None)` → returns a dict like `{"fb1": float}`
- `conll2002_measure(lines, verbose=False)` → convenience wrapper returning `{"fb1": float}` for a list of `"w PRED GOLD"` lines

Example
```python
from src.conll2002_metrics import conll2002_measure

lines = [
  "w B-LOC B-LOC",
  "w O O",
  "w I-LOC I-LOC",
]
result = conll2002_measure(lines)
print(result["fb1"])  # 0..100
```