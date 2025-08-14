## Usage

### CLI quick start

- **Cross-domain Slot Filling (Coach)**
```bash
python slu_main.py --exp_name coach_lstmenc --exp_id atp_50 --bidirection --freeze_emb --tgt_dm AddToPlaylist --n_samples 50
```
- **Coach + Template Regularization (TR)**
```bash
python slu_main.py --exp_name coach_tr_lstmenc --exp_id atp_50 --bidirection --freeze_emb --tr --tgt_dm AddToPlaylist --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy --n_samples 50
```
- **CT baseline (SLU)**
```bash
python slu_baseline.py --exp_name ct --exp_id atp_50 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 300 --tgt_dm AddToPlaylist --n_samples 50
```
- **RZT baseline (SLU)**
```bash
python slu_baseline.py --exp_name rzt --exp_id atp_50 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 200 --use_example --tgt_dm AddToPlaylist --n_samples 50
```
- **Evaluate Coach (SLU)**
```bash
python slu_test.py --model_path ./experiments/coach_lstmenc/atp_50/best_model.pth --model_type coach --n_samples 50 --tgt_dm AddToPlaylist
```
- **Cross-domain NER (Coach)**
```bash
python ner_main.py --exp_name coach --exp_id ner_0 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --trs_hidden_dim 300 --lr 1e-4
```
- **CT baseline (NER)**
```bash
python ner_baseline.py --exp_name ct --exp_id ner_0 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --lr 1e-4
```
- **RZT baseline (NER)**
```bash
python ner_baseline.py --exp_name rzt --exp_id ner_0 --bidirection --emb_file ./data/ner/emb/ner_embs.npy --emb_dim 300 --lr 1e-4 --hidden_dim 150 --use_example
```

More commands (including ablations and seen/unseen evaluation) are in `run.sh` and the project `README.md`.

### Minimal Python usage (SLU training loop)
```python
from config import get_params
from src.slu.dataloader import get_dataloader
from src.slu.model import BinarySLUTagger, SlotNamePredictor, SentRepreGenerator
from src.slu.trainer import SLUTrainer

params = get_params()
params.tgt_dm = "AddToPlaylist"
params.batch_size = 16
params.tr = False  # template regularization off for this example

tr, val, te, vocab = get_dataloader(params.tgt_dm, params.batch_size, params.tr, params.n_samples)
binary = BinarySLUTagger(params, vocab).cuda()
slotname = SlotNamePredictor(params).cuda()
trainer = SLUTrainer(params, binary, slotname)

for X, lengths, y_bin, y_final, y_dm in tr:
    X, lengths = X.cuda(), lengths.cuda()
    loss_bin, loss_slot = trainer.train_step(X, lengths, y_bin, y_final, y_dm)
    break
```

### Minimal Python usage (NER evaluation)
```python
from config import get_params
from src.ner.dataloader import get_dataloader
from src.ner.model import BinaryNERagger, EntityNamePredictor
from src.ner.trainer import NERTrainer

params = get_params()
params.batch_size = 16
tr, val, te, vocab = get_dataloader(params.batch_size, use_label_encoder=False)

binary = BinaryNERagger(params, vocab).cuda()
entity = EntityNamePredictor(params).cuda()
trainer = NERTrainer(params, binary, entity)

bin_f1, final_f1, _ = trainer.evaluate(val, istestset=False)
print(bin_f1, final_f1)
```