## Neural modules (`src/modules.py`)

### Lstm(params, vocab)
- Embedding + LSTM encoder.
- ctor args: `params` with `emb_dim`, `hidden_dim`, `n_layer`, `dropout`, `bidirection`, `freeze_emb`, `emb_file`; `vocab` providing `n_words` and word indices.
- forward(X): LongTensor `[bsz, seq_len]` → FloatTensor `[bsz, seq_len, hidden_dim{*2 if bidir}]`

### Attention(attention_size, return_attention=False)
- Token-level attention producing a sentence representation.
- forward(inputs, input_lengths) → `(representations, attentions|None)`

### CRF(num_tags)
- Linear-chain CRF with Viterbi decoding.
- forward(feats): decode best sequence `[bsz, seq_len]`
- loss(feats, tags): negative log-likelihood (scalar)

### MultiHeadAttention(...)
- Standard MHA for transformer blocks.

### PositionwiseFeedForward(...)
- FFN for transformer layers; supports linear/conv variants.

### LayerNorm(features)
- Layer normalization.

Helpers
- `_gen_bias_mask(max_length)`: upper-triangular -inf mask for autoregressive attention.
- `_gen_timing_signal(length, channels)`: sinusoidal positional encodings.