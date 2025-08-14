## Transformer (`src/transformer.py`)

### TransformerEncoder
- ctor: `(input_size, hidden_size, num_layers, num_heads, dim_key, dim_value, filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0, attn_dropout=0.0, relu_dropout=0.0, use_mask=False)`
- forward(inputs): `[bsz, length, input_size]` → `[bsz, length, hidden_size]`
- Adds sinusoidal timing signal, linear projection to hidden size, then stacks `EncoderLayer`s with pre-LN.

### EncoderLayer
- Multi-head self-attention + position-wise FFN with residual and LayerNorm.
- ctor: `(hidden_size, total_key_depth, total_value_depth, filter_size, num_heads, bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0)`