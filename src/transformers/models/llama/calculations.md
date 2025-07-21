# Llama 模型计算量分析

## 1. LlamaRMSNorm 计算量

```python
# calc: 
# data_type convert to float32
# batch * seq_len * hidden_size * pow2 + batch * seq_len * (hidden_size - 1) * add + batch * seq_len * mul
# batch * seq_len * hidden_size * (add + rsqrt) + batch * seq_len * hidden_size * mul
# data type convert
# batch * seq_len * hidden_size * mul
```

## 2. 旋转位置编码 (RoPE) 计算量

### 2.1 rotate_half 函数
```python
# calc:
# data reshape
```

### 2.2 apply_rotary_pos_emb 函数
```python
# calc:
# batch * num_attention_heads * seq_len * head_dim * (2 * add + 4 * mul)
```

## 3. MLP 层计算量分析

### 3.1 完整 MLP 前向传播
```python
# calc:
# up_proj: batch * seq_len * hidden_size * intermediate_size * mul + batch * seq_len * intermediate_size * (hidden_size - 1) * add
# gate_proj: batch * seq_len * hidden_size * intermediate_size * mul + batch * seq_len * intermediate_size * (hidden_size - 1) * add
# act_fn: batch * seq_len * intermediate_size * silu
# *: batch * seq_len * intermediate_size * mul
# down_proj: batch * seq_len * intermediate_size * hidden_size * mul + batch * seq_len * hidden_size * (intermediate_size - 1) * add
```

## 4. 注意力机制计算量

### 4.1 eager_attention_forward 函数
```python
# calc:
# attn_weights: batch * num_attention_heads * seq_len * seq_len * mul + batch * num_attention_heads * seq_len * (seq_len - 1) * add + batch * num_attention_heads * seq_len * seq_len * mul
# mask: batch * num_attention_heads * seq_len * seq_len * add
# softmax: batch * num_attention_heads * seq_len * softmax(seq_len)
# attn_output: batch * num_attention_heads * seq_len * head_dim * mul + batch * num_attention_heads * head_dim * (seq_len - 1) * add
```

### 4.2 LlamaAttention forward 函数
```python
# calc:
# query_states: batch * seq_len * hidden_size * num_attention_heads * head_dim * mul + batch * seq_len * num_attention_heads * head_dim * (hidden_size - 1) * add
# key_states: batch * seq_len * hidden_size * num_key_value_heads * head_dim * mul + batch * seq_len * num_key_value_heads * head_dim * (hidden_size - 1) * add
# value_states: batch * seq_len * hidden_size * num_key_value_heads * head_dim * mul + batch * seq_len * num_key_value_heads * head_dim * (hidden_size - 1) * add
# apply_rotary_pos_emb: batch * num_attention_heads * seq_len * head_dim * (2 * add + 4 * mul)
# attn_weights: eager_attention_forward
# attn_output: batch * seq_len * num_attention_heads * head_dim * hidden_size * mul + batch * seq_len * hidden_size * (num_attention_heads * head_dim - 1) * add
```

## 5. Decoder Layer 完整计算量

```python
# calc:
# input_layernorm: LlamaRMSNorm->forward(hidden_states)
# self_attn: LlamaAttention->forward(hidden_states, attention_mask, position_ids, past_key_value, use_cache, cache_position, position_embeddings)
# residual: 2 * batch * seq_len * hidden_size * add
# post_attention_layernorm: LlamaRMSNorm->forward(hidden_states)
# mlp: LlamaMLP->forward(hidden_states)
```

## 6. 计算量汇总

### 6.1 各模块计算复杂度对比
- **MLP层**: 主导计算量，包含3个大型线性变换
- **注意力层**: 次要计算量，包含4个线性变换 + 注意力计算
- **归一化层**: 相对较小的计算量
- **位置编码**: 最小的计算量

### 6.2 优化要点
- **KV Cache**: 将注意力计算从 O(seq_len²) 降为 O(seq_len)
- **GQA**: 减少 key/value 投影的计算量
- **矩阵乘法优化**: 是所有计算的核心，需要硬件加速
