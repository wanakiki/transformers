self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)


num_key_value_heads 是上层传递的参数，决定生成 kv 的个数。如果说是1，那group就是 num_attention_heads，只生成1个kv，复制 group 份。如果等于 num_attention_heads ，那group就是1，不需要复制。

num_key_value_heads 影响了 k_proj v_proj 的输出尺寸





group 决定了 kv attention 被复制的个数


bias 是布尔值

## 问题

iscausal 有什么用？