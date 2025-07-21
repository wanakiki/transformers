self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)


num_key_value_heads 是上层传递的参数，决定生成 kv 的个数。如果说是1，那group就是 num_attention_heads，只生成1个kv，复制 group 份。如果等于 num_attention_heads ，那group就是1，不需要复制。

num_key_value_heads 影响了 k_proj v_proj 的输出尺寸





group 决定了 kv attention 被复制的个数


bias 是布尔值

## 问题

iscausal 有什么用？


causal 主要是在训练过程中使用的

## 优化

部分运算会进行与常量的操作



## KV Cache

KV Cache 只会在 Attention 中生效，所以其他地方可以正常以 seq_len 1 进行推理

            KV Cache 的核心优化逻辑：
            1. 输入分析：
               - 当前 key_states, value_states 形状: (batch, num_heads, current_seq_len, head_dim)
               - current_seq_len 在 decode 阶段通常为 1（单个新token）
               - 缓存中包含历史所有 token 的 K/V 表示
            
            2. past_key_value.update() 的工作原理：
               a) 从缓存中取出历史 K/V: (batch, num_heads, past_seq_len, head_dim)
               b) 将新计算的 K/V 拼接到历史 K/V 后面
               c) 更新位置编码以适应新的总序列长度
               d) 返回完整的 K/V: (batch, num_heads, past_seq_len + current_seq_len, head_dim)
            
            3. 计算量对比：
               - 无缓存：需要重新计算所有历史 token 的 K/V 投影
               - 有缓存：只需计算新 token 的 K/V，然后拼接缓存
               
            4. 后续注意力计算的优化：
               - Query 只对应新 token: (batch, num_heads, 1, head_dim)
               - Key/Value 包含所有历史: (batch, num_heads, total_seq_len, head_dim)
               - 注意力矩阵: (batch, num_heads, 1, total_seq_len) 而非 (total_seq_len, total_seq_len)

## 注意力矩阵计算复杂度详解

### O(seq_len²) 的来源分析

**步骤1：矩阵乘法计算注意力分数**
```
attn_weights = torch.matmul(query, key.transpose(-2, -1))
```

**维度分析：**
- Query: (batch, num_heads, seq_len, head_dim)
- Key^T: (batch, num_heads, head_dim, seq_len)
- 结果: (batch, num_heads, seq_len, seq_len)

**计算量分析：**
- 对于每个 (i,j) 位置的注意力分数，需要计算：
  attn_weights[i][j] = sum(query[i][k] * key[j][k] for k in head_dim)
- 这需要 head_dim 次乘法和 (head_dim-1) 次加法
- 总共有 seq_len × seq_len 个位置需要计算
- 因此总计算量：seq_len² × head_dim 次乘法

**具体例子：**
假设 seq_len=4, head_dim=3
```
Q = [[q1_1, q1_2, q1_3],    K^T = [[k1_1, k2_1, k3_1, k4_1],
     [q2_1, q2_2, q2_3],          [k1_2, k2_2, k3_2, k4_2],
     [q3_1, q3_2, q3_3],          [k1_3, k2_3, k3_3, k4_3]]
     [q4_1, q4_2, q4_3]]

注意力矩阵 = Q × K^T (4×3) × (3×4) = (4×4)
```

每个元素的计算：
- attn[0][0] = q1_1×k1_1 + q1_2×k1_2 + q1_3×k1_3 (3次乘法)
- attn[0][1] = q1_1×k2_1 + q1_2×k2_2 + q1_3×k2_3 (3次乘法)
- ...
- 总共16个元素，每个3次乘法 = 48次乘法 = 4²×3 = seq_len²×head_dim

**KV Cache 的优化：**
在 decode 阶段，query 只有1行：
```
Q_new = [q_new_1, q_new_2, q_new_3]  # (1, head_dim)
K^T_all = [[k1_1, k2_1, k3_1, k4_1, k5_1],  # 包含历史+新token
           [k1_2, k2_2, k3_2, k4_2, k5_2],
           [k1_3, k2_3, k3_3, k4_3, k5_3]]

注意力矩阵 = (1×3) × (3×5) = (1×5)
```
计算量：1×5×3 = 15次乘法 = O(total_seq_len×head_dim) 而非 O(total_seq_len²×head_dim)
            