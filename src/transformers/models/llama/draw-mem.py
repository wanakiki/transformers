import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional
import json

# --------------------------------------------------------------------------
# Llama 3 70B 的模型配置
# --------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """模型配置参数 - 默认为 Llama 3 70B 配置"""
    batch_size: int = 1
    seq_len: int = 1024  # 用于 prefill 阶段的序列长度
    hidden_size: int = 8192
    num_attention_heads: int = 64
    num_key_value_heads: int = 8   # GQA
    intermediate_size: int = 28672
    num_layers: int = 80
    past_seq_len: int = 1024 # K/V Cache中的序列长度

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度"""
        return self.hidden_size // self.num_attention_heads

# --------------------------------------------------------------------------
# 根据 modeling_llama.py 中的公式计算访存量
# --------------------------------------------------------------------------

def get_memory_access(config: ModelConfig, stage: str):
    """
    计算单个 Decoder Layer 在不同阶段的访存量。

    Args:
        config (ModelConfig): 模型配置。
        stage (str): 'prefill' 或 'decode'。

    Returns:
        dict: 包含各模块读、写、总访存量的字典。
    """
    
    # 根据阶段设置当前的序列长度
    if stage == 'prefill':
        # Prefill 阶段，处理整个输入序列
        current_seq_len = config.seq_len
        # Prefill 阶段没有 K/V cache
        kv_cache_len = 0
    elif stage == 'decode':
        # Decode 阶段，一次只处理一个 token
        current_seq_len = 1
        # K/V cache 包含了 prefill 阶段的所有 token
        kv_cache_len = config.past_seq_len
    else:
        raise ValueError("Stage must be 'prefill' or 'decode'")

    B = config.batch_size
    H = config.hidden_size
    I = config.intermediate_size
    N_a = config.num_attention_heads
    N_kv = config.num_key_value_heads
    D = config.head_dim
    S = current_seq_len
    S_kv = kv_cache_len

    mem_access = {
        'LlamaRMSNorm (Input)': {'read': 0, 'write': 0},
        'QKV Projection': {'read': 0, 'write': 0},
        'RoPE Application': {'read': 0, 'write': 0},
        'K/V Cache I/O': {'read': 0, 'write': 0},
        'Attention Score Calculation': {'read': 0, 'write': 0},
        'Output Projection': {'read': 0, 'write': 0},
        'Residual 1': {'read': 0, 'write': 0},
        'LlamaRMSNorm (Post)': {'read': 0, 'write': 0},
        'LlamaMLP': {'read': 0, 'write': 0},
        'Residual 2': {'read': 0, 'write': 0},
    }

    # 1. Input LayerNorm
    norm_read = B * S * H + H
    norm_write = B * S * H
    mem_access['LlamaRMSNorm (Input)']['read'] = norm_read
    mem_access['LlamaRMSNorm (Input)']['write'] = norm_write

    # --- Attention Block ---
    # 2. QKV Projection
    qkv_proj_read = (B * S * H) + (H * (N_a + 2 * N_kv) * D) # Read hidden_states + Q,K,V weights
    qkv_proj_write = B * S * (N_a + 2 * N_kv) * D # Write Q, K, V tensors
    mem_access['QKV Projection']['read'] = qkv_proj_read
    mem_access['QKV Projection']['write'] = qkv_proj_write

    # 3. RoPE Application
    rope_read = B * S * (N_a + N_kv) * D + 2 * (B * S * D) # Read Q, K + cos, sin
    rope_write = B * S * (N_a + N_kv) * D # Write new Q, K
    mem_access['RoPE Application']['read'] = rope_read
    mem_access['RoPE Application']['write'] = rope_write

    # 4. K/V Cache I/O
    # Prefill: Write the entire sequence to cache. Decode: Read entire cache, write one new token.
    kv_cache_read = B * N_kv * S_kv * D * 2 # Read K and V from cache
    kv_cache_write = B * S * N_kv * D * 2   # Write K and V to cache
    mem_access['K/V Cache I/O']['read'] = kv_cache_read
    mem_access['K/V Cache I/O']['write'] = kv_cache_write

    # 5. Attention Score Calculation (Q @ K.T * V)
    # Read Q, K, V (K,V might be from cache)
    attn_score_read = (B * N_a * S * D) + (B * N_a * (S_kv + S) * D) * 2 # Q, K, V
    # Write attn_weights and attn_output (intermediate)
    attn_score_write = (B * N_a * S * (S + S_kv)) + (B * N_a * S * D)
    mem_access['Attention Score Calculation']['read'] = attn_score_read
    mem_access['Attention Score Calculation']['write'] = attn_score_write

    # 6. Output Projection
    o_proj_read = (B * S * N_a * D) + (H * N_a * D) # Read attention output + weight
    o_proj_write = B * S * H # Write final hidden_state
    mem_access['Output Projection']['read'] = o_proj_read
    mem_access['Output Projection']['write'] = o_proj_write
    
    # 7. Residual Connection 1
    residual_read = 2 * (B * S * H) # Read residual and attention output
    residual_write = B * S * H      # Write result
    mem_access['Residual 1']['read'] = residual_read
    mem_access['Residual 1']['write'] = residual_write

    # 8. Post Attention LayerNorm
    mem_access['LlamaRMSNorm (Post)']['read'] = norm_read
    mem_access['LlamaRMSNorm (Post)']['write'] = norm_write

    # 9. LlamaMLP
    mlp_read = (B * S * H) + (H * I) + (H * I) + (I * H)
    mlp_write = B * S * H
    mem_access['LlamaMLP']['read'] = mlp_read
    mem_access['LlamaMLP']['write'] = mlp_write
    
    # 10. Residual Connection 2
    mem_access['Residual 2']['read'] = residual_read
    mem_access['Residual 2']['write'] = residual_write

    # 计算总和
    for component in mem_access:
        mem_access[component]['total'] = mem_access[component]['read'] + mem_access[component]['write']

    # --- 创建聚合视图 ---
    mem_access_aggregated = {
        'LlamaRMSNorm (Input)': mem_access['LlamaRMSNorm (Input)'],
        'Attention': {'read': 0, 'write': 0, 'total': 0},
        'Residual 1': mem_access['Residual 1'],
        'LlamaRMSNorm (Post)': mem_access['LlamaRMSNorm (Post)'],
        'LlamaMLP': mem_access['LlamaMLP'],
        'Residual 2': mem_access['Residual 2'],
    }
    attention_components_keys = ['QKV Projection', 'RoPE Application', 'K/V Cache I/O', 'Attention Score Calculation', 'Output Projection']
    for comp in attention_components_keys:
        mem_access_aggregated['Attention']['read'] += mem_access[comp]['read']
        mem_access_aggregated['Attention']['write'] += mem_access[comp]['write']
        mem_access_aggregated['Attention']['total'] += mem_access[comp]['total']

    # --- 创建 Attention 详细视图 ---
    mem_access_attention_detail = {key: mem_access[key] for key in attention_components_keys}

    return mem_access_aggregated, mem_access_attention_detail

# --------------------------------------------------------------------------
# 绘图函数
# --------------------------------------------------------------------------

def plot_memory_access(mem_data, stage, seq_len, chart_type):
    """
    为给定的访存数据绘制堆叠条形图。
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    labels = list(mem_data.keys())
    
    read_values = [mem_data[l]['read'] for l in labels]
    write_values = [mem_data[l]['write'] for l in labels]

    # 将数值转换为 GB (假设使用 2 字节浮点数, e.g., float16)
    bytes_per_element = 2
    read_gb = [v * bytes_per_element / (1024**3) for v in read_values]
    write_gb = [v * bytes_per_element / (1024**3) for v in write_values]

    x = np.arange(len(labels))
    width = 0.6

    # --- 创建图表 ---
    fig, ax = plt.subplots(figsize=(18, 10))
    
    title_map = {
        'overall': 'Overall Component-wise Memory I/O',
        'attention_detail': 'Attention Block Memory I/O Breakdown'
    }
    
    fig.suptitle(f'Llama3-70B Memory Access per Layer in {stage.capitalize()} Stage (SeqLen={seq_len}, FP16)', fontsize=20, y=0.98)
    ax.set_title(title_map.get(chart_type, 'Memory I/O'), fontsize=16, pad=20)

    # 创建堆叠条形图
    rects1 = ax.bar(x, read_gb, width, label='Read', color='#4C72B0')
    rects2 = ax.bar(x, write_gb, width, bottom=read_gb, label='Write', color='#DD8452')

    ax.set_ylabel('Memory Access (GB)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 在条形图上添加总数标签
    max_height = max(r + w for r, w in zip(read_gb, write_gb)) if any(r + w > 0 for r, w in zip(read_gb, write_gb)) else 1.0
    for i, (r, w) in enumerate(zip(read_gb, write_gb)):
        total = r + w
        if total > 0:  # 为所有非零条形图添加标签
            # 如果数值很小，使用科学计数法，否则用浮点数
            label_text = f'{total:.4f}' if total >= 0.001 else f'{total:.2e}'
            ax.text(i, total + max_height * 0.01, label_text, ha='center', va='bottom', fontsize=9)


    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(f'memory_access_{stage}_seqlen_{seq_len}_{chart_type}_fp16.png', dpi=300)
    # plt.show() # 移除此行以避免阻塞


# --------------------------------------------------------------------------
# 主函数
# --------------------------------------------------------------------------
def main(seq_len_arg: Optional[int] = None):
    """
    主函数，计算并绘制 prefill 和 decode 阶段的访存图。
    """
    # 如果未提供 seq_len, 使用默认值
    if seq_len_arg is None:
        seq_len = 1024
        print(f"Using default sequence length: {seq_len}")
    else:
        seq_len = seq_len_arg
        print(f"Using sequence length: {seq_len}")

    # --- Prefill Stage ---
    print("\nCalculating for Prefill Stage...")
    config_prefill = ModelConfig(seq_len=seq_len, past_seq_len=0)
    mem_aggregated_prefill, mem_attention_prefill = get_memory_access(config_prefill, 'prefill')
    
    print("Plotting Overall chart for Prefill...")
    plot_memory_access(mem_aggregated_prefill, 'prefill', seq_len, 'overall')
    
    print("Plotting Attention Detail chart for Prefill...")
    plot_memory_access(mem_attention_prefill, 'prefill', seq_len, 'attention_detail')

    # --- Decode Stage ---
    print("\nCalculating for Decode Stage...")
    config_decode = ModelConfig(seq_len=1, past_seq_len=seq_len)
    mem_aggregated_decode, mem_attention_decode = get_memory_access(config_decode, 'decode')

    # 打印 aaggregated 和 attention detail 字典
    print("\n--- Decode Stage Aggregated Memory Access ---")
    print(json.dumps(mem_aggregated_decode, indent=4))
    
    print("\n--- Decode Stage Attention Detail Memory Access ---")
    print(json.dumps(mem_attention_decode, indent=4))

    print("\nPlotting Overall chart for Decode...")
    plot_memory_access(mem_aggregated_decode, 'decode', seq_len, 'overall')

    print("Plotting Attention Detail chart for Decode...")
    plot_memory_access(mem_attention_decode, 'decode', seq_len, 'attention_detail')

if __name__ == '__main__':
    import sys
    # 从命令行参数获取 seq_len
    if len(sys.argv) > 1:
        try:
            sequence_length = int(sys.argv[1])
            main(sequence_length)
        except ValueError:
            print("Error: Please provide a valid integer for sequence length.")
            sys.exit(1)
    else:
        main()
