#!/usr/bin/env python3
"""简化测试脚本，验证序列长度的平方增长关系"""

from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ModelConfig:
    """模型配置参数"""
    batch_size: int = 1
    seq_len: int = 1
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 14336
    num_layers: int = 32
    
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

@dataclass
class ComputeWeights:
    """计算量权重配置"""
    add: float = 1.0
    mul: float = 2.0
    silu: float = 3.0
    pow2: float = 2.0
    rsqrt: float = 4.0
    softmax: float = 10.0

class LlamaComputeAnalyzer:
    """Llama 模型计算量分析器"""
    
    def __init__(self, config: ModelConfig, weights: ComputeWeights):
        self.config = config
        self.weights = weights
    
    def compute_attention_core(self, past_seq_len: Optional[int] = None) -> Dict[str, float]:
        """计算核心注意力计算量（eager_attention_forward）"""
        cfg = self.config
        seq_len = past_seq_len if past_seq_len is not None else cfg.seq_len
        
        if cfg.seq_len == 1 and past_seq_len is not None:
            # 单token推理的简化情况
            add_ops = cfg.batch_size * cfg.num_attention_heads * past_seq_len * (past_seq_len - 1)
            mul_ops = cfg.batch_size * cfg.num_attention_heads * past_seq_len * 2
            add_ops += cfg.batch_size * cfg.num_attention_heads * cfg.head_dim * (past_seq_len - 1)
            mul_ops += cfg.batch_size * cfg.num_attention_heads * cfg.head_dim
            softmax_ops = cfg.batch_size * cfg.num_attention_heads * past_seq_len
        else:
            # 完整的注意力计算 - 这里有平方项！
            # QK计算: batch * num_heads * seq_len * seq_len * mul
            qk_mul_ops = cfg.batch_size * cfg.num_attention_heads * seq_len * seq_len
            # QK求和: batch * num_heads * seq_len * (seq_len - 1) * add  
            qk_add_ops = cfg.batch_size * cfg.num_attention_heads * seq_len * (seq_len - 1)
            
            # Scale: batch * num_heads * seq_len * seq_len * mul
            scale_mul_ops = cfg.batch_size * cfg.num_attention_heads * seq_len * seq_len
            
            # Mask: batch * num_heads * seq_len * seq_len * add
            mask_add_ops = cfg.batch_size * cfg.num_attention_heads * seq_len * seq_len
            
            # AttnV计算: batch * num_heads * seq_len * head_dim * mul
            attnv_mul_ops = cfg.batch_size * cfg.num_attention_heads * seq_len * cfg.head_dim
            # AttnV求和: batch * num_heads * head_dim * (seq_len - 1) * add
            attnv_add_ops = cfg.batch_size * cfg.num_attention_heads * cfg.head_dim * (seq_len - 1)
            
            add_ops = qk_add_ops + mask_add_ops + attnv_add_ops
            mul_ops = qk_mul_ops + scale_mul_ops + attnv_mul_ops
            softmax_ops = cfg.batch_size * cfg.num_attention_heads * seq_len * seq_len
        
        total = (add_ops * self.weights.add + 
                mul_ops * self.weights.mul + 
                softmax_ops * self.weights.softmax)
        
        return {
            'add': add_ops,
            'mul': mul_ops,
            'softmax': softmax_ops,
            'total': total
        }
    
    def compute_mlp(self) -> Dict[str, float]:
        """计算 LlamaMLP 的计算量"""
        cfg = self.config
        
        add_ops = cfg.batch_size * cfg.seq_len * (3 * cfg.hidden_size * cfg.intermediate_size - 2 * cfg.intermediate_size - cfg.hidden_size)
        mul_ops = cfg.batch_size * cfg.seq_len * (3 * cfg.hidden_size * cfg.intermediate_size + cfg.intermediate_size)
        silu_ops = cfg.batch_size * cfg.seq_len * cfg.intermediate_size
        
        total = (add_ops * self.weights.add + 
                mul_ops * self.weights.mul + 
                silu_ops * self.weights.silu)
        
        return {
            'add': add_ops,
            'mul': mul_ops,
            'silu': silu_ops,
            'total': total
        }

def test_scaling():
    """测试序列长度扩展性"""
    config = ModelConfig()
    weights = ComputeWeights()
    analyzer = LlamaComputeAnalyzer(config, weights)
    
    print("序列长度扩展性测试")
    print("=" * 80)
    
    seq_lens = [1, 2, 4, 8, 16, 32, 64]
    
    print(f"{'SeqLen':>6} | {'Attention':>15} | {'MLP':>15} | {'Attn/MLP比':>12} | {'纯O(n²)部分':>15}")
    print("-" * 80)
    
    base_attention = None
    base_mlp = None
    
    for seq_len in seq_lens:
        config.seq_len = seq_len
        
        attention_stats = analyzer.compute_attention_core()
        mlp_stats = analyzer.compute_mlp()
        
        if base_attention is None:
            base_attention = attention_stats['total']
            base_mlp = mlp_stats['total']
        
        attention_ratio = attention_stats['total'] / base_attention
        mlp_ratio = mlp_stats['total'] / base_mlp
        attn_mlp_ratio = attention_stats['total'] / mlp_stats['total']
        
        # 计算纯O(n²)部分的贡献 (QK计算 + Scale + Mask + Softmax)
        cfg = config
        quadratic_ops = (
            cfg.batch_size * cfg.num_attention_heads * seq_len * seq_len * weights.mul +  # QK
            cfg.batch_size * cfg.num_attention_heads * seq_len * seq_len * weights.mul +  # Scale
            cfg.batch_size * cfg.num_attention_heads * seq_len * seq_len * weights.add +  # Mask
            cfg.batch_size * cfg.num_attention_heads * seq_len * seq_len * weights.softmax  # Softmax
        )
        base_quadratic = cfg.batch_size * cfg.num_attention_heads * 1 * 1 * (weights.mul + weights.mul + weights.add + weights.softmax)
        quadratic_ratio = quadratic_ops / base_quadratic
        
        print(f"{seq_len:6d} | {attention_ratio:13.2f}x | {mlp_ratio:13.2f}x | {attn_mlp_ratio:10.6f} | {quadratic_ratio:13.2f}x")
        
        # 验证理论值
        theoretical_attention = seq_len * seq_len  # O(n^2)
        theoretical_mlp = seq_len  # O(n)
        
        print(f"理论值: | {theoretical_attention:13.2f}x | {theoretical_mlp:13.2f}x | {'':10s} | {theoretical_attention:13.2f}x")
        print()

if __name__ == "__main__":
    test_scaling()
