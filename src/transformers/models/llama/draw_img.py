#!/usr/bin/env python3
"""
Llama 模型计算量分析和可视化脚本

基于 calculations.md 中的计算公式，计算各个函数和整个 decoder 的计算量，
并提供可视化功能来展示计算量分布和随序列长度变化的趋势。

使用说明：

1. 基本计算功能（无需额外依赖）：
   python draw_img.py
   
2. 完整功能（包含可视化，需要安装依赖）：
   pip install matplotlib numpy
   python draw_img.py

3. 自定义配置示例：
   from draw_img import ModelConfig, ComputeWeights, LlamaComputeAnalyzer
   
   config = ModelConfig(hidden_size=4096, num_layers=32)
   weights = ComputeWeights(add=1, mul=2, silu=3)
   analyzer = LlamaComputeAnalyzer(config, weights)
   
   stats = analyzer.compute_full_model()
   print(f"总计算量: {stats['total']}")

功能特性：
- ✅ 支持自定义模型参数
- ✅ 支持自定义操作权重
- ✅ 计算各个函数的计算量
- ✅ 分析计算量随序列长度的变化
- ✅ 生成详细的可视化图表
- ✅ 支持无绘图库的纯计算模式
"""

# 尝试导入绘图库，如果失败则提供纯计算功能
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_PLOTTING_LIBS = True
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    print("Warning: matplotlib and/or numpy not found. Install with: pip install matplotlib numpy")
    print("运行纯计算模式（无可视化功能）")
    HAS_PLOTTING_LIBS = False
    # 提供numpy的基本替代
    class MockNumPy:
        @staticmethod
        def arange(n):
            return list(range(n))
    np = MockNumPy()

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class ModelConfig:
    """模型配置参数 - 默认为 Llama 3 70B 配置"""
    batch_size: int = 1
    seq_len: int = 1
    hidden_size: int = 8192  # Llama 3 70B
    num_attention_heads: int = 64  # Llama 3 70B
    num_key_value_heads: int = 8   # Llama 3 70B (GQA)
    intermediate_size: int = 28672  # Llama 3 70B
    num_layers: int = 80  # Llama 3 70B
    past_seq_len: Optional[int] = None  # 用于增量推理的缓存序列长度
    
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
    softmax: float = 10.0  # softmax 操作相对复杂

class LlamaComputeAnalyzer:
    """Llama 模型计算量分析器"""
    
    def __init__(self, config: ModelConfig, weights: ComputeWeights):
        self.config = config
        self.weights = weights
    
    def compute_rmsnorm(self) -> Dict[str, float]:
        """计算 LlamaRMSNorm 的计算量"""
        cfg = self.config
        
        add_ops = cfg.batch_size * cfg.seq_len * (2 * cfg.hidden_size - 1)
        mul_ops = cfg.batch_size * cfg.seq_len * (2 * cfg.hidden_size + 1)
        pow2_ops = cfg.batch_size * cfg.seq_len * cfg.hidden_size
        rsqrt_ops = cfg.batch_size * cfg.seq_len * cfg.hidden_size
        
        total = (add_ops * self.weights.add + 
                mul_ops * self.weights.mul + 
                pow2_ops * self.weights.pow2 + 
                rsqrt_ops * self.weights.rsqrt)
        
        return {
            'add': add_ops,
            'mul': mul_ops,
            'pow2': pow2_ops,
            'rsqrt': rsqrt_ops,
            'total': total
        }
    
    def compute_rope(self) -> Dict[str, float]:
        """计算 apply_rotary_pos_emb 的计算量"""
        cfg = self.config
        
        add_ops = cfg.batch_size * cfg.seq_len * cfg.head_dim * (cfg.num_attention_heads + cfg.num_key_value_heads) * 1
        mul_ops = cfg.batch_size * cfg.seq_len * cfg.head_dim * (cfg.num_attention_heads + cfg.num_key_value_heads) * 2
        
        total = add_ops * self.weights.add + mul_ops * self.weights.mul
        
        return {
            'add': add_ops,
            'mul': mul_ops,
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
    
    def compute_attention_core(self, past_seq_len: Optional[int] = None) -> Dict[str, float]:
        """计算核心注意力计算量（eager_attention_forward）"""
        cfg = self.config
        seq_len = past_seq_len if past_seq_len is not None else cfg.seq_len
        
        # 对于单token推理，注意力计算简化
        if cfg.seq_len == 1 and past_seq_len is not None:
            # 增量推理：只需计算新token(1个) 与 past_tokens(past_seq_len个) 的attention
            # Q(1) @ K(past_seq_len)^T: batch * num_heads * 1 * past_seq_len
            qk_mul_ops = cfg.batch_size * cfg.num_attention_heads * 1 * past_seq_len
            # QK求和: batch * num_heads * 1 * (past_seq_len - 1) (如果past_seq_len > 1)
            qk_add_ops = cfg.batch_size * cfg.num_attention_heads * 1 * max(past_seq_len - 1, 0)
            
            # Scale: batch * num_heads * 1 * past_seq_len
            scale_mul_ops = cfg.batch_size * cfg.num_attention_heads * 1 * past_seq_len
            
            # Mask (可选): batch * num_heads * 1 * past_seq_len
            mask_add_ops = cfg.batch_size * cfg.num_attention_heads * 1 * past_seq_len
            
            # Attn(1, past_seq_len) @ V(past_seq_len, head_dim): batch * num_heads * 1 * head_dim
            attnv_mul_ops = cfg.batch_size * cfg.num_attention_heads * 1 * cfg.head_dim
            # AttnV求和: batch * num_heads * 1 * (head_dim - 1) - 实际上这里是做矩阵乘法的求和
            attnv_add_ops = cfg.batch_size * cfg.num_attention_heads * 1 * max(cfg.head_dim - 1, 0)
            
            add_ops = qk_add_ops + mask_add_ops + attnv_add_ops
            mul_ops = qk_mul_ops + scale_mul_ops + attnv_mul_ops
            softmax_ops = cfg.batch_size * cfg.num_attention_heads * 1 * past_seq_len
        else:
            # 完整的注意力计算 - 这里有平方项！
            # Q @ K^T: batch * num_heads * seq_len * seq_len 的乘法
            # 加上 mask: batch * num_heads * seq_len * seq_len 的加法  
            # Softmax: batch * num_heads * seq_len * seq_len 的softmax
            # Attn @ V: batch * num_heads * seq_len * head_dim 的乘法
            
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
    
    def compute_attention(self, past_seq_len: Optional[int] = None) -> Dict[str, float]:
        """计算完整的 LlamaAttention 计算量"""
        cfg = self.config
        
        # QKV 投影
        q_add = cfg.batch_size * cfg.seq_len * cfg.num_attention_heads * cfg.head_dim * (cfg.hidden_size - 1)
        q_mul = cfg.batch_size * cfg.seq_len * cfg.num_attention_heads * cfg.head_dim * cfg.hidden_size
        
        k_add = cfg.batch_size * cfg.seq_len * cfg.num_key_value_heads * cfg.head_dim * (cfg.hidden_size - 1)
        k_mul = cfg.batch_size * cfg.seq_len * cfg.num_key_value_heads * cfg.head_dim * cfg.hidden_size
        
        v_add = cfg.batch_size * cfg.seq_len * cfg.num_key_value_heads * cfg.head_dim * (cfg.hidden_size - 1)
        v_mul = cfg.batch_size * cfg.seq_len * cfg.num_key_value_heads * cfg.head_dim * cfg.hidden_size
        
        # RoPE
        rope_stats = self.compute_rope()
        
        # 注意力核心计算
        attn_core_stats = self.compute_attention_core(past_seq_len)
        
        # 输出投影
        o_add = cfg.batch_size * cfg.seq_len * cfg.hidden_size * (cfg.num_attention_heads * cfg.head_dim - 1)
        o_mul = cfg.batch_size * cfg.seq_len * cfg.num_attention_heads * cfg.head_dim * cfg.hidden_size
        
        total_add = q_add + k_add + v_add + rope_stats['add'] + attn_core_stats['add'] + o_add
        total_mul = q_mul + k_mul + v_mul + rope_stats['mul'] + attn_core_stats['mul'] + o_mul
        total_softmax = attn_core_stats['softmax']
        
        total = (total_add * self.weights.add + 
                total_mul * self.weights.mul + 
                total_softmax * self.weights.softmax)
        
        return {
            'add': total_add,
            'mul': total_mul,
            'softmax': total_softmax,
            'total': total
        }
    
    def compute_decoder_layer(self, past_seq_len: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """计算单个 Decoder Layer 的计算量"""
        cfg = self.config
        
        # 各个组件
        rmsnorm1 = self.compute_rmsnorm()
        attention = self.compute_attention(past_seq_len)
        residual1_add = 2 * cfg.batch_size * cfg.seq_len * cfg.hidden_size
        rmsnorm2 = self.compute_rmsnorm()
        mlp = self.compute_mlp()
        residual2_add = cfg.batch_size * cfg.seq_len * cfg.hidden_size
        
        # 总计
        total_add = (rmsnorm1['add'] + attention['add'] + residual1_add + 
                    rmsnorm2['add'] + mlp['add'] + residual2_add)
        total_mul = rmsnorm1['mul'] + attention['mul'] + rmsnorm2['mul'] + mlp['mul']
        total_silu = mlp['silu']
        total_pow2 = rmsnorm1['pow2'] + rmsnorm2['pow2']
        total_rsqrt = rmsnorm1['rsqrt'] + rmsnorm2['rsqrt']
        total_softmax = attention['softmax']
        
        total_compute = (total_add * self.weights.add + 
                        total_mul * self.weights.mul + 
                        total_silu * self.weights.silu + 
                        total_pow2 * self.weights.pow2 + 
                        total_rsqrt * self.weights.rsqrt + 
                        total_softmax * self.weights.softmax)
        
        return {
            'rmsnorm': {'total': rmsnorm1['total'] + rmsnorm2['total']},
            'attention': {'total': attention['total']},
            'mlp': {'total': mlp['total']},
            'residual': {'total': (residual1_add + residual2_add) * self.weights.add},
            'total': {
                'add': total_add,
                'mul': total_mul,
                'silu': total_silu,
                'pow2': total_pow2,
                'rsqrt': total_rsqrt,
                'softmax': total_softmax,
                'total': total_compute
            }
        }
    
    def compute_full_model(self, past_seq_len: Optional[int] = None) -> Dict[str, float]:
        """计算完整模型的计算量"""
        layer_stats = self.compute_decoder_layer(past_seq_len)
        
        return {
            'rmsnorm': layer_stats['rmsnorm']['total'] * self.config.num_layers,
            'attention': layer_stats['attention']['total'] * self.config.num_layers,
            'mlp': layer_stats['mlp']['total'] * self.config.num_layers,
            'residual': layer_stats['residual']['total'] * self.config.num_layers,
            'total': layer_stats['total']['total'] * self.config.num_layers
        }

class LlamaVisualizer:
    """Llama 模型计算量可视化器"""
    
    def __init__(self, analyzer: LlamaComputeAnalyzer):
        self.analyzer = analyzer
        
    def plot_component_distribution(self, save_path: Optional[str] = None):
        """绘制各组件计算量分布饼图"""
        if not HAS_PLOTTING_LIBS:
            print("绘图功能需要安装 matplotlib 和 numpy")
            return
            
        stats = self.analyzer.compute_full_model()
        
        # 准备数据
        components = ['MLP', 'Attention', 'RMSNorm', 'Residual']
        values = [stats['mlp'], stats['attention'], stats['rmsnorm'], stats['residual']]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # 创建饼图
        fig, ax = plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = ax.pie(values, labels=components, colors=colors,
                                         autopct='%1.1f%%', startangle=90,
                                         textprops={'fontsize': 12})
        
        ax.set_title(f'Llama Model Component Compute Distribution\n'
                    f'(Hidden Size: {self.analyzer.config.hidden_size}, '
                    f'Layers: {self.analyzer.config.num_layers})', 
                    fontsize=14, fontweight='bold')
        
        # 添加图例
        ax.legend(wedges, [f'{comp}: {val:.2e}' for comp, val in zip(components, values)],
                 title="Components", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_function_distribution(self, save_path: Optional[str] = None):
        """绘制各函数计算量分布柱状图"""
        if not HAS_PLOTTING_LIBS:
            print("绘图功能需要安装 matplotlib 和 numpy")
            return
            
        layer_stats = self.analyzer.compute_decoder_layer()
        
        # 准备数据
        functions = ['RMSNorm', 'Attention', 'MLP', 'Residual']
        single_layer_values = [
            layer_stats['rmsnorm']['total'],
            layer_stats['attention']['total'],
            layer_stats['mlp']['total'],
            layer_stats['residual']['total']
        ]
        full_model_values = [v * self.analyzer.config.num_layers for v in single_layer_values]
        
        x = np.arange(len(functions))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 单层对比
        bars1 = ax1.bar(x, single_layer_values, width, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_xlabel('Functions')
        ax1.set_ylabel('Compute Units')
        ax1.set_title('Single Decoder Layer Compute Distribution')
        ax1.set_xticks(x)
        ax1.set_xticklabels(functions)
        ax1.set_yscale('log')
        
        # 添加数值标签
        for bar, val in zip(bars1, single_layer_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2e}', ha='center', va='bottom', fontsize=10)
        
        # 完整模型对比
        bars2 = ax2.bar(x, full_model_values, width, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_xlabel('Functions')
        ax2.set_ylabel('Compute Units')
        ax2.set_title(f'Full Model ({self.analyzer.config.num_layers} Layers) Compute Distribution')
        ax2.set_xticks(x)
        ax2.set_xticklabels(functions)
        ax2.set_yscale('log')
        
        # 添加数值标签
        for bar, val in zip(bars2, full_model_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2e}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_seq_len_scaling(self, seq_lens: List[int], save_path: Optional[str] = None):
        """绘制计算量随序列长度变化的曲线"""
        if not HAS_PLOTTING_LIBS:
            print("绘图功能需要安装 matplotlib 和 numpy")
            return
            
        total_computes = []
        component_computes = {'MLP': [], 'Attention': [], 'RMSNorm': [], 'Residual': []}
        
        original_seq_len = self.analyzer.config.seq_len
        
        for seq_len in seq_lens:
            # 更新序列长度
            self.analyzer.config.seq_len = seq_len
            
            # 计算各组件计算量 - 注意这里要用新的序列长度进行计算
            stats = self.analyzer.compute_full_model()
            total_computes.append(stats['total'])
            component_computes['MLP'].append(stats['mlp'])
            component_computes['Attention'].append(stats['attention'])
            component_computes['RMSNorm'].append(stats['rmsnorm'])
            component_computes['Residual'].append(stats['residual'])
        
        # 恢复原始序列长度
        self.analyzer.config.seq_len = original_seq_len
        
        # 绘制图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 总计算量变化
        ax1.plot(seq_lens, total_computes, 'b-', linewidth=2, marker='o', markersize=6)
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Total Compute Units')
        ax1.set_title('Total Compute vs Sequence Length')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        
        # 各组件计算量变化
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for i, (component, values) in enumerate(component_computes.items()):
            ax2.plot(seq_lens, values, color=colors[i], linewidth=2, 
                    marker='o', markersize=6, label=component)
        
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Compute Units')
        ax2.set_title('Component Compute vs Sequence Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_detailed_breakdown(self, save_path: Optional[str] = None):
        """绘制详细的操作类型分解图"""
        if not HAS_PLOTTING_LIBS:
            print("绘图功能需要安装 matplotlib 和 numpy")
            return
            
        layer_stats = self.analyzer.compute_decoder_layer()
        
        # 操作类型统计
        ops_stats = layer_stats['total']
        operations = ['Add', 'Mul', 'SiLU', 'Pow2', 'Rsqrt', 'Softmax']
        counts = [
            ops_stats['add'],
            ops_stats['mul'], 
            ops_stats['silu'],
            ops_stats['pow2'],
            ops_stats['rsqrt'],
            ops_stats['softmax']
        ]
        
        # 计算加权值
        weights = [
            self.analyzer.weights.add,
            self.analyzer.weights.mul,
            self.analyzer.weights.silu,
            self.analyzer.weights.pow2,
            self.analyzer.weights.rsqrt,
            self.analyzer.weights.softmax
        ]
        
        weighted_values = [c * w for c, w in zip(counts, weights)]
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 操作次数分布
        colors_set3 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462']
        bars1 = ax1.bar(operations, counts, color=colors_set3[:len(operations)])
        ax1.set_ylabel('Operation Count')
        ax1.set_title('Operation Count Distribution (Single Layer)')
        ax1.set_yscale('log')
        for bar, val in zip(bars1, counts):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., val,
                        f'{val:.1e}', ha='center', va='bottom', fontsize=9)
        
        # 加权计算量分布
        bars2 = ax2.bar(operations, weighted_values, color=colors_set3[:len(operations)])
        ax2.set_ylabel('Weighted Compute Units')
        ax2.set_title('Weighted Compute Distribution (Single Layer)')
        ax2.set_yscale('log')
        for bar, val in zip(bars2, weighted_values):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., val,
                        f'{val:.1e}', ha='center', va='bottom', fontsize=9)
        
        # 权重设置
        colors_pastel = ['#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc']
        bars3 = ax3.bar(operations, weights, color=colors_pastel[:len(operations)])
        ax3.set_ylabel('Weight Value')
        ax3.set_title('Operation Weight Settings')
        for bar, val in zip(bars3, weights):
            ax3.text(bar.get_x() + bar.get_width()/2., val,
                    f'{val}', ha='center', va='bottom', fontsize=10)
        
        # 占比饼图
        non_zero_indices = [i for i, v in enumerate(weighted_values) if v > 0]
        non_zero_operations = [operations[i] for i in non_zero_indices]
        non_zero_values = [weighted_values[i] for i in non_zero_indices]
        
        ax4.pie(non_zero_values, labels=non_zero_operations, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Weighted Compute Distribution')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_attention_analysis(self, max_seq_len: int = 2048, step: int = 128, save_path: Optional[str] = None):
        """绘制attention各组件的详细分析"""
        if not HAS_PLOTTING_LIBS:
            print("绘图功能需要安装 matplotlib 和 numpy")
            return
            
        seq_lens = list(range(step, max_seq_len + 1, step))
        
        attention_total_flops = []
        attention_core_flops = []
        attention_linear_flops = []
        mlp_flops = []
        
        original_seq_len = self.analyzer.config.seq_len
        
        for seq_len in seq_lens:
            self.analyzer.config.seq_len = seq_len
            cfg = self.analyzer.config
            
            # 完整attention
            attention_stats = self.analyzer.compute_attention()
            attention_total_flops.append(attention_stats['total'] / 1e9)
            
            # 核心attention（平方项）
            core_stats = self.analyzer.compute_attention_core()
            attention_core_flops.append(core_stats['total'] / 1e9)
            
            # attention中的线性项（QKV投影 + 输出投影）
            qkv_ops = (cfg.batch_size * cfg.seq_len * 
                      (cfg.num_attention_heads + 2 * cfg.num_key_value_heads) * 
                      cfg.head_dim * cfg.hidden_size) * self.analyzer.weights.mul
            o_ops = (cfg.batch_size * cfg.seq_len * cfg.hidden_size * 
                    cfg.num_attention_heads * cfg.head_dim) * self.analyzer.weights.mul
            attention_linear_flops.append((qkv_ops + o_ops) / 1e9)
            
            # MLP用于对比
            mlp_stats = self.analyzer.compute_mlp()
            mlp_flops.append(mlp_stats['total'] / 1e9)
        
        # 恢复原始配置
        self.analyzer.config.seq_len = original_seq_len
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # 左图：Attention详细分解
        ax1.plot(seq_lens, attention_core_flops, 'purple', linewidth=2.5, 
                label='Attention核心 (O(n²))', marker='d', markersize=5)
        ax1.plot(seq_lens, attention_linear_flops, 'orange', linewidth=2.5, 
                label='Attention线性投影 (O(n))', marker='x', markersize=6)
        ax1.plot(seq_lens, attention_total_flops, 'blue', linewidth=2.5, 
                label='Attention总计', marker='o', markersize=5)
        
        ax1.set_xlabel('序列长度', fontsize=12)
        ax1.set_ylabel('计算量 (GFLOP)', fontsize=12)
        ax1.set_title('Attention内部组件分析', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        
        # 右图：Attention vs MLP对比
        ax2.plot(seq_lens, attention_core_flops, 'purple', linewidth=2.5, 
                label='Attention核心 (O(n²))', marker='d', markersize=5)
        ax2.plot(seq_lens, mlp_flops, 'red', linewidth=2.5, 
                label='MLP (O(n))', marker='s', markersize=5)
        ax2.plot(seq_lens, attention_total_flops, 'blue', linewidth=2.5, 
                label='Attention总计', marker='o', markersize=5)
        
        # 添加理论曲线用于对比
        if len(seq_lens) >= 2:
            base_seq = seq_lens[0]
            base_core = attention_core_flops[0]
            base_mlp = mlp_flops[0]
            
            theoretical_quadratic = [base_core * (s/base_seq)**2 for s in seq_lens]
            theoretical_linear = [base_mlp * (s/base_seq) for s in seq_lens]
            
            ax2.plot(seq_lens, theoretical_quadratic, '--', color='gray', alpha=0.7, 
                    label='理论O(n²)', linewidth=1.5)
            ax2.plot(seq_lens, theoretical_linear, '--', color='lightcoral', alpha=0.7, 
                    label='理论O(n)', linewidth=1.5)
        
        ax2.set_xlabel('序列长度', fontsize=12)
        ax2.set_ylabel('计算量 (GFLOP)', fontsize=12)
        ax2.set_title('Attention vs MLP 增长趋势对比', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 输出数值分析
        print("\n=== Attention 增长趋势分析 ===")
        print("序列长度\tAttention核心\t增长倍数\t理论倍数")
        for i, seq_len in enumerate(seq_lens):
            if i > 0:
                actual_ratio = attention_core_flops[i] / attention_core_flops[i-1]
                seq_ratio = seq_lens[i] / seq_lens[i-1]
                theoretical_ratio = seq_ratio ** 2
                print(f"{seq_len}\t\t{attention_core_flops[i]:.2f}\t\t{actual_ratio:.2f}x\t\t{theoretical_ratio:.2f}x")
            else:
                print(f"{seq_len}\t\t{attention_core_flops[i]:.2f}\t\t-\t\t-")

def demonstrate_calculations():
    """演示计算功能（不需要绘图库）"""
    print("\n=== 计算功能演示 ===")
    
    # 测试不同的配置
    configs = [
        {"name": "Llama 3 70B", "hidden_size": 8192, "num_layers": 80, "intermediate_size": 28672, "num_attention_heads": 64, "num_key_value_heads": 8},
        {"name": "Llama 3 8B", "hidden_size": 4096, "num_layers": 32, "intermediate_size": 14336, "num_attention_heads": 32, "num_key_value_heads": 8},
        {"name": "Llama 2 7B-like", "hidden_size": 4096, "num_layers": 32, "intermediate_size": 14336, "num_attention_heads": 32, "num_key_value_heads": 32},
        {"name": "Small Model", "hidden_size": 768, "num_layers": 12, "intermediate_size": 3072, "num_attention_heads": 12, "num_key_value_heads": 12},
    ]
    
    weights = ComputeWeights()
    
    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        config = ModelConfig(
            hidden_size=cfg['hidden_size'],
            num_layers=cfg['num_layers'],
            intermediate_size=cfg['intermediate_size'],
            num_attention_heads=cfg.get('num_attention_heads', cfg['hidden_size'] // 128),
            num_key_value_heads=cfg.get('num_key_value_heads', cfg.get('num_attention_heads', cfg['hidden_size'] // 128))
        )
        
        analyzer = LlamaComputeAnalyzer(config, weights)
        
        # 单层分析
        layer_stats = analyzer.compute_decoder_layer()
        print(f"单层计算量: {layer_stats['total']['total']:.2e}")
        
        # 各组件占比
        components = ['rmsnorm', 'attention', 'mlp', 'residual']
        print("组件分布:")
        total = sum(layer_stats[comp]['total'] for comp in components)
        for comp in components:
            percentage = layer_stats[comp]['total'] / total * 100
            print(f"  {comp:10s}: {percentage:5.1f}%")
        
        # 完整模型
        full_stats = analyzer.compute_full_model()
        print(f"完整模型计算量: {full_stats['total']:.2e}")
    
    print("\n=== 序列长度影响分析 ===")
    config = ModelConfig()  # 默认配置
    analyzer = LlamaComputeAnalyzer(config, weights)
    
    if HAS_PLOTTING_LIBS:
        print("生成可视化图表...")
        visualizer = LlamaVisualizer(analyzer)
        
        # 生成attention详细分析 - 扩展到 8192 上下文长度
        print("绘制 Attention 组件详细分析...")
        visualizer.plot_attention_analysis(max_seq_len=8192, step=512)
        
        # 生成序列长度影响图 - 覆盖到 8192 tokens
        seq_lens = [512, 1024, 2048, 4096, 8192]
        print("绘制序列长度影响分析...")
        visualizer.plot_seq_len_scaling(seq_lens)
        
        # 4. 详细操作分解
        visualizer.plot_detailed_breakdown()
        
        # 5. 注意力分析 - 扩展到 8192 上上下文长度
        visualizer.plot_attention_analysis(max_seq_len=8192, step=512)
        
        print("可视化图表生成完成！")
    else:
        print("进行数值分析（无可视化）...")
        seq_lens = [512, 1024, 2048, 4096, 8192]  # 扩展到 8192 上下文长度
        
        print("Attention各组件计算量分析:")
        print("seq_len\tAttention总计\t核心Attention\t线性投影\t\tMLP")
        for seq_len in seq_lens:
            config.seq_len = seq_len
            analyzer = LlamaComputeAnalyzer(config, weights)
            
            # 各组件计算量
            attention_stats = analyzer.compute_attention()
            core_stats = analyzer.compute_attention_core()
            mlp_stats = analyzer.compute_mlp()
            
            # 线性投影部分
            qkv_ops = (config.batch_size * config.seq_len * 
                      (config.num_attention_heads + 2 * config.num_key_value_heads) * 
                      config.head_dim * config.hidden_size) * weights.mul
            o_ops = (config.batch_size * config.seq_len * config.hidden_size * 
                    config.num_attention_heads * config.head_dim) * weights.mul
            linear_ops = qkv_ops + o_ops
            
            print(f"{seq_len}\t\t{attention_stats['total']:.2e}\t{core_stats['total']:.2e}\t{linear_ops:.2e}\t{mlp_stats['total']:.2e}")
        
        print("\n核心Attention增长趋势验证:")
        print("seq_len\t核心计算量\t\t增长倍数\t理论倍数")
        prev_core = None
        prev_seq = None
        for seq_len in seq_lens:
            config.seq_len = seq_len
            analyzer = LlamaComputeAnalyzer(config, weights)
            core_stats = analyzer.compute_attention_core()
            
            if prev_core is not None and prev_seq is not None:
                actual_ratio = core_stats['total'] / prev_core
                theoretical_ratio = (seq_len / prev_seq) ** 2
                print(f"{seq_len}\t\t{core_stats['total']:.2e}\t\t{actual_ratio:.2f}x\t\t{theoretical_ratio:.2f}x")
            else:
                print(f"{seq_len}\t\t{core_stats['total']:.2e}\t\t-\t\t-")
            
            prev_core = core_stats['total']
            prev_seq = seq_len
    print("序列长度分析 (显示各组件计算量):")
    print(f"{'SeqLen':>6} | {'Total':>12} | {'MLP':>12} | {'Attention':>12} | {'RMSNorm':>12} | {'Residual':>12}")
    print("-" * 80)
    
    for seq_len in seq_lens:
        config.seq_len = seq_len
        layer_stats = analyzer.compute_decoder_layer()
        full_stats = analyzer.compute_full_model()
        
        print(f"{seq_len:6d} | {full_stats['total']:12.2e} | {full_stats['mlp']:12.2e} | "
              f"{full_stats['attention']:12.2e} | {full_stats['rmsnorm']:12.2e} | {full_stats['residual']:12.2e}")
    
    # 显示增长趋势
    print(f"\n序列长度增长趋势分析:")
    print("(以seq_len=1为基准)")
    config.seq_len = 1
    base_stats = analyzer.compute_full_model()
    
    for seq_len in [512, 1024, 2048, 4096, 8192]:  # 扩展到 8192 上下文长度
        config.seq_len = seq_len
        current_stats = analyzer.compute_full_model()
        
        attention_ratio = current_stats['attention'] / base_stats['attention']
        mlp_ratio = current_stats['mlp'] / base_stats['mlp']
        
        print(f"seq_len={seq_len:4d}: Attention增长 {attention_ratio:6.1f}x, MLP增长 {mlp_ratio:6.1f}x")
        
        # 理论分析
        theoretical_attention_growth = seq_len * seq_len  # O(n^2) for attention
        theoretical_mlp_growth = seq_len  # O(n) for MLP
        print(f"             理论: Attention {theoretical_attention_growth:6.1f}x, MLP {theoretical_mlp_growth:6.1f}x")

def analyze_llama3_comparison():
    """对比分析 Llama 3 70B 和 8B 版本"""
    print("\n=== Llama 3 模型对比分析 ===")
    
    weights = ComputeWeights()
    
    # Llama 3 70B 配置
    config_70b = ModelConfig(
        hidden_size=8192,
        num_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=28672
    )
    
    # Llama 3 8B 配置
    config_8b = ModelConfig(
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336
    )
    
    analyzer_70b = LlamaComputeAnalyzer(config_70b, weights)
    analyzer_8b = LlamaComputeAnalyzer(config_8b, weights)
    
    print("模型参数对比:")
    print(f"{'参数':<20} {'Llama 3 8B':<15} {'Llama 3 70B':<15} {'倍数':<10}")
    print("-" * 65)
    print(f"{'Hidden Size':<20} {config_8b.hidden_size:<15} {config_70b.hidden_size:<15} {config_70b.hidden_size/config_8b.hidden_size:<10.1f}")
    print(f"{'Layers':<20} {config_8b.num_layers:<15} {config_70b.num_layers:<15} {config_70b.num_layers/config_8b.num_layers:<10.1f}")
    print(f"{'Attention Heads':<20} {config_8b.num_attention_heads:<15} {config_70b.num_attention_heads:<15} {config_70b.num_attention_heads/config_8b.num_attention_heads:<10.1f}")
    print(f"{'Intermediate Size':<20} {config_8b.intermediate_size:<15} {config_70b.intermediate_size:<15} {config_70b.intermediate_size/config_8b.intermediate_size:<10.1f}")
    
    # 不同序列长度下的计算量对比
    seq_lens = [512, 1024, 2048, 4096, 8192]
    
    print(f"\n计算量对比 (序列长度 vs 总计算量):")
    print(f"{'Seq Len':<8} {'Llama 3 8B':<15} {'Llama 3 70B':<15} {'倍数':<10}")
    print("-" * 55)
    
    for seq_len in seq_lens:
        config_8b.seq_len = seq_len
        config_70b.seq_len = seq_len
        
        stats_8b = analyzer_8b.compute_full_model()
        stats_70b = analyzer_70b.compute_full_model()
        
        ratio = stats_70b['total'] / stats_8b['total']
        print(f"{seq_len:<8} {stats_8b['total']:<15.2e} {stats_70b['total']:<15.2e} {ratio:<10.1f}")
    
    # Attention vs MLP 占比对比
    print(f"\n组件占比对比 (seq_len=2048):")
    config_8b.seq_len = 2048
    config_70b.seq_len = 2048
    
    stats_8b = analyzer_8b.compute_full_model()
    stats_70b = analyzer_70b.compute_full_model()
    
    print(f"{'组件':<12} {'Llama 3 8B':<15} {'Llama 3 70B':<15}")
    print("-" * 45)
    
    for component in ['attention', 'mlp']:
        pct_8b = stats_8b[component] / stats_8b['total'] * 100
        pct_70b = stats_70b[component] / stats_70b['total'] * 100
        print(f"{component.capitalize():<12} {pct_8b:<15.1f}% {pct_70b:<15.1f}%")
    
    # 序列长度对内存/计算的影响
    print(f"\n序列长度对计算量增长的影响:")
    base_seq = 512
    config_8b.seq_len = base_seq
    config_70b.seq_len = base_seq
    base_8b = analyzer_8b.compute_full_model()['total']
    base_70b = analyzer_70b.compute_full_model()['total']
    
    print(f"{'Seq Len':<8} {'8B 增长倍数':<12} {'70B 增长倍数':<15}")
    print("-" * 40)
    
    for seq_len in [1024, 2048, 4096, 8192]:
        config_8b.seq_len = seq_len
        config_70b.seq_len = seq_len
        
        current_8b = analyzer_8b.compute_full_model()['total']
        current_70b = analyzer_70b.compute_full_model()['total']
        
        growth_8b = current_8b / base_8b
        growth_70b = current_70b / base_70b
        
        print(f"{seq_len:<8} {growth_8b:<12.1f} {growth_70b:<15.1f}")

def analyze_long_sequence_scaling():
    """分析极长序列（8192 tokens）的计算量特性"""
    print("\n=== 极长序列计算量分析 (Llama 3 70B) ===")
    
    config = ModelConfig()  # Llama 3 70B 默认配置
    weights = ComputeWeights()
    analyzer = LlamaComputeAnalyzer(config, weights)
    
    # 分析从短序列到极长序列的变化
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192]
    
    print("序列长度对各组件计算量的影响:")
    print(f"{'Seq Len':<8} {'Attention核心':<15} {'Attention线性':<15} {'MLP':<15} {'Attention/MLP':<12}")
    print("-" * 75)
    
    for seq_len in seq_lens:
        config.seq_len = seq_len
        
        # 计算各组件
        attention_stats = analyzer.compute_attention()
        core_stats = analyzer.compute_attention_core()
        mlp_stats = analyzer.compute_mlp()
        
        # 线性投影部分
        qkv_ops = (config.batch_size * config.seq_len * 
                  (config.num_attention_heads + 2 * config.num_key_value_heads) * 
                  config.head_dim * config.hidden_size) * weights.mul
        o_ops = (config.batch_size * config.seq_len * config.hidden_size * 
                config.num_attention_heads * config.head_dim) * weights.mul
        linear_ops = qkv_ops + o_ops
        
        ratio = attention_stats['total'] / mlp_stats['total']
        
        print(f"{seq_len:<8} {core_stats['total']:<15.2e} {linear_ops:<15.2e} {mlp_stats['total']:<15.2e} {ratio:<12.3f}")
    
    # 分析attention内部平方项 vs 线性项的转折点
    print(f"\nAttention 内部组件占比变化:")
    print(f"{'Seq Len':<8} {'核心占比%':<12} {'线性占比%':<12} {'核心是否占主导':<15}")
    print("-" * 50)
    
    for seq_len in seq_lens:
        config.seq_len = seq_len
        
        core_stats = analyzer.compute_attention_core()
        attention_stats = analyzer.compute_attention()
        
        core_ratio = core_stats['total'] / attention_stats['total'] * 100
        linear_ratio = 100 - core_ratio
        dominant = "是" if core_ratio > 50 else "否"
        
        print(f"{seq_len:<8} {core_ratio:<12.1f} {linear_ratio:<12.1f} {dominant:<15}")
    
    # 内存和计算复杂度分析
    print(f"\n复杂度增长分析 (相对于 seq_len=512):")
    base_seq = 512
    config.seq_len = base_seq
    base_attention = analyzer.compute_attention()['total']
    base_mlp = analyzer.compute_mlp()['total']
    
    print(f"{'Seq Len':<8} {'Attention增长':<15} {'MLP增长':<12} {'理论Attention':<15} {'理论MLP':<12}")
    print("-" * 70)
    
    for seq_len in [1024, 2048, 4096, 8192]:
        config.seq_len = seq_len
        
        current_attention = analyzer.compute_attention()['total']
        current_mlp = analyzer.compute_mlp()['total']
        
        attention_growth = current_attention / base_attention
        mlp_growth = current_mlp / base_mlp
        
        theoretical_attention = (seq_len / base_seq) ** 2  # O(n²) 理论
        theoretical_mlp = seq_len / base_seq  # O(n) 理论
        
        print(f"{seq_len:<8} {attention_growth:<15.1f} {mlp_growth:<12.1f} {theoretical_attention:<15.1f} {theoretical_mlp:<12.1f}")

def analyze_incremental_inference():
    """分析增量推理（seq_len=1, past_seq_len>0）的计算量节省效果"""
    print("\n=== 增量推理计算量分析 ===")
    
    # 使用 Llama 3 70B 配置
    config = ModelConfig(
        seq_len=1,  # 增量推理时只处理一个新token
        hidden_size=8192,
        num_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=28672
    )
    weights = ComputeWeights()
    analyzer = LlamaComputeAnalyzer(config, weights)
    
    # 分析不同缓存长度下的计算量
    cache_lengths = [128, 512, 1024, 2048, 4096, 8192]
    
    print("增量推理 vs 完整推理计算量对比:")
    print(f"{'缓存长度':<10} {'增量推理':<15} {'完整推理':<15} {'节省比例':<10} {'Attention核心':<15} {'MLP占比':<10}")
    print("-" * 85)
    
    for cache_len in cache_lengths:
        # 增量推理：seq_len=1, past_seq_len=cache_len
        config.seq_len = 1
        config.past_seq_len = cache_len
        incremental_stats = analyzer.compute_full_model(past_seq_len=cache_len)
        
        # 完整推理：seq_len=cache_len+1
        config.seq_len = cache_len + 1
        config.past_seq_len = None
        full_stats = analyzer.compute_full_model()
        
        # 计算节省比例
        savings = (1 - incremental_stats['total'] / full_stats['total']) * 100
        
        # 分析增量推理中attention核心计算量
        config.seq_len = 1
        config.past_seq_len = cache_len
        incremental_attention_core = analyzer.compute_attention_core(past_seq_len=cache_len)
        
        # MLP在增量推理中的占比
        mlp_percentage = incremental_stats['mlp'] / incremental_stats['total'] * 100
        
        print(f"{cache_len:<10} {incremental_stats['total']:<15.2e} {full_stats['total']:<15.2e} "
              f"{savings:<10.1f}% {incremental_attention_core['total']:<15.2e} {mlp_percentage:<10.1f}%")
    
    # 详细分析各组件在增量推理中的占比
    print(f"\n增量推理组件分析 (缓存长度=2048):")
    config.seq_len = 1
    config.past_seq_len = 2048
    
    incremental_stats = analyzer.compute_full_model(past_seq_len=2048)
    
    components = ['attention', 'mlp', 'rmsnorm', 'residual']
    print("组件分布:")
    for comp in components:
        percentage = incremental_stats[comp] / incremental_stats['total'] * 100
        print(f"  {comp.capitalize():<12}: {percentage:5.1f}% ({incremental_stats[comp]:.2e})")
    
    # 分析attention内部组件
    attention_stats = analyzer.compute_attention(past_seq_len=2048)
    core_stats = analyzer.compute_attention_core(past_seq_len=2048)
    
    print(f"\nAttention内部分析 (增量推理, 缓存=2048):")
    core_percentage = core_stats['total'] / attention_stats['total'] * 100
    linear_percentage = 100 - core_percentage
    print(f"  核心Attention: {core_percentage:5.1f}% ({core_stats['total']:.2e})")
    print(f"  线性投影:     {linear_percentage:5.1f}% ({attention_stats['total'] - core_stats['total']:.2e})")
    
    return {
        'cache_lengths': cache_lengths,
        'incremental_stats': [],
        'full_stats': [],
        'savings': []
    }

def plot_incremental_vs_full_analysis(save_path: Optional[str] = None):
    """可视化增量推理 vs 完整推理的计算量对比"""
    if not HAS_PLOTTING_LIBS:
        print("绘图功能需要安装 matplotlib 和 numpy")
        return
    
    config = ModelConfig(hidden_size=8192, num_layers=80, num_attention_heads=64, 
                        num_key_value_heads=8, intermediate_size=28672)
    weights = ComputeWeights()
    analyzer = LlamaComputeAnalyzer(config, weights)
    
    cache_lengths = [128, 512, 1024, 2048, 4096, 8192]
    incremental_totals = []
    full_totals = []
    savings_percentages = []
    incremental_mlp_ratios = []
    incremental_attention_ratios = []
    
    for cache_len in cache_lengths:
        # 增量推理
        config.seq_len = 1
        config.past_seq_len = cache_len
        incremental_stats = analyzer.compute_full_model(past_seq_len=cache_len)
        
        # 完整推理
        config.seq_len = cache_len + 1
        config.past_seq_len = None
        full_stats = analyzer.compute_full_model()
        
        incremental_totals.append(incremental_stats['total'])
        full_totals.append(full_stats['total'])
        savings = (1 - incremental_stats['total'] / full_stats['total']) * 100
        savings_percentages.append(savings)
        
        # 组件占比
        incremental_mlp_ratios.append(incremental_stats['mlp'] / incremental_stats['total'] * 100)
        incremental_attention_ratios.append(incremental_stats['attention'] / incremental_stats['total'] * 100)
    
    # 创建图表
    if not HAS_PLOTTING_LIBS:
        return
    
    import matplotlib.pyplot as plt  # 局部导入确保可用性
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 总计算量对比
    ax1.plot(cache_lengths, incremental_totals, 'green', linewidth=2.5, 
             marker='o', markersize=6, label='增量推理')
    ax1.plot(cache_lengths, full_totals, 'red', linewidth=2.5, 
             marker='s', markersize=6, label='完整推理')
    ax1.set_xlabel('缓存长度')
    ax1.set_ylabel('总计算量')
    ax1.set_title('增量推理 vs 完整推理 - 总计算量对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    
    # 2. 节省比例
    ax2.plot(cache_lengths, savings_percentages, 'blue', linewidth=2.5, 
             marker='d', markersize=6)
    ax2.set_xlabel('缓存长度')
    ax2.set_ylabel('计算量节省比例 (%)')
    ax2.set_title('增量推理计算量节省效果')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # 3. 增量推理组件占比
    ax3.plot(cache_lengths, incremental_mlp_ratios, 'red', linewidth=2.5, 
             marker='o', markersize=6, label='MLP')
    ax3.plot(cache_lengths, incremental_attention_ratios, 'purple', linewidth=2.5, 
             marker='s', markersize=6, label='Attention')
    ax3.set_xlabel('缓存长度')
    ax3.set_ylabel('组件占比 (%)')
    ax3.set_title('增量推理中各组件占比变化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 效率比较（节省倍数）
    efficiency_multipliers = [full / incremental for full, incremental in zip(full_totals, incremental_totals)]
    ax4.plot(cache_lengths, efficiency_multipliers, 'orange', linewidth=2.5, 
             marker='^', markersize=6)
    ax4.set_xlabel('缓存长度')
    ax4.set_ylabel('效率提升倍数')
    ax4.set_title('增量推理效率提升倍数')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出数值分析
    print("\n=== 增量推理效率分析 ===")
    print("缓存长度\t节省比例\t效率提升\tMLP占比\tAttention占比")
    for i, cache_len in enumerate(cache_lengths):
        print(f"{cache_len}\t\t{savings_percentages[i]:.1f}%\t\t{efficiency_multipliers[i]:.1f}x\t\t"
              f"{incremental_mlp_ratios[i]:.1f}%\t\t{incremental_attention_ratios[i]:.1f}%")

def analyze_function_internal_operations(seq_len=1):
    """分析每个函数内部各操作类型的详细分配"""
    print(f"\n=== 各函数内部操作分配分析 (seq_len={seq_len}) ===")
    
    # 创建配置 - 使用 Llama 3 70B 配置  
    config = ModelConfig(seq_len=seq_len)
    weights = ComputeWeights()
    analyzer = LlamaComputeAnalyzer(config, weights)
    
    print(f"模型配置: hidden_size={config.hidden_size}, num_layers={config.num_layers}, seq_len={config.seq_len}")
    print(f"权重配置: Add={weights.add}, Mul={weights.mul}, SiLU={weights.silu}, Pow2={weights.pow2}, Rsqrt={weights.rsqrt}, Softmax={weights.softmax}")
    print()
    
    # 1. RMSNorm 分析
    rmsnorm_stats = analyzer.compute_rmsnorm()
    print("1. RMSNorm 内部操作分配:")
    print(f"   Add 操作:   {rmsnorm_stats['add']:>12,} 次")
    print(f"   Mul 操作:   {rmsnorm_stats['mul']:>12,} 次")
    print(f"   Pow2 操作:  {rmsnorm_stats['pow2']:>12,} 次")
    print(f"   Rsqrt 操作: {rmsnorm_stats['rsqrt']:>12,} 次")
    print(f"   总计算量:   {rmsnorm_stats['total']:>12.2e}")
    print()
    
    # 2. RoPE 分析
    rope_stats = analyzer.compute_rope()
    print("2. RoPE 内部操作分配:")
    print(f"   Add 操作:   {rope_stats['add']:>12,} 次")
    print(f"   Mul 操作:   {rope_stats['mul']:>12,} 次")
    print(f"   总计算量:   {rope_stats['total']:>12.2e}")
    print()
    
    # 3. MLP 分析
    mlp_stats = analyzer.compute_mlp()
    print("3. MLP 内部操作分配:")
    print(f"   Add 操作:   {mlp_stats['add']:>12,} 次")
    print(f"   Mul 操作:   {mlp_stats['mul']:>12,} 次")
    print(f"   SiLU 操作:  {mlp_stats['silu']:>12,} 次")
    print(f"   总计算量:   {mlp_stats['total']:>12.2e}")
    print()
    
    # 4. Attention Core 分析
    attn_core_stats = analyzer.compute_attention_core()
    print("4. Attention Core 内部操作分配:")
    print(f"   Add 操作:    {attn_core_stats['add']:>12,} 次")
    print(f"   Mul 操作:    {attn_core_stats['mul']:>12,} 次")
    print(f"   Softmax 操作: {attn_core_stats['softmax']:>12,} 次")
    print(f"   总计算量:    {attn_core_stats['total']:>12.2e}")
    print()
    
    # 5. Full Attention 分析
    full_attn_stats = analyzer.compute_attention()
    print("5. Full Attention 内部操作分配:")
    print(f"   Add 操作:    {full_attn_stats['add']:>12,} 次")
    print(f"   Mul 操作:    {full_attn_stats['mul']:>12,} 次")
    print(f"   Softmax 操作: {full_attn_stats['softmax']:>12,} 次")
    print(f"   总计算量:    {full_attn_stats['total']:>12.2e}")
    print()
    
    # 6. 单层 Decoder 总体分析
    layer_stats = analyzer.compute_decoder_layer()
    total_stats = layer_stats['total']
    print("6. 单层 Decoder 总体操作分配:")
    print(f"   Add 操作:    {total_stats['add']:>12,} 次")
    print(f"   Mul 操作:    {total_stats['mul']:>12,} 次")
    print(f"   SiLU 操作:   {total_stats['silu']:>12,} 次")
    print(f"   Pow2 操作:   {total_stats['pow2']:>12,} 次")
    print(f"   Rsqrt 操作:  {total_stats['rsqrt']:>12,} 次")
    print(f"   Softmax 操作: {total_stats['softmax']:>12,} 次")
    print(f"   总计算量:    {total_stats['total']:>12.2e}")
    print()
    
    # 7. 操作类型占比分析
    print("7. 单层 Decoder 操作类型占比:")
    total_weighted = total_stats['total']
    
    operations = [
        ('Add', total_stats['add'], weights.add),
        ('Mul', total_stats['mul'], weights.mul),
        ('SiLU', total_stats['silu'], weights.silu),
        ('Pow2', total_stats['pow2'], weights.pow2),
        ('Rsqrt', total_stats['rsqrt'], weights.rsqrt),
        ('Softmax', total_stats['softmax'], weights.softmax)
    ]
    
    for op_name, count, weight in operations:
        weighted_value = count * weight
        percentage = weighted_value / total_weighted * 100 if total_weighted > 0 else 0
        print(f"   {op_name:<8}: {count:>12,} 次 × {weight:>4.1f} = {weighted_value:>12.2e} ({percentage:>5.1f}%)")
    
    print()
    
    # 8. 各函数操作数量对比表格
    print("8. 各函数操作数量对比:")
    print(f"{'函数名':<15} {'Add':<12} {'Mul':<12} {'SiLU':<12} {'Pow2':<12} {'Rsqrt':<12} {'Softmax':<12}")
    print("-" * 90)
    
    functions_data = [
        ('RMSNorm', rmsnorm_stats['add'], rmsnorm_stats['mul'], 0, rmsnorm_stats['pow2'], rmsnorm_stats['rsqrt'], 0),
        ('RoPE', rope_stats['add'], rope_stats['mul'], 0, 0, 0, 0),
        ('MLP', mlp_stats['add'], mlp_stats['mul'], mlp_stats['silu'], 0, 0, 0),
        ('Attention Core', attn_core_stats['add'], attn_core_stats['mul'], 0, 0, 0, attn_core_stats['softmax']),
        ('Full Attention', full_attn_stats['add'], full_attn_stats['mul'], 0, 0, 0, full_attn_stats['softmax'])
    ]
    
    for func_name, add, mul, silu, pow2, rsqrt, softmax in functions_data:
        print(f"{func_name:<15} {add:<12,} {mul:<12,} {silu:<12,} {pow2:<12,} {rsqrt:<12,} {softmax:<12,}")
    
    print()
    
    # 9. 计算量密集型操作分析
    print("9. 计算量密集型操作分析:")
    weighted_ops = []
    for op_name, count, weight in operations:
        if count > 0:
            weighted_value = count * weight
            weighted_ops.append((op_name, count, weight, weighted_value))
    
    # 按加权计算量排序
    weighted_ops.sort(key=lambda x: x[3], reverse=True)
    
    print("   排名    操作类型     操作次数        权重    加权计算量      占比")
    print("-" * 70)
    for i, (op_name, count, weight, weighted_value) in enumerate(weighted_ops, 1):
        percentage = weighted_value / total_weighted * 100
        print(f"   {i:2d}      {op_name:<8} {count:>12,} × {weight:>4.1f} = {weighted_value:>12.2e} ({percentage:>5.1f}%)")
    
    return {
        'rmsnorm': rmsnorm_stats,
        'rope': rope_stats,
        'mlp': mlp_stats,
        'attention_core': attn_core_stats,
        'full_attention': full_attn_stats,
        'decoder_layer': total_stats
    }

# 在现有函数后面添加
def main():
    """主函数：演示脚本功能"""
    
    # 配置模型参数 - Llama 3 70B
    config = ModelConfig(
        batch_size=1,
        seq_len=1,
        hidden_size=8192,  # Llama 3 70B
        num_attention_heads=64,  # Llama 3 70B
        num_key_value_heads=8,   # Llama 3 70B (GQA)
        intermediate_size=28672,  # Llama 3 70B
        num_layers=80  # Llama 3 70B
    )
    
    # 配置计算权重
    weights = ComputeWeights(
        add=1.0,
        mul=2.0,
        silu=3.0,
        pow2=2.0,
        rsqrt=4.0,
        softmax=10.0
    )
    
    # 创建分析器和可视化器
    analyzer = LlamaComputeAnalyzer(config, weights)
    visualizer = LlamaVisualizer(analyzer)
    
    print("=== Llama 模型计算量分析 ===")
    print(f"模型配置: Hidden Size={config.hidden_size}, Layers={config.num_layers}")
    print(f"权重配置: Add={weights.add}, Mul={weights.mul}, SiLU={weights.silu}")
    print()
    
    # 计算并显示统计信息
    full_stats = analyzer.compute_full_model()
    print("完整模型计算量分布:")
    for component, value in full_stats.items():
        if component != 'total':
            percentage = value / full_stats['total'] * 100
            print(f"  {component:10s}: {value:12.2e} ({percentage:5.1f}%)")
    print(f"  {'Total':10s}: {full_stats['total']:12.2e}")
    print()
    
    # 生成可视化
    if HAS_PLOTTING_LIBS:
        print("正在生成可视化图表...")
        
        # 1. 组件分布饼图
        visualizer.plot_component_distribution()
        
        # 2. 函数分布柱状图
        visualizer.plot_function_distribution()
        
        # 3. 序列长度扩展性分析 - 扩展到 8192 上下文长度
        seq_lens = [512, 1024, 2048, 4096, 8192]
        visualizer.plot_seq_len_scaling(seq_lens)
        
        # 4. 详细操作分解
        visualizer.plot_detailed_breakdown()
        
        # 5. 注意力分析 - 扩展到 8192 上上下文长度
        visualizer.plot_attention_analysis(max_seq_len=8192, step=512)
        
        print("可视化图表生成完成！")
    else:
        print("跳过可视化（需要安装 matplotlib 和 numpy）")
        print("如需安装绘图库，请运行: pip install matplotlib numpy")

if __name__ == "__main__":
    print("🚀 Llama 模型计算量分析工具")
    print("=" * 50)
    
    # 首先演示基本计算功能
    demonstrate_calculations()
    
    print("\n" + "=" * 50)
    print("🎨 开始详细分析和可视化...")
    main()
    
    # print("\n" + "=" * 50)
    # # Llama 3 模型对比分析
    # analyze_llama3_comparison()
    
    # print("\n" + "=" * 50)
    # # 极长序列计算量分析
    # analyze_long_sequence_scaling()
    
    # # 额外的 Llama 3 模型对比分析
    # analyze_llama3_comparison()

    print("\n" + "=" * 50)
    # 极长序列计算量分析
    analyze_long_sequence_scaling()
    
    print("\n" + "=" * 50)
    # 增量推理计算量分析
    analyze_incremental_inference()
    
    # 增量推理可视化
    if HAS_PLOTTING_LIBS:
        print("\n生成增量推理可视化图表...")
        plot_incremental_vs_full_analysis()
    else:
        print("\n跳过增量推理可视化（需要安装 matplotlib 和 numpy）")

    print("\n" + "=" * 50)
    # 增量推理分析
    analyze_incremental_inference()
    
    print("\n" + "=" * 50)
    # 增量推理可视化对比
    plot_incremental_vs_full_analysis()
    
    print("\n" + "=" * 50)
    # 函数内部操作分配分析
    print("🔍 函数内部操作分配分析...")
    analyze_function_internal_operations(seq_len=1)
    analyze_function_internal_operations(seq_len=1024)
