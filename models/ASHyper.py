import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.data import data as D
from torch.nn import Linear
import torch_scatter
from math import sqrt
from torch_geometric.utils import scatter
import math

class CrossModalHyperedgeInteraction(nn.Module):
    """论文式跨模态超边交互：按目标模态 m 做 cross-attn（m←n）并带残差项。

    对每个目标模态 m：
      A_{m<-n} = softmax( (E_m W_Q)(E_n W_K)^T / sqrt(D) )
      \tilde E_m = sigma( E_m W_O + sum_{n!=m} A_{m<-n} (E_n W_V) )

    输入/输出均为 list[Tensor]，每个 Tensor 形状 [B, K_m, D]。
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, modal_hyperedges):
        """modal_hyperedges: list of [B, K_m, D]."""
        if not isinstance(modal_hyperedges, (list, tuple)):
            raise ValueError("CrossModalHyperedgeInteraction expects a list/tuple of modality hyperedge tensors")
        if len(modal_hyperedges) == 0:
            return []

        outs = []
        scale = math.sqrt(self.d_model)
        for m, E_m in enumerate(modal_hyperedges):
            # Q from target modality
            Q = self.W_Q(E_m)  # [B, K_m, D]
            msg_sum = 0.0
            for n, E_n in enumerate(modal_hyperedges):
                if n == m:
                    continue
                K = self.W_K(E_n)  # [B, K_n, D]
                V = self.W_V(E_n)  # [B, K_n, D]
                # [B, K_m, K_n]
                att = torch.matmul(Q, K.transpose(1, 2)) / scale
                att = F.softmax(att, dim=-1)
                att = self.dropout(att)
                # [B, K_m, D]
                msg_sum = msg_sum + torch.matmul(att, V)

            out = self.W_O(E_m) + msg_sum
            out = F.relu(out)
            outs.append(out)

        return outs


class HypergraphConv(MessagePassing):
    """超图卷积层，支持节点到超边和超边到节点的双向传播"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_attention=True,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0.1,
                 bias=False,
                 multi_head_attention=False):
        super(HypergraphConv, self).__init__(aggr='add')
        self.soft = nn.Softmax(dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        # 是否启用真正的多头注意力（用于 ablation）
        self.multi_head_attention = multi_head_attention

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

            # 多头注意力要求 out_channels 可以被 heads 整除
            if self.multi_head_attention and self.heads > 1 and (out_channels % heads != 0):
                raise ValueError(
                    f"HypergraphConv multi-head attention requires out_channels "
                    f"({out_channels}) divisible by heads ({heads})."
                )

            self.att = Parameter(torch.Tensor(1, heads, 2 * int(out_channels / heads)))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        # 仅当 bias 存在时才初始化，避免 bias=None 时调用 zeros 出错
        if self.bias is not None:
            zeros(self.bias)

    def message(self, x_j, edge_index_i, norm, alpha, edge_weight):
        if norm is not None:
            out = norm[edge_index_i].view(-1, 1, 1) * x_j
        else:
            out = x_j
        if edge_weight is not None:
            out = edge_weight.view(-1, 1, 1) * out
        if alpha is not None:
            out = alpha.unsqueeze(-1) * out
        return out

    def node2edge(self, hyperedge_index, x_node, norm_node=None, alpha=None, edge_weight=None):
        """
        节点到超边的传播：提取超边特征
        hyperedge_index: [2, num_conn], row=node_id, col=edge_id
        x_node: [B, N, d] 节点特征
        norm_node: 节点侧的归一化系数
        alpha: [num_conn] or [B, num_conn, 1] 节点-超边注意力
        return: x_edge: [B, M, d] 超边特征
        """
        x_node = x_node.transpose(0, 1)  # [N, B, d]

        # 设置传播方向
        self.flow = 'source_to_target'

        # 使用propagate进行节点到超边的聚合
        x_edge = self.propagate(
            hyperedge_index,
            x=x_node,
            norm=norm_node,
            alpha=alpha,
            edge_weight=edge_weight,
        )

        # 转置回 [B, M, d] 格式
        x_edge = x_edge.transpose(0, 1)  # [B, M, d]

        return x_edge

    def edge2node(self, hyperedge_index, x_edge, norm_edge=None, alpha=None, edge_weight=None):
        """
        超边到节点的传播：更新节点特征
        hyperedge_index: [2, num_conn], row=node_id, col=edge_id
        x_edge: [B, M, d] 已更新的超边特征
        norm_edge: 超边侧的归一化系数
        alpha: [num_conn] or [B, num_conn, 1] 超边-节点注意力
        return: x_node_updated: [B, N, d]
        """
        # 转置为 [M, B, d] 以适配propagate
        x_edge = x_edge.transpose(0, 1)  # [M, B, d]

        # 设置传播方向
        self.flow = 'target_to_source'

        # 使用propagate进行超边到节点的聚合
        x_node_updated = self.propagate(
            hyperedge_index,
            x=x_edge,
            norm=norm_edge,
            alpha=alpha,
            edge_weight=edge_weight,
        )

        # 转置回 [B, N, d] 格式
        x_node_updated = x_node_updated.transpose(0, 1)  # [B, N, d]

        return x_node_updated

    def forward(self, x, hyperedge_index, edge_weight=None):
        """
        向后兼容的完整双向传播接口
        x: [B, N, in_channels] 原始节点特征
        hyperedge_index: [2, num_conn]
        edge_weight: [num_conn]，用于稀疏加权边的强度
        return: x_updated: [B, N, out_channels], constrain_loss: 标量
        """
        if x.dim() != 3:
            raise ValueError(f"HypergraphConv forward expects x with shape [B, N, C], got {x.shape}")

        batch_size, num_nodes, _ = x.shape

        # 1) 先对节点特征做线性投影
        x_proj = torch.matmul(x, self.weight)  # [B, N, out_channels]

        # 2) 计算节点度和超边度（加权度）
        node_index = hyperedge_index[0]  # [E]
        edge_index = hyperedge_index[1]  # [E]
        num_edges = edge_index.max().item() + 1

        # D_N: node degrees. 论文的 Edge->Node 归一化使用 D_N^{-1}
        if edge_weight is None:
            D = degree(node_index, num_nodes=num_nodes, dtype=x.dtype)  # [N]
        else:
            D = scatter(edge_weight, node_index, dim=0, dim_size=num_nodes, reduce='sum')  # [N]
        D_inv = 1.0 / D
        D_inv[D_inv == float("inf")] = 0
        if edge_weight is None:
            edge_deg = degree(edge_index, num_nodes=num_edges, dtype=x.dtype)  # [M]
        else:
            edge_deg = scatter(edge_weight, edge_index, dim=0, dim_size=num_edges, reduce='sum')  # [M]
        B_norm = 1.0 / edge_deg
        B_norm[B_norm == float("inf")] = 0

        # 3) 计算节点-超边注意力权重 alpha（仅在需要时构造 x_i/x_j）
        alpha = None
        if self.use_attention:
            x1 = x_proj.transpose(0, 1)  # [N, B, out_channels]
            # x_i: 节点特征（按超边连接展开）
            x_i = x1[node_index]  # [E, B, out_channels]

            # 计算每条超边的特征和 (edge_sums)
            edge_sums = torch.zeros(
                num_edges, batch_size, self.out_channels,
                device=x.device, dtype=x.dtype
            )
            edge_sums = edge_sums.index_add(0, edge_index, x_i)  # [M, B, out_channels]

            # x_j: 对应每条 (node, hyperedge) 的超边特征
            x_j = edge_sums[edge_index]  # [E, B, out_channels]

            # 单头注意力（向后兼容）：与当前实现保持完全一致
            if (not self.multi_head_attention) or self.heads == 1:
                # 拼接节点和超边特征，然后与可学习向量 self.att 做点乘
                cat_ij = torch.cat([x_i, x_j], dim=-1)  # [E, B, 2 * out_channels]
                # 将 (1, heads, 2 * out_channels / heads) 展平成 (1, 1, 2 * out_channels)
                att_vec = self.att.view(1, 1, -1)  # [1, 1, 2 * out_channels]
                if att_vec.size(-1) != cat_ij.size(-1):
                    raise ValueError(
                        f"HypergraphConv attention dimension mismatch: "
                        f"att_vec last dim {att_vec.size(-1)} vs cat_ij last dim {cat_ij.size(-1)}"
                    )
                e = (cat_ij * att_vec).sum(dim=-1)  # [E, B]
                e = F.leaky_relu(e, self.negative_slope)

                # 按节点维度规范化（向量化）
                alpha = softmax(e, node_index, num_nodes=num_nodes)  # [E, B]
                alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            else:
                # 多头注意力：在 head 维度上显式建模，然后聚合回单一 alpha
                head_dim = int(self.out_channels / self.heads)
                if self.out_channels % self.heads != 0:
                    raise ValueError(
                        f"HypergraphConv multi-head attention requires out_channels "
                        f"({self.out_channels}) divisible by heads ({self.heads})."
                    )

                # 重塑为 [E, B, heads, head_dim]
                x_i_h = x_i.view(x_i.size(0), batch_size, self.heads, head_dim)
                x_j_h = x_j.view(x_j.size(0), batch_size, self.heads, head_dim)

                # 拼接得到 [E, B, heads, 2 * head_dim]
                cat_ij = torch.cat([x_i_h, x_j_h], dim=-1)

                # self.att: [1, heads, 2 * head_dim] -> [1, 1, heads, 2 * head_dim]
                att_vec = self.att.view(1, self.heads, 2 * head_dim).unsqueeze(1)

                if att_vec.size(-1) != cat_ij.size(-1):
                    raise ValueError(
                        f"HypergraphConv multi-head attention dimension mismatch: "
                        f"att_vec last dim {att_vec.size(-1)} vs cat_ij last dim {cat_ij.size(-1)}"
                    )

                # 点乘得到每个 head 的注意力打分 [E, B, heads]
                e = (cat_ij * att_vec).sum(dim=-1)
                e = F.leaky_relu(e, self.negative_slope)

                # 向量化 softmax：[E, B, heads] -> [E, B * heads]
                e_flat = e.reshape(e.size(0), -1)
                alpha_flat = softmax(e_flat, node_index, num_nodes=num_nodes)
                alpha = alpha_flat.view(e.size(0), batch_size, self.heads)

                # 将多头注意力聚合为单一权重（这里取平均，后续可做 ablation 改为求和等）
                alpha = alpha.sum(dim=-1) / self.heads  # [E, B]
                alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 4) 使用注意力和归一化系数做 Node -> Edge -> Node 双向传播
        x_edge = self.node2edge(
            hyperedge_index=hyperedge_index,
            x_node=x_proj,          # 已投影的节点特征
            norm_node=B_norm,      # 超边侧的归一化
            alpha=alpha,
            edge_weight=edge_weight,
        )
        x_updated = self.edge2node(
            hyperedge_index=hyperedge_index,
            x_edge=x_edge,
            norm_edge=D_inv,        # 节点侧的归一化（D_N^{-1}）
            alpha=alpha,
            edge_weight=edge_weight,
        )

        constrain_losstotal = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x_updated, constrain_losstotal

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class MaskedAdaptiveHypergraphGenerator(nn.Module):
    """
    纯 Masking 流派：全序列构图，通过 Mask 屏蔽 Padding 节点。
    彻底移除 effective_len 截断逻辑，解决索引越界问题。
    """
    def __init__(self, modality, configs):
        super().__init__()
        self.modality = modality
        
        # 配置中的最大长度（仅用于初始化 Embedding 的最大容量）
        # 即使 batch 数据只有 40，这里保留 100 也没关系，切片取前 40 即可
        self.max_seq_len = getattr(configs, f'seq_len_{modality}', getattr(configs, 'seq_len', 100))
        
        self.dim = configs.d_model
        self.hyper_num = getattr(configs, f'hyper_num_{modality}', 50)
        self.alpha = getattr(configs, 'hyper_tau', 1.0)
        self.k = getattr(configs, 'k', 3)
        self.topk = getattr(configs, 'hyper_topk', self.k)

        self.threshold = {'text': 0.001, 'audio': 0.1, 'video': 0.001}.get(self.modality, 0.1)

        # 可学习嵌入 (预留足够大的空间)
        self.node_embeds = nn.Parameter(torch.randn(self.max_seq_len, self.dim))
        self.hyper_embeds = nn.Parameter(torch.randn(self.hyper_num, self.dim))

        self.dropout = nn.Dropout(p=0.1)
        self.cached_hypergraph = None
        self.cached_seq_len = -1  # 记录缓存时的序列长度

    def forward(self, features, mask, update_hyper=True):
        """
        features: [batch_size, current_seq_len, feature_dim]
        mask: [batch_size, current_seq_len] (1=valid, 0=padding)
        """
        batch_size, seq_len, feature_dim = features.shape

        # ==================== 1. 缓存检查逻辑 ====================
        # 只有当 (不需更新) 且 (缓存存在) 且 (缓存的长度 == 当前序列长度) 时，才复用
        # 这完美解决了 batch 间长度不一致导致的越界问题
        if (not update_hyper) and (self.cached_hypergraph is not None) and (self.cached_seq_len == seq_len):
            cached_index, cached_weight = self.cached_hypergraph
            return [(cached_index, cached_weight.detach())] * batch_size

        # ==================== 2. 动态构图逻辑 ====================
        
        # 动态切片：根据当前 batch 的实际长度 seq_len 取嵌入
        # 假设 node_embeds 够长；如果不够(极其罕见)，则取 min
        slice_len = min(seq_len, self.node_embeds.size(0))
        node_embeddings = self.node_embeds[:slice_len]  # [seq_len, dim]
        
        # 如果当前序列比预设的 max_seq_len 还长（防御性编程），需要扩展 features 对应的维度
        # 但通常 seq_len <= max_seq_len。这里直接用 node_embeddings 计算
        
        hyperedge_embeddings = self.hyper_embeds # [hyper_num, dim]

        # 计算相似度 [seq_len, hyper_num]
        similarity = torch.mm(node_embeddings, hyperedge_embeddings.transpose(0, 1))
        
        # 应用温度系数
        similarity = F.relu(self.alpha * similarity) # [seq_len, hyper_num]

        # ==================== 3. 核心 Masking ====================
        # mask: [batch_size, seq_len] -> 取平均或首个样本的 mask (假设 batch 内长度对齐)
        # 注意：通常 PyTorch DataLoader 出来的 batch 内 tensor 长度是对齐的（padding 到了该 batch 的 max）。
        # 我们用 mask 来屏蔽 padding 节点，使其不连接超边。
        
        # 获取当前 batch 的有效 mask (取平均以平滑，或者取 min 这种严格策略)
        # 简单做法：只要该位置大部分样本是 padding，就认为是 padding
        avg_mask = mask.mean(dim=0)[:slice_len] # [seq_len]
        
        # 扩展 mask 维度以匹配 similarity
        # 对应 mask 为 0 的位置（Padding），我们把相似度设为 -inf
        # 这样 Softmax 之后权重为 0
        similarity = similarity.masked_fill(avg_mask.unsqueeze(-1) < 0.5, -1e9)

        # Softmax [seq_len, hyper_num]
        soft_adj = F.softmax(similarity, dim=1) 
        
        # ==================== 4. 稀疏化与输出 ====================
        k = min(self.topk, self.hyper_num)
        values, indices = soft_adj.topk(k, dim=1) # [seq_len, k]

        # 构建 edge_index [2, E]
        # node_indices: 0, 0, 0, 1, 1, 1 ... (完全匹配当前 seq_len)
        node_indices = torch.arange(slice_len, device=features.device).unsqueeze(1).expand(-1, k)
        
        row = node_indices.reshape(-1)
        col = indices.reshape(-1)
        edge_index = torch.stack([row, col], dim=0)

        # 权重扁平化
        edge_weight = values.reshape(-1)

        # 更新缓存状态
        self.cached_hypergraph = (edge_index, edge_weight)
        self.cached_seq_len = seq_len

        return [(edge_index, edge_weight)] * batch_size


class MultimodalClassifier(nn.Module):
    """
    多模态分类器：文本 + 音频 + 视频
    将ASHyper的时间序列预测架构改造为多模态分类
    """
    def __init__(self, configs):
        super(MultimodalClassifier, self).__init__()
        self.configs = configs

        # 模态配置
        self.modalities = ['text', 'audio', 'video']
        # 各模态原始特征维度（由数据集决定）
        self.feature_dims = {
            'text': getattr(configs, 'feature_dim_text', 1024),
            'audio': getattr(configs, 'feature_dim_audio', 1024),
            'video': getattr(configs, 'feature_dim_video', 2048)
        }

        # 模态特定的超图生成器
        self.hyper_generators = nn.ModuleList([
            MaskedAdaptiveHypergraphGenerator(modality, configs)
            for modality in self.modalities
        ])

        # 模态特定的超图卷积 (使用bottleneck结构控制参数量)
        # 不同模态允许不同输入维度，统一映射到 d_model
        # 从配置中读取多头注意力相关超参数，便于做 ablation
        hyper_heads = getattr(configs, 'hyper_heads', 1)
        hyper_multi_head_attention = bool(getattr(configs, 'hyper_multi_head_attention', 0))

        self.hyper_convs = nn.ModuleList([
            HypergraphConv(
                self.feature_dims[modality],
                configs.d_model,
                use_attention=True,
                heads=hyper_heads,
                multi_head_attention=hyper_multi_head_attention,
            )
            for modality in self.modalities
        ])

        # 模态间交互机制（论文式：按模态 m 做 cross-attn：m<-n + 残差）
        self.inter_modal_attention = CrossModalHyperedgeInteraction(
            configs.d_model,
            dropout=getattr(configs, 'dropout', 0.1),
        )

        # ECR参数
        self.kappa = getattr(configs, 'kappa', 0.1)  # ECR权重

        # 动态超图更新频率（每N步更新一次超图结构，减少计算开销）
        self.hyper_update_freq = getattr(configs, 'hyper_update_freq', 1)  # 默认每步更新
        self.step_counter = 0

        # 按照HyperGAMER论文的融合方式 - 6 * d_model (3模态 × 2表示)
        self.task_mode = str(getattr(configs, 'task_mode', 'classification'))
        self.use_coral = bool(getattr(configs, 'use_coral', 0))
        self.num_ordinal_levels = int(getattr(configs, 'num_ordinal_levels', 7))
        if self.task_mode == 'ordinal':
            output_dim = self.num_ordinal_levels - 1
        elif self.task_mode == 'regression':
            output_dim = 1
        else:
            output_dim = getattr(configs, 'num_classes', 10)
        self.use_head_ln = bool(getattr(configs, 'use_head_ln', 0))
        self.use_modal_gate = bool(getattr(configs, 'use_modal_gate', 0)) if self.task_mode == 'regression' else False
        self.use_attn_pooling = bool(getattr(configs, 'use_attn_pooling', 0))
        self.attn_pooling_dropout = float(getattr(configs, 'attn_pooling_dropout', 0.1))
        if self.use_attn_pooling:
            self.attn_pooling_dropout_layer = nn.Dropout(self.attn_pooling_dropout)
            self.attn_queries_node = nn.ParameterList([
                nn.Parameter(torch.randn(configs.d_model)) for _ in self.modalities
            ])
            self.attn_queries_edge = nn.ParameterList([
                nn.Parameter(torch.randn(configs.d_model)) for _ in self.modalities
            ])
        if self.use_head_ln:
            self.head_ln = nn.LayerNorm(6 * configs.d_model)
        if self.use_modal_gate:
            self.modal_gate = nn.Sequential(
                nn.Linear(2 * configs.d_model, configs.d_model),
                nn.ReLU(),
                nn.Linear(configs.d_model, 1)
            )
        self.classifier = nn.Linear(6 * configs.d_model, output_dim)

        # CORAL learnable thresholds (K-1), monotonic via positive deltas
        if self.task_mode == 'ordinal' and self.use_coral:
            init_thresholds = getattr(configs, 'ordinal_thresholds', [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
            if len(init_thresholds) != (self.num_ordinal_levels - 1):
                raise ValueError(
                    f"Expected {self.num_ordinal_levels - 1} thresholds, got {len(init_thresholds)}"
                )
            init_thresholds = torch.tensor(init_thresholds, dtype=torch.float32)
            init_deltas = torch.diff(init_thresholds, prepend=init_thresholds[:1])
            self.coral_deltas = nn.Parameter(init_deltas)
            self.coral_base = nn.Parameter(init_thresholds[:1].clone())

    def _attn_pool(self, seq, mask, query):
        """单头可学习 query attention pooling.
        seq: [B, L, D], mask: [B, L] (1=valid, 0=pad) or None, query: [D]
        """
        scores = torch.matmul(seq, query) / math.sqrt(seq.size(-1))  # [B, L]
        if mask is not None:
            mask_bool = mask > 0
            scores = scores.masked_fill(~mask_bool, -1e9)
        weights = F.softmax(scores, dim=1)
        if mask is not None:
            mask_sum = mask_bool.sum(dim=1, keepdim=True)
            if (mask_sum == 0).any():
                uniform = torch.full_like(weights, 1.0 / seq.size(1))
                weights = torch.where(mask_sum > 0, weights, uniform)
        if self.use_attn_pooling:
            weights = self.attn_pooling_dropout_layer(weights)
        pooled = torch.bmm(weights.unsqueeze(1), seq).squeeze(1)  # [B, D]
        return pooled

    def forward(self, batch_data, x_mark_enc=None):
        """
        前向传播 - 严格按照原ASHyper的融合方式
        batch_data: 包含多模态特征和掩码的字典
        x_mark_enc: 保持兼容性，但不使用
        """
        # 检查输入类型
        if not isinstance(batch_data, dict):
            raise ValueError("MultimodalClassifier only supports multimodal input as dict")

        # 提取各模态特征和掩码
        text_features = batch_data['text_vector']     # [batch_size, 160, 1024]
        audio_features = batch_data['audio_vector']   # [batch_size, 518, 1024]
        video_features = batch_data['video_vector']   # [batch_size, 16, 2048]
        text_mask = batch_data['text_mask']           # [batch_size, 160]
        audio_mask = batch_data['audio_mask']         # [batch_size, 518]
        video_mask = batch_data['video_mask']         # [batch_size, 16]

        batch_size = text_features.size(0)

        # 模态内处理 - 第一阶段：Node -> Edge (提取超边特征)
        modal_hyper_reprs = []  # 各模态的超边特征
        modal_hypergraphs = []  # 各模态的超图结构
        modal_seq_lens = []
        modal_masks = []

        # 检查是否需要更新超图结构
        self.step_counter += 1
        update_hyper = (self.step_counter % self.hyper_update_freq == 0) or (self.step_counter == 1)

        for i, (modality, features, mask) in enumerate([
            ('text', text_features, text_mask),
            ('audio', audio_features, audio_mask),
            ('video', video_features, video_mask),
        ]):
            modal_seq_lens.append(features.shape[1])
            modal_masks.append(mask)
            # 生成模态内超图（动态模式下按频率更新）
            hypergraphs = self.hyper_generators[i](features, mask, update_hyper=update_hyper)
            edge_index, edge_weight = hypergraphs[0]
            edge_index = edge_index.to(features.device)
            edge_weight = edge_weight.to(features.device)
            modal_hypergraphs.append((edge_index, edge_weight))

            # 使用超图卷积的 forward 完整走一遍 Node -> Edge -> Node，并获得约束损失
            updated_nodes_pre, _ = self.hyper_convs[i](
                features,
                edge_index,
                edge_weight=edge_weight,
            )

            # 使用更新后的节点特征再次做 Node -> Edge，用于跨模态超边注意力
            hyper_repr = self.hyper_convs[i].node2edge(
                hyperedge_index=edge_index,
                x_node=updated_nodes_pre,
                edge_weight=edge_weight,
            )
            modal_hyper_reprs.append(hyper_repr)


        # 模态间交互（论文式）：逐模态 cross-attn（m<-n）+ 残差，无需拼接与切分
        if modal_hyper_reprs:
            enhanced_hyper_reprs = self.inter_modal_attention(modal_hyper_reprs)
        else:
            enhanced_hyper_reprs = [
                torch.zeros(batch_size, 1, self.configs.d_model, device=text_features.device)
                for _ in modal_hypergraphs
            ]

        # 模态内处理 - 第二阶段：Edge -> Node (用增强超边更新节点)
        modal_outputs = []
        for i, (hypergraph, enhanced_hyper_repr) in enumerate(zip(modal_hypergraphs, enhanced_hyper_reprs)):
            # 获取模态的序列长度
            seq_len_modality = modal_seq_lens[i]
            edge_index, edge_weight = hypergraph

            # Edge -> Node传播：利用跨模态增强后的超边特征更新节点
            updated_node_features = self.hyper_convs[i].edge2node(
                hyperedge_index=edge_index.to(enhanced_hyper_repr.device),
                x_edge=enhanced_hyper_repr,  # [B, num_edges, d_model]
                edge_weight=edge_weight.to(enhanced_hyper_repr.device),
            )  # 输出: [B, seq_len_modality, d_model]

            # 确保维度匹配（如果seq_len_modality != 超图节点数，可能需要调整）
            if updated_node_features.size(1) != seq_len_modality:
                # 如果不匹配，使用插值或平均扩展
                updated_node_features = updated_node_features.mean(dim=1, keepdim=True).expand(-1, seq_len_modality, -1)

            modal_outputs.append(updated_node_features)

        # 按照HyperGAMER论文的Representation Fusion
        modal_summaries = []
        for i, (hyper_repr, node_output) in enumerate(zip(enhanced_hyper_reprs, modal_outputs)):
            # 节点池化（attention pooling or mean pooling）
            if self.use_attn_pooling:
                z_n = self._attn_pool(node_output, modal_masks[i], self.attn_queries_node[i])
                z_e = self._attn_pool(hyper_repr, None, self.attn_queries_edge[i])
            else:
                z_n = node_output.mean(dim=1)  # [batch_size, d_model]
                z_e = hyper_repr.mean(dim=1)   # [batch_size, d_model]
            # 模态表示拼接
            Z_m = torch.cat([z_n, z_e], dim=1)  # [batch_size, 2 * d_model]
            modal_summaries.append(Z_m)
        # 多模态融合
        if self.use_modal_gate:
            gate_logits = []
            for Z_m in modal_summaries:
                gate_logits.append(self.modal_gate(Z_m))  # [B, 1]
            gate_logits = torch.cat(gate_logits, dim=1)  # [B, 3]
            gate_weights = F.softmax(gate_logits, dim=1)  # [B, 3]
            weighted = [Z_m * gate_weights[:, i].unsqueeze(1) for i, Z_m in enumerate(modal_summaries)]
            Z_fusion = torch.cat(weighted, dim=1)  # [batch_size, 6 * d_model]
        else:
            Z_fusion = torch.cat(modal_summaries, dim=1)  # [batch_size, 6 * d_model]
        if self.use_head_ln:
            Z_fusion = self.head_ln(Z_fusion)
        # 分类输出
        logits = self.classifier(Z_fusion)  # [batch_size, out_dim]

        if self.task_mode == 'ordinal' and self.use_coral:
            deltas = F.softplus(self.coral_deltas)
            thresholds = self.coral_base + torch.cumsum(deltas, dim=0)
            logits = logits - thresholds.view(1, -1)

        # 计算情感一致性正则化 (ECR)
        # 论文：ECR 作用在跨模态交互后的超边表示 \tilde{E}_m 上
        ecr_loss = self.compute_ecr_loss(modal_hypergraphs, enhanced_hyper_reprs, modal_outputs)

        # =========================
        # 论文目标：L_total = L_ER + κ * L_ECR
        # 这里 forward 不包含监督标签，因此不在模型内部计算 L_ER。
        # 训练脚本应当：loss = CE(logits, y) + κ * L_ECR
        # =========================
        reg_loss = self.kappa * ecr_loss

        return logits, reg_loss

    def compute_ecr_loss(self, modal_hypergraphs, modal_hyper_reprs, modal_node_outputs):
        """
        计算情感一致性正则化 (ECR) - 向量化优化版本
        """
        total_ecd = 0.0
        total_epc = 0.0
        num_modalities = len(modal_hypergraphs)

        for m in range(num_modalities):
            # ==================== 修正开始 ====================
            # 1. 正确解包元组 (edge_index, edge_weight)
            edge_index, _ = modal_hypergraphs[m] 
            
            # 2. 确保索引是 Long 类型 (以防万一)
            node_idx = edge_index[0].long()
            edge_idx = edge_index[1].long()
            # ==================== 修正结束 ====================

            hyper_repr = modal_hyper_reprs[m]   # [B, M, D]
            node_repr = modal_node_outputs[m]   # [B, N, D]

            batch_size, num_nodes, _ = node_repr.shape
            _, num_edges, _ = hyper_repr.shape

            # 构建二值关联矩阵 H [B, N, M]
            H = torch.zeros(batch_size, num_nodes, num_edges, device=hyper_repr.device)
            
            # 这里现在使用正确的 node_idx 和 edge_idx 就不会报错了
            H[:, node_idx, edge_idx] = 1.0  # [B, N, M]

            # ECD: 节点-超边距离 - 向量化
            dist_node_hyper = torch.cdist(node_repr, hyper_repr, p=2)  # [B, N, M]

            # 只计算关联的距离，mask掉非关联
            masked_dist = dist_node_hyper * H  # [B, N, M]
            
            # 对每个节点，平均其关联超边的距离
            
            # 假设你有 mask 信息传入 compute_ecr_loss，或者从 node_degrees 推断
            valid_nodes_mask = (node_degrees.squeeze(2) > 1e-7).float() # 度数大于0的才是有效节点
            ecd_per_node = (masked_dist.sum(dim=2) / node_degrees.squeeze(2)) * valid_nodes_mask
            # 只对有效节点求平均
            num_valid_nodes = valid_nodes_mask.sum()
            if num_valid_nodes > 0:
                ecd_m = ecd_per_node.sum() / num_valid_nodes
            else:
                ecd_m = torch.tensor(0.0, device=hyper_repr.device)
            total_ecd += ecd_m

            # EPC: 超边间关系 - 向量化
            if num_edges >= 2:
                # 计算余弦相似度 [B, M, M]
                norm_hyper = F.normalize(hyper_repr, p=2, dim=2)  # [B, M, D]
                cos_sim = torch.bmm(norm_hyper, norm_hyper.transpose(1, 2))  # [B, M, M]

                # 计算欧几里得距离 [B, M, M]
                dist_hyper = torch.cdist(hyper_repr, hyper_repr, p=2)  # [B, M, M]

                # 论文中的margin η，假设为4.2
                eta = 4.2

                # 计算EPC项
                epc_matrix = cos_sim * dist_hyper + (1 - cos_sim) * torch.clamp(eta - dist_hyper, min=0.0)  # [B, M, M]

                # 排除对角线（i==j）
                mask = torch.eye(num_edges, device=hyper_repr.device).unsqueeze(0).bool()  # [1, M, M]
                epc_matrix = epc_matrix.masked_fill(mask, 0.0)

                # 平均
                epc_m = epc_matrix.sum() / (batch_size * num_edges * (num_edges - 1))
                total_epc += epc_m

        # 平均ECD和EPC，λ=0.5
        lambda_ecd = 0.5
        lambda_epc = 0.5
        
        # 防止除以0（虽然通常num_modalities=3）
        if num_modalities > 0:
            ecr_loss = lambda_ecd * (total_ecd / num_modalities) + lambda_epc * (total_epc / num_modalities)
        else:
            ecr_loss = torch.tensor(0.0, device=modal_hyper_reprs[0].device)

        return ecr_loss

    def __repr__(self):
        return f"{self.__class__.__name__}({self.configs.d_model}, {getattr(self.configs, 'num_classes', 10)})"
