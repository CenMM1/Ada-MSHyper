import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

    def message(self, x_j, edge_index_i, norm, alpha):
        if norm is not None:
            out = norm[edge_index_i].view(-1, 1, 1) * x_j
        else:
            out = x_j
        if alpha is not None:
            out = alpha.unsqueeze(-1) * out
        return out

    def node2edge(self, hyperedge_index, x_node, norm_node=None, alpha=None):
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
        x_edge = self.propagate(hyperedge_index, x=x_node, norm=norm_node, alpha=alpha)

        # 转置回 [B, M, d] 格式
        x_edge = x_edge.transpose(0, 1)  # [B, M, d]

        return x_edge

    def edge2node(self, hyperedge_index, x_edge, norm_edge=None, alpha=None):
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
        x_node_updated = self.propagate(hyperedge_index, x=x_edge, norm=norm_edge, alpha=alpha)

        # 转置回 [B, N, d] 格式
        x_node_updated = x_node_updated.transpose(0, 1)  # [B, N, d]

        return x_node_updated

    def forward(self, x, hyperedge_index):
        """
        向后兼容的完整双向传播接口
        x: [B, N, in_channels] 原始节点特征
        hyperedge_index: [2, num_conn]
        return: x_updated: [B, N, out_channels], constrain_loss: 标量
        """
        if x.dim() != 3:
            raise ValueError(f"HypergraphConv forward expects x with shape [B, N, C], got {x.shape}")

        batch_size, num_nodes, _ = x.shape

        # 1) 先对节点特征做线性投影
        x_proj = torch.matmul(x, self.weight)  # [B, N, out_channels]

        # 2) 计算节点度和超边度
        node_index = hyperedge_index[0]  # [E]
        edge_index = hyperedge_index[1]  # [E]
        num_edges = edge_index.max().item() + 1

        # D_N: node degrees. 论文的 Edge->Node 归一化使用 D_N^{-1}
        D = degree(node_index, num_nodes=num_nodes, dtype=x.dtype)  # [N]
        D_inv = 1.0 / D
        D_inv[D_inv == float("inf")] = 0
        edge_deg = degree(edge_index, num_nodes=num_edges, dtype=x.dtype)  # [M]
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
        )
        x_updated = self.edge2node(
            hyperedge_index=hyperedge_index,
            x_edge=x_edge,
            norm_edge=D_inv,        # 节点侧的归一化（D_N^{-1}）
            alpha=alpha,
        )

        constrain_losstotal = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return x_updated, constrain_losstotal

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class MaskedAdaptiveHypergraphGenerator(nn.Module):
    """基于掩码的自适应超图生成器，支持动态结构学习"""
    def __init__(self, modality, configs):
        super().__init__()
        self.modality = modality

        # 根据模态确定序列长度
        self.seq_len = getattr(configs, f'seq_len_{modality}', getattr(configs, 'seq_len', 100))

        self.dim = configs.d_model
        self.hyper_num = getattr(configs, f'hyper_num_{modality}', 50)  # 模态特定的超边数
        self.alpha = 1.0
        self.k = getattr(configs, 'k', 3)

        # 模态特定阈值，避免超图过密导致CUDA错误
        self.threshold = {'text': 0.001, 'audio': 0.1, 'video': 0.001}.get(self.modality, 0.1)

        # 仅启用动态超图结构学习
        self.dynamic = True

        # 模态特定的可学习嵌入参数（用于动态结构学习）
        self.node_embeds = nn.Parameter(torch.randn(self.seq_len, self.dim))  # 可学习节点嵌入
        self.hyper_embeds = nn.Parameter(torch.randn(self.hyper_num, self.dim))  # 可学习超边嵌入

        self.dropout = nn.Dropout(p=0.1)

        # 缓存上次的超图（用于减少重计算）
        self.cached_hypergraph = None

    def forward(self, features, mask, update_hyper=True):
        """生成动态超图结构，基于可学习嵌入和反馈驱动适应
        features: [batch_size, seq_len, feature_dim]
        mask: [batch_size, seq_len]
        update_hyper: 是否更新超图结构（减少计算开销）
        注意：实际序列长度 seq_len 可能与配置的 self.seq_len 不一致，这里统一采用
        effective_len = min(self.seq_len, seq_len)，以避免索引越界。
        """
        if not update_hyper and self.cached_hypergraph is not None:
            # 使用缓存的超图
            return [self.cached_hypergraph] * len(features) if isinstance(features, list) else [self.cached_hypergraph] * features.shape[0]
        batch_size, seq_len, feature_dim = features.shape

        # 有效长度：不超过真实序列长度，避免后续超图节点索引越界
        effective_len = min(self.seq_len, seq_len)

        # 使用可学习嵌入（仅前 effective_len 个位置参与构图）
        node_embeddings = self.node_embeds[:effective_len]  # [effective_len, dim]
        hyperedge_embeddings = self.hyper_embeds  # [hyper_num, dim]

        # 计算基础相似度（动态重计算）
        similarity = torch.mm(node_embeddings, hyperedge_embeddings.transpose(0, 1))  # [effective_len, hyper_num]
        original_similarity = similarity.clone()

        # 应用温度系数
        similarity = F.relu(self.alpha * similarity)

        # 基于掩码调整相似度（仅使用前 effective_len 个时间步）
        avg_mask = mask.mean(dim=0)  # [seq_len]
        mask_penalty = avg_mask[:effective_len]  # [effective_len]

        similarity = similarity * mask_penalty.unsqueeze(-1)  # [effective_len, hyper_num]
        masked_similarity = similarity.clone()
        mask_penalty_values = mask_penalty.clone()

        # 应用SoftMax得到软关联矩阵（用于梯度传播）
        soft_adj = F.softmax(similarity, dim=1)  # [effective_len, hyper_num]
        softmax_similarity = soft_adj.clone()

        # 选择top-k超边（稀疏化）
        mask_adj = torch.zeros_like(soft_adj)
        s1, t1 = soft_adj.topk(min(self.k, self.hyper_num), 1)
        mask_adj.scatter_(1, t1, s1.fill_(1))
        adj = soft_adj * mask_adj

        # 二值化（用于消息传递）
        adj = torch.where(
            adj > self.threshold,
            torch.tensor(1.0, device=features.device),
            torch.tensor(0.0, device=features.device),
        )
        binary_similarity = adj.clone()

        # 只保留有连接的超边
        valid_hyperedges = (adj != 0).any(dim=0)
        if valid_hyperedges.any():
            adj = adj[:, valid_hyperedges]

            # 构建超图：节点列表和超边列表
            matrix_array = adj.cpu().int()
            result_list = [
                torch.nonzero(matrix_array[:, col]).flatten().tolist()
                for col in range(matrix_array.shape[1])
            ]

            if result_list and any(len(sublist) > 0 for sublist in result_list):
                node_list = torch.cat(
                    [torch.tensor(sublist) for sublist in result_list if len(sublist) > 0]
                ).tolist()
                count_list = [
                    len(torch.nonzero(matrix_array[:, col]).flatten())
                    for col in range(matrix_array.shape[1])
                ]
                hyperedge_list = torch.cat(
                    [torch.full((count,), idx) for idx, count in enumerate(count_list)]
                ).tolist()
            else:
                node_list = [0]
                hyperedge_list = [0]
        else:
            node_list = [0]
            hyperedge_list = [0]

        # 构建共享超图
        hypergraph = np.vstack((node_list, hyperedge_list))
        hypergraph = torch.tensor(hypergraph, dtype=torch.long)


        # 缓存超图
        self.cached_hypergraph = hypergraph

        # 返回相同的超图给所有批次样本
        return [hypergraph] * batch_size


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
        output_dim = (self.num_ordinal_levels - 1) if (self.task_mode == 'ordinal') else getattr(configs, 'num_classes', 10)
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

        # 检查是否需要更新超图结构
        self.step_counter += 1
        update_hyper = (self.step_counter % self.hyper_update_freq == 0) or (self.step_counter == 1)

        for i, (modality, features, mask) in enumerate([
            ('text', text_features, text_mask),
            ('audio', audio_features, audio_mask),
            ('video', video_features, video_mask),
        ]):
            modal_seq_lens.append(features.shape[1])
            # 生成模态内超图（动态模式下按频率更新）
            hypergraphs = self.hyper_generators[i](features, mask, update_hyper=update_hyper)
            hypergraph = hypergraphs[0].to(features.device)
            modal_hypergraphs.append(hypergraph)

            # 使用超图卷积的 forward 完整走一遍 Node -> Edge -> Node，并获得约束损失
            updated_nodes_pre, _ = self.hyper_convs[i](
                features,
                hypergraph
            )

            # 使用更新后的节点特征再次做 Node -> Edge，用于跨模态超边注意力
            hyper_repr = self.hyper_convs[i].node2edge(
                hyperedge_index=hypergraph,
                x_node=updated_nodes_pre,
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

            # Edge -> Node传播：利用跨模态增强后的超边特征更新节点
            updated_node_features = self.hyper_convs[i].edge2node(
                hyperedge_index=hypergraph.to(enhanced_hyper_repr.device),
                x_edge=enhanced_hyper_repr  # [B, num_edges, d_model]
            )  # 输出: [B, seq_len_modality, d_model]

            # 确保维度匹配（如果seq_len_modality != 超图节点数，可能需要调整）
            if updated_node_features.size(1) != seq_len_modality:
                # 如果不匹配，使用插值或平均扩展
                updated_node_features = updated_node_features.mean(dim=1, keepdim=True).expand(-1, seq_len_modality, -1)

            modal_outputs.append(updated_node_features)

        # 按照HyperGAMER论文的Representation Fusion
        modal_summaries = []
        for i, (hyper_repr, node_output) in enumerate(zip(enhanced_hyper_reprs, modal_outputs)):
            # 节点池化
            z_n = node_output.mean(dim=1)  # [batch_size, d_model]
            # 超边池化
            z_e = hyper_repr.mean(dim=1)   # [batch_size, d_model]
            # 模态表示拼接
            Z_m = torch.cat([z_n, z_e], dim=1)  # [batch_size, 2 * d_model]
            modal_summaries.append(Z_m)
        # 多模态融合
        Z_fusion = torch.cat(modal_summaries, dim=1)  # [batch_size, 6 * d_model]
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
        modal_hypergraphs: 各模态超图结构
        modal_hyper_reprs: 各模态超边表示 [batch_size, num_edges_m, d_model]
        modal_node_outputs: 各模态节点输出 [batch_size, seq_len_m, d_model]
        """
        total_ecd = 0.0
        total_epc = 0.0
        num_modalities = len(modal_hypergraphs)

        for m in range(num_modalities):
            hypergraph = modal_hypergraphs[m]  # [2, num_conn]
            hyper_repr = modal_hyper_reprs[m]  # [B, M, D]
            node_repr = modal_node_outputs[m]  # [B, N, D]

            batch_size, num_nodes, _ = node_repr.shape
            _, num_edges, _ = hyper_repr.shape

            # 构建二值关联矩阵 H [B, N, M] - 扩展到batch维度
            H = torch.zeros(batch_size, num_nodes, num_edges, device=hyper_repr.device)
            node_idx = hypergraph[0]
            edge_idx = hypergraph[1]
            H[:, node_idx, edge_idx] = 1.0  # [B, N, M]

            # ECD: 节点-超边距离 - 向量化
            # 计算所有节点到所有超边的距离 [B, N, M]
            dist_node_hyper = torch.cdist(node_repr, hyper_repr, p=2)  # [B, N, M]

            # 只计算关联的距离，mask掉非关联
            masked_dist = dist_node_hyper * H  # [B, N, M]
            # 对每个节点，平均其关联超边的距离
            node_degrees = H.sum(dim=2, keepdim=True)  # [B, N, 1]
            node_degrees = torch.clamp(node_degrees, min=1e-8)  # 避免除零
            ecd_per_node = (masked_dist.sum(dim=2) / node_degrees.squeeze(2))  # [B, N]
            ecd_m = ecd_per_node.mean()  # 标量
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

                # 计算EPC项：α * dist + (1-α) * clamp(η - dist, 0)
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
        ecr_loss = lambda_ecd * (total_ecd / num_modalities) + lambda_epc * (total_epc / num_modalities)

        return ecr_loss

    def __repr__(self):
        return f"{self.__class__.__name__}({self.configs.d_model}, {getattr(self.configs, 'num_classes', 10)})"
