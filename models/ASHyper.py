import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


class InterModalAttention(nn.Module):
    """模态间注意力机制"""
    def __init__(self, d_model):
        super().__init__()
        self.query_weight = nn.Linear(d_model, d_model)
        self.key_weight = nn.Linear(d_model, d_model)
        self.value_weight = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.query_weight(x)
        k = self.key_weight(x)
        v = self.value_weight(x)
        attention_scores = F.softmax(torch.matmul(q, k.transpose(1, 2)) / (k.shape[-1] ** 0.5), dim=-1)
        attended_values = torch.matmul(attention_scores, v)
        return attended_values


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

    def node2edge(self, hyperedge_index, x_node, norm_node=None, alpha=None, project=True):
        """
        节点到超边的传播：提取超边特征
        hyperedge_index: [2, num_conn], row=node_id, col=edge_id
        x_node: [B, N, d] 节点特征
        norm_node: 节点侧的归一化系数
        alpha: [num_conn] or [B, num_conn, 1] 节点-超边注意力
        project: 是否在该函数内部对节点特征做线性投影
        return: x_edge: [B, M, d] 超边特征
        """
        # 应用权重变换（默认进行投影，可在forward中复用已投影特征）
        if project:
            x_node = torch.matmul(x_node, self.weight)  # [B, N, out_channels]
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
        x1 = x_proj.transpose(0, 1)  # [N, B, out_channels]

        # 2) 计算节点度和超边度
        node_index = hyperedge_index[0]  # [E]
        edge_index = hyperedge_index[1]  # [E]
        num_edges = edge_index.max().item() + 1

        D = degree(node_index, num_nodes=num_nodes, dtype=x.dtype)  # [N]
        edge_deg = degree(edge_index, num_nodes=num_edges, dtype=x.dtype)  # [M]
        B_norm = 1.0 / edge_deg
        B_norm[B_norm == float("inf")] = 0

        # 3) 计算每条关联 (node, hyperedge) 对的节点/超边表示
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

        # 4) 计算约束损失（超边间的几何约束 + 节点-超边差异）
        # 将超边特征展平到一个向量空间后计算两两相似度和距离
        edge_flat = edge_sums.reshape(num_edges, -1)  # [M, B * out_channels]
        edge_flat_norm = F.normalize(edge_flat, p=2, dim=1)
        cos_sim = torch.mm(edge_flat_norm, edge_flat_norm.t())  # [M, M]

        diff = edge_flat.unsqueeze(1) - edge_flat.unsqueeze(0)  # [M, M, D']
        dist = diff.norm(dim=-1)  # [M, M]

        alpha_cm = cos_sim
        margin = 4.2
        loss_item = alpha_cm * dist + (1 - alpha_cm) * torch.clamp(margin - dist, min=0.0)
        loss_hyper = torch.abs(loss_item.mean())
        loss_hyper = loss_hyper / ((num_edges + 1) ** 2)

        constrain = x_i - x_j  # [E, B, out_channels]
        constrain_lossfin1 = torch.mean(constrain)
        constrain_losstotal = torch.abs(constrain_lossfin1) + loss_hyper

        # 5) 计算节点-超边注意力权重 alpha
        alpha = None
        if self.use_attention:
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

                # 对每个 batch 独立做 softmax，按节点维度规范化
                alphas = []
                for b in range(batch_size):
                    alpha_b = softmax(e[:, b], node_index, num_nodes=num_nodes)  # [E]
                    alphas.append(alpha_b)
                alpha = torch.stack(alphas, dim=1)  # [E, B]
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

                # 对每个 batch、每个 head 独立做 softmax，按节点维度规范化
                alphas = []
                for b in range(batch_size):
                    alpha_b_heads = []
                    for h in range(self.heads):
                        # e[:, b, h]: [E]
                        alpha_bh = softmax(e[:, b, h], node_index, num_nodes=num_nodes)  # [E]
                        alpha_b_heads.append(alpha_bh)
                    # [E, heads]
                    alpha_b = torch.stack(alpha_b_heads, dim=1)
                    alphas.append(alpha_b)

                # [E, B, heads]
                alpha = torch.stack(alphas, dim=1)

                # 将多头注意力聚合为单一权重（这里取平均，后续可做 ablation 改为求和等）
                alpha = alpha.mean(dim=-1)  # [E, B]
                alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 6) 使用注意力和归一化系数做 Node -> Edge -> Node 双向传播
        x_edge = self.node2edge(
            hyperedge_index=hyperedge_index,
            x_node=x_proj,          # 已投影的节点特征
            norm_node=B_norm,      # 超边侧的归一化
            alpha=alpha,
            project=False,         # 已经投影过，这里不再重复
        )
        x_updated = self.edge2node(
            hyperedge_index=hyperedge_index,
            x_edge=x_edge,
            norm_edge=D,           # 节点侧的归一化
            alpha=alpha,
        )

        return x_updated, constrain_losstotal

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class MaskedAdaptiveHypergraphGenerator(nn.Module):
    """基于掩码的自适应超图生成器"""
    def __init__(self, modality, configs):
        super().__init__()
        self.modality = modality

        # 根据模态确定序列长度
        self.seq_len = getattr(configs, f'seq_len_{modality}', getattr(configs, 'seq_len', 100))

        self.dim = configs.d_model
        self.hyper_num = getattr(configs, f'hyper_num_{modality}', 50)  # 模态特定的超边数
        self.alpha = 3
        self.k = getattr(configs, 'k', 3)

        # 模态特定的固定嵌入层
        self.node_embed = nn.Embedding(self.seq_len, self.dim)  # 节点嵌入
        self.hyperedge_embed = nn.Embedding(self.hyper_num, self.dim)  # 超边嵌入

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, features, mask):
        """生成共享超图结构，补零节点通过掩码影响相似度
        features: [batch_size, seq_len, feature_dim]
        mask: [batch_size, seq_len]
        注意：实际序列长度 seq_len 可能与配置的 self.seq_len 不一致，这里统一采用
        effective_len = min(self.seq_len, seq_len)，以避免索引越界。
        """
        batch_size, seq_len, feature_dim = features.shape

        # 有效长度：不超过真实序列长度，避免后续超图节点索引越界
        effective_len = min(self.seq_len, seq_len)

        # 使用固定嵌入层（仅前 effective_len 个位置参与构图）
        node_embeddings = self.node_embed(
            torch.arange(effective_len, device=features.device)
        )  # [effective_len, dim]

        hyperedge_embeddings = self.hyperedge_embed(
            torch.arange(self.hyper_num, device=features.device)
        )  # [hyper_num, dim]

        # 计算基础相似度
        similarity = torch.mm(node_embeddings, hyperedge_embeddings.transpose(0, 1))  # [effective_len, hyper_num]

        # 应用温度系数
        similarity = F.relu(self.alpha * similarity)

        # 基于掩码调整相似度（仅使用前 effective_len 个时间步）
        avg_mask = mask.mean(dim=0)  # [seq_len]
        mask_penalty = avg_mask[:effective_len]  # [effective_len]

        similarity = similarity * mask_penalty.unsqueeze(-1)  # [effective_len, hyper_num]

        # 应用softmax
        adj = F.softmax(similarity, dim=0)  # [effective_len, hyper_num]

        # 选择top-k超边
        mask_adj = torch.zeros_like(adj)
        s1, t1 = adj.topk(min(self.k, self.hyper_num), 1)
        mask_adj.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask_adj

        # 二值化
        adj = torch.where(
            adj > 0.5,
            torch.tensor(1.0, device=features.device),
            torch.tensor(0.0, device=features.device),
        )

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

        # 返回相同的超图给所有批次样本
        return [hypergraph] * batch_size


class BimodalClassifier(nn.Module):
    """
    多模态分类器：文本 + 音频 + 视频
    将ASHyper的时间序列预测架构改造为多模态分类
    """
    def __init__(self, configs):
        super(BimodalClassifier, self).__init__()
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

        # 模态间交互机制 (超边注意力，使用d_model维度)
        self.inter_modal_attention = InterModalAttention(configs.d_model)

        # 按照原ASHyper的融合方式 - 只保留必要的变换 (动态total_edges)
        self.classifier = nn.Linear(configs.d_model, getattr(configs, 'num_classes', 10))

    def forward(self, batch_data, x_mark_enc=None):
        """
        前向传播 - 严格按照原ASHyper的融合方式
        batch_data: 包含多模态特征和掩码的字典
        x_mark_enc: 保持兼容性，但不使用
        """
        # 检查输入类型
        if not isinstance(batch_data, dict):
            raise ValueError("BimodalClassifier only supports multimodal input as dict")

        # 提取各模态特征和掩码
        text_features = batch_data['text_vector']     # [batch_size, 160, 1024]
        audio_features = batch_data['audio_vector']   # [batch_size, 518, 1024]
        video_features = batch_data['video_vector']   # [batch_size, 16, 2048]
        text_mask = batch_data['text_mask']           # [batch_size, 160]
        audio_mask = batch_data['audio_mask']         # [batch_size, 518]
        video_mask = batch_data['video_mask']         # [batch_size, 16]

        # 模态内处理 - 第一阶段：Node -> Edge (提取超边特征)
        modal_hyper_reprs = []  # 各模态的超边特征
        modal_hypergraphs = []  # 各模态的超图结构
        # 使用标量 Tensor 以便与约束损失累加，并保持梯度
        total_constrain_loss = torch.tensor(0.0, device=text_features.device)

        for i, (modality, features, mask) in enumerate([
            ('text', text_features, text_mask),
            ('audio', audio_features, audio_mask),
            ('video', video_features, video_mask),
        ]):
            # 生成模态内超图
            hypergraphs = self.hyper_generators[i](features, mask)
            hypergraph = hypergraphs[0].to(features.device)
            modal_hypergraphs.append(hypergraph)

            # 使用超图卷积的 forward 完整走一遍 Node -> Edge -> Node，并获得约束损失
            updated_nodes_pre, constrain_loss_mod = self.hyper_convs[i](
                features,
                hypergraph
            )
            total_constrain_loss = total_constrain_loss + constrain_loss_mod

            # 使用更新后的节点特征再次做 Node -> Edge，用于跨模态超边注意力
            hyper_repr = self.hyper_convs[i].node2edge(
                hyperedge_index=hypergraph,
                x_node=updated_nodes_pre,
                project=False,
            )
            modal_hyper_reprs.append(hyper_repr)

            # 记录超图结构和权重矩阵信息（可选，用于调试）
            # 注释掉推理时的记录，避免干扰训练记录
            # self.hyper_generators[i].record_hypergraph(
            #     hypergraph, stage='inference', epoch=0, batch=0,
            #     weight_matrix=self.hyper_convs[i].weight
            # )

        # 模态间交互 (跨模态超边注意力) - 动态total_edges，无填充
        if modal_hyper_reprs:
            inter_modal_features = torch.cat(modal_hyper_reprs, dim=1)  # [batch_size, total_edges, d_model]
            inter_modal_attention = self.inter_modal_attention(inter_modal_features)  # [batch_size, total_edges, d_model]
        else:
            inter_modal_attention = torch.zeros(batch_size, 1, self.configs.d_model, device=features.device)

        # 切分回各模态的增强超边特征 - 基于动态total_edges
        if modal_hyper_reprs:
            edge_counts = [repr.size(1) for repr in modal_hyper_reprs]
            enhanced_hyper_reprs = []
            start_idx = 0
            for count in edge_counts:
                enhanced_repr = inter_modal_attention[:, start_idx:start_idx + count, :]
                enhanced_hyper_reprs.append(enhanced_repr)
                start_idx += count
        else:
            enhanced_hyper_reprs = [torch.zeros(batch_size, 1, self.configs.d_model, device=features.device) for _ in modal_hypergraphs]

        # 模态内处理 - 第二阶段：Edge -> Node (用增强超边更新节点)
        modal_outputs = []
        for i, (hypergraph, enhanced_hyper_repr) in enumerate(zip(modal_hypergraphs, enhanced_hyper_reprs)):
            # 获取模态的序列长度
            seq_len_modality = [
                text_features.shape[1],
                audio_features.shape[1],
                video_features.shape[1],
            ][i]

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

        # 模态内输出拼接 (类似尺度输出拼接) - 现在已经是d_model维度
        if modal_outputs:
            # [batch_size, (L_text + L_audio + L_video), d_model]
            result_tensor = torch.cat(modal_outputs, dim=1)
        else:
            total_len = text_features.shape[1] + audio_features.shape[1] + video_features.shape[1]
            result_tensor = torch.zeros(
                text_features.shape[0],
                total_len,
                self.configs.d_model,
                device=text_features.device,
            )

        # 最终融合 (严格按照原ASHyper的方式)
        # x_out = 模态内输出, x = 原始输入, x_out_inter = 模态间输出

        # 模态内输出池化
        x_out = result_tensor.mean(dim=1)  # [batch_size, d_model] - 类似原ASHyper的尺度输出

        # 模态间输出 - 动态total_edges
        x_out_inter = inter_modal_attention.mean(dim=1)  # [batch_size, d_model] - 类似原ASHyper的尺度间输出

        # 相加融合 (严格按照原ASHyper: x_out + x + x_out_inter)
        # 现在维度都对齐为d_model
        fused_features = x_out + x_out_inter  # [batch_size, d_model]

        # 分类输出
        logits = self.classifier(fused_features)  # [batch_size, num_classes]

        # 对多模态约束损失做简单平均，避免因模态数目放大尺度
        total_constrain_loss = total_constrain_loss / len(self.modalities)

        return logits, total_constrain_loss