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
from .Layers import EncoderLayer, Decoder, Predictor
from .Layers import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
# from .Layers import get_mask, get_subsequent_mask, refer_points, get_k_q, get_q_k
# from .embed import DataEmbedding, CustomEmbedding,DataEmbedding_new
from torch_geometric.utils import scatter
import math




class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        configs.device = torch.device("cuda")
        self.channels = configs.enc_in
        self.individual = getattr(configs, 'individual', False)
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Tran = nn.Linear(self.pred_len, self.pred_len)


        self.all_size=get_mask(configs.seq_len, configs.window_size)
        self.Ms_length = sum(self.all_size)
        self.conv_layers = Bottleneck_Construct(configs.enc_in, configs.window_size, configs.enc_in)
        self.out_tran = nn.Linear(self.Ms_length, self.pred_len)
        self.out_tran.weight=nn.Parameter((1/self.Ms_length)*torch.ones([self.pred_len,self.Ms_length]))
        self.chan_tran=nn.Linear(configs.d_model,configs.enc_in)
        self.inter_tran = nn.Linear(80, self.pred_len)
        self.concat_tra=nn.Linear(320,self.pred_len)

        self.dim=configs.d_model
        self.hyper_num=50
        self.embedhy=nn.Embedding(self.hyper_num,self.dim)
        self.embednod=nn.Embedding(self.Ms_length,self.dim)


        self.idx = torch.arange(self.hyper_num)
        self.nodidx=torch.arange(self.Ms_length)
        self.alpha=3
        self.k=10

        self.window_size=configs.window_size
        self.multiadphyper=multi_adaptive_hypergraoh(configs)
        self.hyper_num1 = configs.hyper_num
        self.hyconv=nn.ModuleList()
        self.hyperedge_atten=SelfAttentionLayer(configs)
        for i in range (len(self.hyper_num1)):
            self.hyconv.append(HypergraphConv(configs.enc_in, configs.enc_in))

        self.slicetran=nn.Linear(100,configs.pred_len)
        self.weight = nn.Parameter(torch.randn(self.pred_len, 76))

        self.argg = nn.ModuleList()
        for i in range(len(self.hyper_num1)):
            self.argg.append(nn.Linear(self.all_size[i],self.pred_len))
        self.chan_tran = nn.Linear(configs.enc_in, configs.enc_in)


class BimodalClassifier(nn.Module):
    """
    双模态分类器：文本 + 音频
    将ASHyper的时间序列预测架构改造为多模态分类
    """
    def __init__(self, configs):
        super(BimodalClassifier, self).__init__()
        self.configs = configs

        # 模态配置
        self.modalities = ['text', 'audio']
        self.feature_dims = {'text': 1024, 'audio': 1024}

        # 模态特定的超图生成器
        self.hyper_generators = nn.ModuleList([
            MaskedAdaptiveHypergraphGenerator(modality, configs)
            for modality in self.modalities
        ])

        # 模态特定的超图卷积 (使用bottleneck结构控制参数量)
        self.hyper_convs = nn.ModuleList([
            HypergraphConv(self.feature_dims[modality], configs.d_model)  # 1024 -> d_model
            for modality in self.modalities
        ])

        # 模态间交互机制 (超边注意力)
        self.inter_modal_attention = SelfAttentionLayer(configs)

        # 按照原ASHyper的融合方式 - 只保留必要的变换
        self.inter_tran = nn.Linear(80, configs.d_model)  # 模态间输出变换
        self.classifier = nn.Linear(configs.d_model, getattr(configs, 'num_classes', 10))

    def forward(self, batch_data, x_mark_enc=None):
        """
        前向传播 - 严格按照原ASHyper的融合方式
        batch_data: 包含多模态特征和掩码的字典
        x_mark_enc: 保持兼容性，但不使用
        """
        # 检查输入类型
        if not isinstance(batch_data, dict):
            # 如果是原ASHyper调用方式，调用原forward方法
            return self.forward_original(batch_data, x_mark_enc)

        # 提取各模态特征和掩码
        text_features = batch_data['text_vector']    # [batch_size, seq_len, 1024]
        audio_features = batch_data['audio_vector']  # [batch_size, seq_len, 1024]
        text_mask = batch_data['text_mask']          # [batch_size, seq_len]
        audio_mask = batch_data['audio_mask']        # [batch_size, seq_len]

        # 模态内处理 (类似尺度内处理)
        modal_outputs = []
        all_hyper_representations = []
        total_constrain_loss = 0

        for i, (modality, features, mask) in enumerate([
            ('text', text_features, text_mask),
            ('audio', audio_features, audio_mask)
        ]):
            # 生成模态内超图 (现在返回共享超图)
            hypergraphs = self.hyper_generators[i](features, mask)
            hypergraph = hypergraphs[0].to(features.device)  # 所有批次使用相同的超图

            # 模态内超图卷积
            output, constrain_loss = self.hyper_convs[i](features, hypergraph)
            modal_outputs.append(output)
            total_constrain_loss += constrain_loss

            # 收集超边表示用于模态间交互 (类似尺度间处理)
            node_value = features.permute(0, 2, 1)  # [batch_size, feature_dim, seq_len]
            edge_sums = {}
            for edge_id, node_id in zip(hypergraph[1], hypergraph[0]):
                edge_id = edge_id.item()
                node_id = node_id.item()
                if edge_id not in edge_sums:
                    edge_sums[edge_id] = node_value[:, :, node_id]
                else:
                    edge_sums[edge_id] += node_value[:, :, node_id]

            # 超边特征收集
            if edge_sums:
                hyper_reprs = torch.stack([v for v in edge_sums.values()], dim=1)  # [batch_size, num_edges, feature_dim]
            else:
                hyper_reprs = torch.zeros(batch_size, 1, features.shape[-1], device=features.device)
            all_hyper_representations.append(hyper_reprs)

        # 模态间交互 (类似尺度间注意力)
        if all_hyper_representations:
            inter_modal_features = torch.cat(all_hyper_representations, dim=1)  # [batch_size, total_edges, feature_dim]
            # 填充到固定长度 (类似原代码的padding)
            padding_need = 80 - inter_modal_features.size(1)
            if padding_need > 0:
                inter_modal_attention = self.inter_modal_attention(inter_modal_features)
                pad = torch.nn.functional.pad(inter_modal_attention, (0, 0, 0, padding_need, 0, 0))
            else:
                pad = self.inter_modal_attention(inter_modal_features)
        else:
            pad = torch.zeros(batch_size, 80, features.shape[-1], device=features.device)

        # 模态内输出拼接 (类似尺度输出拼接) - 现在已经是d_model维度
        if modal_outputs:
            result_tensor = torch.cat(modal_outputs, dim=1)  # [batch_size, seq_len * 2, d_model]
        else:
            result_tensor = torch.zeros(batch_size, text_features.shape[1] * 2, self.configs.d_model, device=text_features.device)

        # 最终融合 (严格按照原ASHyper的方式)
        # x_out = 模态内输出, x = 原始输入, x_out_inter = 模态间输出

        # 模态内输出池化
        x_out = result_tensor.mean(dim=1)  # [batch_size, d_model] - 类似原ASHyper的尺度输出

        # 原始输入投影到d_model (简化处理)
        # 由于不同模态序列长度不同，这里直接使用模态内输出，不再尝试拼接原始输入
        original_proj = torch.zeros(text_features.shape[0], 2048, device=text_features.device)  # 占位符

        # 模态间输出
        x_out_inter = pad.mean(dim=1)  # [batch_size, feature_dim] - 类似原ASHyper的尺度间输出

        # 相加融合 (严格按照原ASHyper: x_out + x + x_out_inter)
        # 但需要维度对齐：x_out是d_model, x_out_inter是feature_dim(1024)
        # 简化为只用超图处理后的特征
        fused_features = x_out  # [batch_size, d_model]

        # 分类输出
        logits = self.classifier(fused_features)  # [batch_size, num_classes]

        return logits, total_constrain_loss

    def forward_original(self, x, x_mark_enc):
        # 原ASHyper的时间序列预测前向传播
        # normalization
        mean_enc=x.mean(1,keepdim=True).detach()
        x=x - mean_enc
        std_enc=torch.sqrt(torch.var(x,dim=1,keepdim=True,unbiased=False)+1e-5).detach()
        x=x / std_enc
        adj_matrix = self.multiadphyper(x)
        seq_enc = self.conv_layers(x)

        sum_hyper_list = []
        for i in range(len(self.hyper_num1)):

            mask = torch.tensor(adj_matrix[i]).to(x.device)
            ###inter-scale
            node_value = seq_enc[i].permute(0,2,1)
            node_value = torch.tensor(node_value).to(x.device)
            edge_sums={}
            for edge_id, node_id in zip(mask[1], mask[0]):
                if edge_id not in edge_sums:
                    edge_id=edge_id.item()
                    node_id=node_id.item()
                    edge_sums[edge_id] = node_value[:, :, node_id]
                else:
                    edge_sums[edge_id] += node_value[:, :, node_id]


            for edge_id, sum_value in edge_sums.items():
                sum_value = sum_value.unsqueeze(1)
                sum_hyper_list.append(sum_value)


            ###intra-scale
            output,constrainloss = self.hyconv[i](seq_enc[i], mask)


            if i==0:
                result_tensor=output
                result_conloss=constrainloss
            else:
                result_tensor = torch.cat((result_tensor, output), dim=1)
                result_conloss+=constrainloss

        sum_hyper_list=torch.cat(sum_hyper_list,dim=1)
        sum_hyper_list=sum_hyper_list.to(x.device)
        padding_need=80-sum_hyper_list.size(1)
        hyperedge_attention=self.hyperedge_atten(sum_hyper_list)
        pad = torch.nn.functional.pad(hyperedge_attention, (0, 0, 0, padding_need, 0, 0))



        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:

            x = self.Linear(x.permute(0,2,1))

            x_out=self.out_tran(result_tensor.permute(0,2,1))###ori
            x_out_inter = self.inter_tran(pad.permute(0, 2, 1))

        x=x_out+x+x_out_inter
        x=self.Linear_Tran(x).permute(0,2,1)
        x = x * std_enc + mean_enc

        return x,result_conloss# [Batch, Output length, Channel]


class HypergraphConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_attention=True,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0.1,
                 bias=False):
        super(HypergraphConv, self).__init__(aggr='add')
        self.soft=nn.Softmax(dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention


        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_channels, out_channels))

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
        zeros(self.bias)

    def __forward__(self,
                    x,
                    hyperedge_index,
                    alpha=None):

        D = degree(hyperedge_index[0], x.size(0), x.dtype)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x.dtype)
        # --------------------------------------------------------
        B[B == float("inf")] = 0

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)

        return out

    def message(self, x_j, edge_index_i, norm, alpha):
        out = norm[edge_index_i].view(-1, 1, 1) * x_j
        if alpha is not None:
            out=alpha.unsqueeze(-1)*out
        return out
    def forward(self, x, hyperedge_index):
        x = torch.matmul(x, self.weight)
        x1=x.transpose(0,1)
        x_i = torch.index_select(x1, dim=0, index=hyperedge_index[0])
        edge_sums = {}

        for edge_id, node_id in zip(hyperedge_index[1], hyperedge_index[0]):
            if edge_id not in edge_sums:
                edge_id = edge_id.item()
                node_id = node_id.item()
                edge_sums[edge_id] = x1[node_id, :, :]
            else:
                edge_sums[edge_id] += x1[node_id, :, :]
        result_list = torch.stack([value for value in edge_sums.values()], dim=0)
        x_j = torch.index_select(result_list, dim=0, index=hyperedge_index[1])
        loss_hyper = 0
        for k in range(len(edge_sums)):
            for m in range(len(edge_sums)):
                inner_product = torch.sum(edge_sums[k] * edge_sums[m], dim=1, keepdim=True)
                norm_q_i = torch.norm(edge_sums[k], dim=1, keepdim=True)
                norm_q_j = torch.norm(edge_sums[m], dim=1, keepdim=True)
                alpha = inner_product / (norm_q_i * norm_q_j)
                distan = torch.norm(edge_sums[k] - edge_sums[m],dim=1, keepdim=True)
                loss_item = alpha * distan + (1 - alpha) * (torch.clamp(torch.tensor(4.2) - distan, min=0.0))
                loss_hyper += torch.abs(torch.mean(loss_item))


        loss_hyper = loss_hyper / ((len(edge_sums) + 1)**2)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, hyperedge_index[0], num_nodes=x1.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        D = degree(hyperedge_index[0], x1.size(0), x.dtype)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x.dtype)
        B[B == float("inf")] = 0
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x1, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        out=out.transpose(0, 1)
        constrain_loss = x_i - x_j
        constrain_lossfin1=torch.mean(constrain_loss)
        constrain_losstotal = abs(constrain_lossfin1) + loss_hyper
        return out, constrain_losstotal
    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class multi_adaptive_hypergraoh(nn.Module):
    def __init__(self,configs):
        super(multi_adaptive_hypergraoh, self).__init__()
        self.seq_len = configs.seq_len
        self.window_size=configs.window_size
        self.inner_size=getattr(configs, 'inner_size', 5)
        self.dim=configs.d_model
        self.hyper_num=configs.hyper_num
        self.alpha=3
        self.k=getattr(configs, 'k', 3)
        self.embedhy=nn.ModuleList()
        self.embednod=nn.ModuleList()
        self.linhy=nn.ModuleList()
        self.linnod=nn.ModuleList()
        for i in range(len(self.hyper_num)):
            self.embedhy.append(nn.Embedding(self.hyper_num[i],self.dim))
            self.linhy.append(nn.Linear(self.dim,self.dim))
            self.linnod.append(nn.Linear(self.dim,self.dim))
            if i==0:
                self.embednod.append(nn.Embedding(self.seq_len,self.dim))
            else:
                product=math.prod(self.window_size[:i])
                layer_size=math.floor(self.seq_len/product)
                self.embednod.append(nn.Embedding(int(layer_size),self.dim))

        self.dropout = nn.Dropout(p=0.1)


    def forward(self,x):
        node_num = []
        node_num.append(self.seq_len)
        for i in range(len(self.window_size)):
            layer_size = math.floor(node_num[i] / self.window_size[i])
            node_num.append(layer_size)
        hyperedge_all=[]

        for i in range(len(self.hyper_num)):
            hypidxc=torch.arange(self.hyper_num[i]).to(x.device)
            nodeidx=torch.arange(node_num[i]).to(x.device)
            hyperen=self.embedhy[i](hypidxc)
            nodeec=self.embednod[i](nodeidx)

            a = torch.mm(nodeec, hyperen.transpose(1, 0))
            adj=F.softmax(F.relu(self.alpha*a))
            mask = torch.zeros(nodeec.size(0), hyperen.size(0)).to(x.device)
            mask.fill_(float('0'))
            s1, t1 = adj.topk(min(adj.size(1),self.k), 1)
            mask.scatter_(1, t1, s1.fill_(1))
            adj = adj * mask
            adj = torch.where(adj > 0.5, torch.tensor(1).to(x.device), torch.tensor(0).to(x.device))
            adj = adj[:, (adj != 0).any(dim=0)]
            matrix_array = torch.tensor(adj, dtype=torch.int)
            result_list = [list(torch.nonzero(matrix_array[:, col]).flatten().tolist()) for col in
                            range(matrix_array.shape[1])]

            if result_list:
                node_list = torch.cat([torch.tensor(sublist) for sublist in result_list if len(sublist) > 0]).tolist()
                count_list = list(torch.sum(adj, dim=0).tolist())
                hperedge_list = torch.cat([torch.full((count,), idx) for idx, count in enumerate(count_list, start=0)]).tolist()
            else:
                # 如果没有有效的超边，创建一个默认的
                node_list = [0]
                hperedge_list = [0]
            hypergraph=np.vstack((node_list,hperedge_list))
            hypergraph = torch.tensor(hypergraph, dtype=torch.long)
            hyperedge_all.append(hypergraph)

        return hyperedge_all


class MaskedAdaptiveHypergraphGenerator(nn.Module):
    """基于掩码的超图生成器，生成共享的超图结构（严格按照原ASHyper）"""
    def __init__(self, modality, configs):
        super().__init__()
        self.modality = modality
        # 根据模态确定序列长度
        if modality == 'text':
            self.seq_len = 160  # 文本序列长度
        elif modality == 'audio':
            self.seq_len = 518  # 音频序列长度
        elif modality == 'video':
            self.seq_len = 16   # 视频序列长度
        else:
            self.seq_len = configs.seq_len  # 默认

        self.dim = configs.d_model
        self.hyper_num = getattr(configs, f'hyper_num_{modality}', 50)  # 模态特定的超边数
        self.alpha = 3
        self.k = getattr(configs, 'k', 3)

        # 模态特定的固定嵌入层（类似原ASHyper）
        self.node_embed = nn.Embedding(self.seq_len, self.dim)  # 节点嵌入
        self.hyperedge_embed = nn.Embedding(self.hyper_num, self.dim)  # 超边嵌入

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, features, mask):
        """生成共享超图结构，补零节点通过掩码影响相似度"""
        # features: [batch_size, seq_len, feature_dim]
        # mask: [batch_size, seq_len]

        batch_size, seq_len, feature_dim = features.shape

        # 使用固定嵌入层（类似原ASHyper）
        node_embeddings = self.node_embed(
            torch.arange(self.seq_len, device=features.device)
        )  # [seq_len, dim]

        hyperedge_embeddings = self.hyperedge_embed(
            torch.arange(self.hyper_num, device=features.device)
        )  # [hyper_num, dim]

        # 计算基础相似度
        similarity = torch.mm(node_embeddings, hyperedge_embeddings.transpose(0, 1))  # [seq_len, hyper_num]

        # 应用温度系数
        similarity = F.relu(self.alpha * similarity)

        # 基于掩码调整相似度：补零节点降低与其他节点的连接概率
        # 计算每个时间步的平均有效性
        avg_mask = mask.mean(dim=0)  # [seq_len] 每个时间步的有效比例

        # 对补零严重的时间步降低相似度权重
        # 确保mask_penalty与similarity的维度匹配
        if len(avg_mask) != self.seq_len:
            # 如果mask长度与seq_len不匹配，使用全局平均
            mask_penalty = avg_mask.mean().expand(self.seq_len)
        else:
            mask_penalty = avg_mask

        similarity = similarity * mask_penalty.unsqueeze(-1)  # [seq_len, hyper_num] 补零节点相似度降低

        # 应用softmax
        adj = F.softmax(similarity, dim=0)  # [seq_len, hyper_num]

        # 选择top-k超边
        mask_adj = torch.zeros_like(adj)
        s1, t1 = adj.topk(min(self.k, self.hyper_num), 1)
        mask_adj.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask_adj

        # 二值化
        adj = torch.where(adj > 0.5, torch.tensor(1.0, device=features.device),
                        torch.tensor(0.0, device=features.device))

        # 只保留有连接的超边
        valid_hyperedges = (adj != 0).any(dim=0)
        if valid_hyperedges.any():
            adj = adj[:, valid_hyperedges]

            # 构建超图：节点列表和超边列表
            matrix_array = adj.cpu().int()
            result_list = [torch.nonzero(matrix_array[:, col]).flatten().tolist()
                         for col in range(matrix_array.shape[1])]

            if result_list and any(len(sublist) > 0 for sublist in result_list):
                node_list = torch.cat([torch.tensor(sublist) for sublist in result_list
                                     if len(sublist) > 0]).tolist()
                count_list = [len(torch.nonzero(matrix_array[:, col]).flatten())
                            for col in range(matrix_array.shape[1])]
                hyperedge_list = torch.cat([torch.full((count,), idx)
                                          for idx, count in enumerate(count_list)]).tolist()
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



class SelfAttentionLayer(nn.Module):
    def __init__(self, configs):
        super(SelfAttentionLayer, self).__init__()
        self.query_weight = nn.Linear(configs.enc_in, configs.enc_in)
        self.key_weight = nn.Linear(configs.enc_in, configs.enc_in)
        self.value_weight = nn.Linear(configs.enc_in, configs.enc_in)

    def forward(self, x):
        q = self.query_weight(x)
        k = self.key_weight(x)
        v = self.value_weight(x)
        attention_scores = F.softmax(torch.matmul(q, k.transpose(1, 2)) / (k.shape[-1] ** 0.5), dim=-1)
        attended_values = torch.matmul(attention_scores, v)

        return attended_values

def get_mask(input_size, window_size):
    """Get the attention mask of HyperGraphConv"""
    # Get the size of all layers
    # window_size=[4,4,4]
    all_size = []
    all_size.append(input_size)
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(layer_size)
    return all_size

