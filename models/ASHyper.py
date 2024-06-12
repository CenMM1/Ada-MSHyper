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
from .embed import DataEmbedding, CustomEmbedding,DataEmbedding_new
from torch_geometric.utils import scatter

import math



"""
ASHyper不加特征维度转换版本，原始多少维就是多少维，加loss约束
"""
class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        configs.device = torch.device("cuda")
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Tran = nn.Linear(self.pred_len, self.pred_len)

        #以下为超图设计代码
        # self.enc_embedding=DataEmbedding(configs.enc_in, configs.d_model, configs.dropout)###数据embedding，在loss约束中没有加
        self.all_size=get_mask(configs.seq_len, configs.window_size)
        self.Ms_length = sum(self.all_size)
        # self.conv_layers = eval(configs.CSCM)(configs.d_model, configs.window_size, configs.d_bottleneck)
        self.conv_layers = eval(configs.CSCM)(configs.enc_in, configs.window_size, configs.enc_in)
        # self.conv1 = HypergraphConv(configs.d_model, configs.enc_in)
        # self.conv1 = HypergraphConv(512, 16)
        self.out_tran = nn.Linear(self.Ms_length, self.pred_len)
        self.out_tran.weight=nn.Parameter((1/self.Ms_length)*torch.ones([self.pred_len,self.Ms_length]))
        # self.in_tran=nn.Linear(self.seq_len,self.pred_len)
        self.chan_tran=nn.Linear(configs.d_model,configs.enc_in)
        # self.chan_tran.weight=nn.Parameter((1/configs.d_mode)*torch.ones([configs.enc_in,configs.d_mode]))
        # self.pad=torch.nn.functional.pad()
        self.inter_tran = nn.Linear(80, self.pred_len)
        self.concat_tra=nn.Linear(320,self.pred_len)

        ###以下为embedding实现
        self.dim=configs.d_model
        self.hyper_num=50
        # self.nod_num=
        self.embedhy=nn.Embedding(self.hyper_num,self.dim)
        self.embednod=nn.Embedding(self.Ms_length,self.dim)
        # self.embedhy.weight=nn.Parameter((1/self.hyper_num)*torch.ones([self.dim,self.hyper_num]))
        # self.embednod.weight=nn.Parameter((1/self.Ms_length)*torch.ones([self.dim,self.Ms_length]))

        self.idx = torch.arange(self.hyper_num)
        self.nodidx=torch.arange(self.Ms_length)
        # self.lin1=nn.Linear(self.dim,self.dim)
        # self.lin2=nn.Linear(self.dim,self.dim)
        self.alpha=3
        self.k=10

        self.window_size=configs.window_size
        self.multiadphyper=multi_adaptive_hypergraoh(configs)
        self.hyper_num1 = configs.hyper_num
        self.hyconv=nn.ModuleList()
        self.hyperedge_atten=SelfAttentionLayer(configs)
        for i in range (len(self.hyper_num1)):
            # self.hyconv.append(HypergraphConv(configs.d_model, configs.enc_in))
            self.hyconv.append(HypergraphConv(configs.enc_in, configs.enc_in))

        self.slicetran=nn.Linear(100,configs.pred_len)
        self.weight = nn.Parameter(torch.randn(self.pred_len, 76))

        self.argg = nn.ModuleList()
        for i in range(len(self.hyper_num1)):
            self.argg.append(nn.Linear(self.all_size[i],self.pred_len))
        self.chan_tran = nn.Linear(configs.enc_in, configs.enc_in)

    def forward(self, x,x_mark_enc):
        # normalization
        mean_enc=x.mean(1,keepdim=True).detach()
        x=x - mean_enc
        std_enc=torch.sqrt(torch.var(x,dim=1,keepdim=True,unbiased=False)+1e-5).detach()
        x=x / std_enc

        adj_matrix = self.multiadphyper(x)


        # x = x - seq_last
        # seq_enc = self.enc_embedding(x, x_mark_enc)
        # seq_enc = self.conv_layers(seq_enc)
        seq_enc = self.conv_layers(x)

        sum_hyper_list = []
        result_tensor1=[]
        for i in range(len(self.hyper_num1)):
            # mask=torch.tensor(adj_matrix[i], dtype=torch.long).to(x.device)
            mask = torch.tensor(adj_matrix[i]).to(x.device)
            ###尺度间关系
            node_value = seq_enc[i].permute(0,2,1)
            # node_value = torch.tensor(node_value, dtype=torch.float).to(x.device)
            node_value = torch.tensor(node_value).to(x.device)
            # node_value=seq_enc[i].to(x.device)
            edge_sums={}
            # scale_matrix=adj_matrix[i].to(x.device)
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
                # print(f"超边{edge_id}连接节点的索引和为: {sum_value}")
                # print(f"超边{edge_id}的形状为: {sum_value.shape}")
            # mask=mask.to(x.device)

            # kkk=edge_sums

            ###尺度内关系
            # output=self.hyconv[i](seq_enc[i].permute(0, 2, 1),mask).permute(0, 2, 1)
            output,constrainloss = self.hyconv[i](seq_enc[i], mask)
            result_tensor1.append(self.argg[i](seq_enc[i].permute(0, 2, 1)))
            # result_tensor1.append(self.argg[i](output.permute(0,2,1)))

            if i==0:
                result_tensor=output
                result_conloss=constrainloss
                # bbb = output
            else:
                result_tensor = torch.cat((result_tensor, output), dim=1)
                result_conloss+=constrainloss



        result_tensor1=sum(result_tensor1)/len(self.hyper_num1)
        # result_tensor1=torch.mean(result_tensor1,dim=2,keepdim=False)

        # kkk=result_tensor1
        sum_hyper_list=torch.cat(sum_hyper_list,dim=1)
        sum_hyper_list=sum_hyper_list.to(x.device)
        padding_need=80-sum_hyper_list.size(1)
        # pad=torch.nn.functional.pad(sum_hyper_list,(0, 0, 0, padding_need, 0, 0))
        hyperedge_attention=self.hyperedge_atten(sum_hyper_list)
        pad = torch.nn.functional.pad(hyperedge_attention, (0, 0, 0, padding_need, 0, 0))
        # hyperedge_attention=hyperedge_attention[:,:32,:]

        # inter_and_intra=torch.cat((result_tensor,hyperedge_attention),dim=1)



        # linear_layer=nn.Linear(inter_and_intra.size(1), 50)
        # output=linear_layer(inter_and_intra.permute(0,2,1)).to(x.device)

        # # attention
        # weight = nn.Parameter(torch.randn(self.pred_len, hyperedge_attention.size(1))).to(x.device)
        # inter_output=torch.matmul(hyperedge_attention.permute(0,2,1), self.weight.t()).permute(0,2,1)
        # inter_output=self.slicetran(hyperedge_attention.permute(0,2,1))

        ###concat
        concat_result=torch.cat((result_tensor,pad),dim=1)

        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            # x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
            # x_out=self.out_tran(x_out.permute(0,2,1)).permute(0,2,1)
            x = self.Linear(x.permute(0,2,1))
            # x_out1=self.concat_tra(concat_result.permute(0,2,1))#ori
            # x_out=self.in_tran(bbb.permute(0,2,1))
            x_out=self.out_tran(result_tensor.permute(0,2,1))###ori
            x_out_inter = self.inter_tran(pad.permute(0, 2, 1))
            # result_tensor1=self.chan_tran(result_tensor1.permute(0,2,1)).permute(0,2,1)
        # x = x+ x_out+inter_output
        # x=x.permute(0,2,1)
        x=x_out+x+x_out_inter
        x=self.Linear_Tran(x).permute(0,2,1)
        # x = self.chan_tran(x)
        # x = x + output.permute(0,2,1)
        x = x * std_enc + mean_enc
        # x = x  + mean_enc
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
        # self.fc = nn.Linear(self.out_channels * heads, self.out_channels)

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            # self.weight = Parameter(
            #     torch.Tensor(in_channels, heads * out_channels))
            self.weight = Parameter(
                torch.Tensor(in_channels, out_channels))
            # self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * int(out_channels / heads)))
            # aaa=self.att
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            # self.bias = Parameter(torch.Tensor(heads * out_channels))
            self.register_parameter('bias', None)

        self.reset_parameters()
    #初始化权重和偏置参数
    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)
    ####计算超图的度，然后根据度计算超图的归一化权重
    def __forward__(self,
                    x,
                    hyperedge_index,
                    hyperedge_weight=None,
                    alpha=None):

        D = degree(hyperedge_index[0], x.size(0), x.dtype)#[336]
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x.dtype)
        # --------------------------------------------------------
        B[B == float("inf")] = 0

        #调用propagate方法执行消息传递，传递信息为节点特征和归一化权重
        ####propogate执行消息聚合和更新节点特征的操作
        ####propogate执行两遍，因为需要执行源节点到目标节点和目标节点到源节点
        ####输出结果为超图卷积结果
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        # print("okkkk")
        # alpha=torch.matmul(out1, out)
        # alpha = softmax(alpha)
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # outla=torch.matmul(alpha, x)
        return out


    ####message在消息传递中计算每个节点收到的消息
    #####将输入的节点特征和超边归一化权重相乘
    ####并根据头数和输出通道数将结果重新组织
    def message(self, x_j, edge_index_i, norm, alpha):
        # out = norm[edge_index_i].view(-1, 1, 1) * x_j.view(-1, self.heads, self.out_channels)###origional
        out = norm[edge_index_i].view(-1, 1, 1) * x_j####
        if alpha is not None:
            out=alpha.unsqueeze(-1)*out
            # out = alpha.view(-1, self.heads, 1) * out
        return out
    ####forward是对__forward__方法的封装，传入了输入的节点特征和超图，返回超图卷积结果
    def forward(self, x, hyperedge_index, hyperedge_weight=None):
        # device = torch.device("cuda")
        x = torch.matmul(x, self.weight)
        x1=x.transpose(0,1)
        # hyperedge_weight=data.edge_attr
        alpha = None
        # x_new1=x.view(-1,self.heads,self.out_channels)
        x_i = torch.index_select(x1, dim=0, index=hyperedge_index[0])
        # f = open("result_draw.txt", 'a')
        # f.write(setting + "  \n")
        # f.write("{} {} {} {} {} {}".format(x_i[12][1], x_i[20][1], x_i[32][1], x_i[36][1], x_i[80][1], x_i[87][1]))
        # f.write('\n')
        # f.write('\n')
        # f.close()
        # print()
        edge_sums = {}
        # scale_matrix=adj_matrix[i].to(x.device)
        for edge_id, node_id in zip(hyperedge_index[1], hyperedge_index[0]):
            if edge_id not in edge_sums:
                edge_id = edge_id.item()
                node_id = node_id.item()
                edge_sums[edge_id] = x1[node_id, :, :]
            else:
                edge_sums[edge_id] += x1[node_id, :, :]
        result_list = torch.stack([value for value in edge_sums.values()], dim=0)  ####新的做法
        x_j = torch.index_select(result_list, dim=0, index=hyperedge_index[1])  ####新的做法
        loss_hyper = 0
        # s = 1
        # if s.device.type=='cuda':
        #     print("Tensor is on GPU")
        # else:
        #     print("Tensor is on CPU")
        """
        # 将字典中的张量转换为张量列表
        edge_sums_list = [tensor for tensor in edge_sums.values()]

        # 将张量列表堆叠成形状为 [M, N, 7] 的张量
        edge_sums = torch.stack(edge_sums_list, dim=0).transpose(0,1)

        inner_product_matrix = torch.matmul(edge_sums, edge_sums.transpose(1, 2))
        norm_q = torch.norm(edge_sums, dim=1, keepdim=True)
        distan_matrix = torch.norm(edge_sums.unsqueeze(1) - edge_sums.unsqueeze(0), dim=1)
        alpha = inner_product_matrix / (norm_q * norm_q.transpose(1, 2))
        loss_item = alpha * torch.pow(distan_matrix, 2) + (1 - alpha) * torch.pow(torch.clamp(torch.tensor(s) - distan_matrix, min=0.0))
        loss_hyper = torch.abs(torch.mean(loss_item))
        """
        ####原始计算方式
        for k in range(len(edge_sums)):
            for m in range(len(edge_sums)):
                inner_product = torch.sum(edge_sums[k] * edge_sums[m], dim=1, keepdim=True)
                norm_q_i = torch.norm(edge_sums[k], dim=1, keepdim=True)
                norm_q_j = torch.norm(edge_sums[m], dim=1, keepdim=True)
                alpha = inner_product / (norm_q_i * norm_q_j)
                # if alpha.device.type=='cuda':
                #     print("Tensor is on GPU")
                # else:
                #     print("Tensor is on CPU")
                distan = torch.norm(edge_sums[k] - edge_sums[m],dim=1, keepdim=True)
                # if distan.device.type=='cuda':
                #     print("Tensor is on GPU")
                # else:
                #     print("Tensor is on CPU")
                # loss_item=alpha*torch.pow(distan,2)+(1-alpha)*torch.pow(torch.clamp(torch.tensor(6.0)-distan,min=0.0),2)###双loss中超边约束为平方的形式
                loss_item = alpha * distan + (1 - alpha) * (torch.clamp(torch.tensor(4.2) - distan, min=0.0))
                loss_hyper += torch.abs(torch.mean(loss_item))
                '''
                alpha=torch.matmul(edge_sums[k].unsqueeze(1), edge_sums[m].unsqueeze(1).transpose(1, 2))
                norm_q_i = torch.norm(edge_sums[k].unsqueeze(1), dim=1, keepdim=True)
                norm_q_j = torch.norm(edge_sums[m].unsqueeze(1), dim=1, keepdim=True)
                alpha = alpha / (norm_q_i * norm_q_j)
                # alpha = abs(torch.mm(edge_sums[k].unsqueeze(-1), edge_sums[m].unsqueeze(-1).t()) / (torch.norm(edge_sums[k].unsqueeze(-1)) * torch.norm(edge_sums[m].unsqueeze(-1))))
                distan = torch.norm(edge_sums[k] - edge_sums[m])
                loss_item = alpha * distan ** 2 + (1 - alpha) * max(s - distan, 0) ** 2
                loss_hyper += loss_item
                '''
        # x_i=x1[hyperedge_index[0]]
        # x_j=x1[hyperedge_index[1]]
        # loss_hyper = loss_hyper / ((len(edge_sums) + 1))
        loss_hyper = loss_hyper / ((len(edge_sums) + 1)**2)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)  # [1008,1]
        alpha = softmax(alpha, hyperedge_index[0], num_nodes=x1.size(0))  # [1008,1]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        D = degree(hyperedge_index[0], x1.size(0), x.dtype)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x.dtype)
        B[B == float("inf")] = 0
        # D = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                    # dim=0, dim_size=num_nodes, reduce='sum')
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x1, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        out=out.transpose(0, 1)
        constrain_loss = x_i - x_j
        constrain_lossfin1=torch.mean(constrain_loss)
        constrain_losstotal = abs(constrain_lossfin1) + loss_hyper
        # constrain_lossfin1 = torch.sum(constrain_loss, dim=(0, 1, 2)) / (
        #             constrain_loss.size(0) * constrain_loss.size(1) * constrain_loss.size(2))


        """
        for i in range(x.size(0)):
            x_new = x[i, :, :]
            #####hyperedge_index_new[0]是节点索引，hyperedge_index_new[1]是超边索引
            # hyperedge_index_new = hyperedge_index[i, :, :]
            hyperedge_index_new=hyperedge_index

            if self.use_attention:
                # aaa=x[i,:,:]#####[32,223,512]--[223,512]
                # x=x[i,:,:]
                # hyperedge_index=hyperedge_index[i,:,:]
                # print(x_new.size(0))
                # x_new = x_new.view(-1, self.heads, self.out_channels)####[223,512]--[223,head,512]
                x_new = x_new.view(x_new.size(0), self.heads, -1) ####[223,512]--[223,head,512]
                ####把对应位置的值拿出来
                # x_i, x_j = x_new[hyperedge_index_new[0]], x_new[hyperedge_index_new[1]]  #####x_i,x_j[2,223,1,512]

                x_i=torch.index_select(x_new,dim=0,index=hyperedge_index_new[0])###新的做法
                edge_sums = {}
                # scale_matrix=adj_matrix[i].to(x.device)
                for edge_id, node_id in zip(hyperedge_index_new[1], hyperedge_index_new[0]):
                    if edge_id not in edge_sums:
                        edge_id = edge_id.item()
                        node_id = node_id.item()
                        edge_sums[edge_id] = x_new[node_id, :, :]
                    else:
                        edge_sums[edge_id] += x_new[node_id, :, :]

                kkk=len(edge_sums)
                loss_hyper=0
                s=1
                for k in range(len(edge_sums)):
                    for m in range (len(edge_sums)):
                        alpha=abs(torch.mm(edge_sums[k],edge_sums[m].t())/(torch.norm(edge_sums[k])*torch.norm(edge_sums[m])))
                        distan=torch.norm(edge_sums[k]-edge_sums[m])
                        loss_item=alpha * distan**2+(1-alpha)*max(s-distan,0)**2
                        loss_hyper+=loss_item




                # result_list = [value for value_list in edge_sums.values() for value in value_list]####无用代码
                # resultloss=loss_hyper/len(edge_sums)
                result_list=torch.stack([value for value in edge_sums.values()],dim=0)####新的做法
                x_j=torch.index_select(result_list,dim=0,index=hyperedge_index_new[1])####新的做法

                # print(result_list)
                # result_list = [value for value_list in edge_sums.values() for value in value_list]
                ###origional
                alpha = (torch.cat([x_i, x_j], dim=-1)* self.att).sum(dim=-1)####alpha=[223,head][1008,1]
                # alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)####alpha=[223,head]
                alpha = F.leaky_relu(alpha, self.negative_slope)#[1008,1]
                alpha = softmax(alpha, hyperedge_index_new[0], num_nodes=x_new.size(0))#[1008,1]
                # alpha = softmax(alpha, hyperedge_index_new[0], x_new.size(0))
                alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            constrain_loss = x_i - x_j
            # out_batch = self.__forward__(x[i,:,:], hyperedge_index[i,:,:], hyperedge_weight, alpha)
            out_batch = self.__forward__(x_new, hyperedge_index_new, hyperedge_weight, alpha)###out_batch=[223,1024]
            # out_batch = out_batch.view(-1,1, self.heads * self.out_channels)  ####out=[224,32]
            out_batch = out_batch.view(-1, 1, self.out_channels)

            # out_batch = self.fc(out_batch)####原来需要加维度转换
            if i==0:
                out=out_batch
                constrain_lossfin=constrain_loss
                loss_hyperfin=loss_hyper
            else:
                out=torch.cat((out,out_batch),1)
                constrain_lossfin = torch.cat((constrain_lossfin, constrain_loss), 1)
                loss_hyperfin+=loss_hyper
        # out_test=out###[223,32,512]
        out=out.transpose(0,1)
        constrain_lossfin = constrain_lossfin.transpose(0, 1)
        constrain_lossfin1=torch.sum(constrain_lossfin, dim=(0,1,2))/(constrain_lossfin.size(0)*constrain_lossfin.size(1)*constrain_lossfin.size(2))
        constrain_losstotal=constrain_lossfin1+loss_hyperfin




        # out = self.__forward__(x, hyperedge_index, hyperedge_weight, alpha)

        # if self.concat is True:
        #     out = out.view(-1, self.heads * self.out_channels)
        # else:
        #     out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        

        return out, constrain_losstotal
        """

        return out, constrain_losstotal
    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class multi_adaptive_hypergraoh(nn.Module):
    def __init__(self,configs):
        super(multi_adaptive_hypergraoh, self).__init__()
        self.seq_len = configs.seq_len
        self.window_size=configs.window_size
        self.inner_size=configs.inner_size
        self.dim=configs.d_model
        self.hyper_num=configs.hyper_num
        self.alpha=3
        self.k=3
        self.embedhy=nn.ModuleList()
        self.embednod=nn.ModuleList()
        self.linhy=nn.ModuleList()
        self.linnod=nn.ModuleList()
        # self.embedhy.append(nn.Embedding())
        # predf_hyper=pred_hyper(self.seq_len,self.window_size,self.inner_size,configs.device)
        for i in range(len(self.hyper_num)):
            # self.idx=torch.arange(self.hyper_num)
            # ttt=self.hyper_num[i]
            self.embedhy.append(nn.Embedding(self.hyper_num[i],self.dim))
            self.linhy.append(nn.Linear(self.dim,self.dim))
            self.linnod.append(nn.Linear(self.dim,self.dim))
            if i==0:
                self.embednod.append(nn.Embedding(self.seq_len,self.dim))
            else:
                # product=reduce(lambda x, y: x * y, self.window_size[:i+1])
                product=math.prod(self.window_size[:i])
                layer_size=math.floor(self.seq_len/product)
                self.embednod.append(nn.Embedding(int(layer_size),self.dim))

        self.dropout = nn.Dropout(p=0.1)
        # for i in range(len(se))
            # self.embednod.append(nn.Embedding())


    def forward(self,x):
        node_num = []
        node_num.append(self.seq_len)
        #window_size[4,4],node_num变为[336,84,21]
        for i in range(len(self.window_size)):
            layer_size = math.floor(node_num[i] / self.window_size[i])
            node_num.append(layer_size)
        hyperedge_all=[]
        node_all=[]

        #每个尺度的超边数量是超参[50,20,10]
        for i in range(len(self.hyper_num)):
            # ttt=self.hyper_num[i]
            # ttt=node_num[i]
            # predf_hyper = pred_hyper(node_num[i], self.window_size, self.inner_size)
            hypidxc=torch.arange(self.hyper_num[i]).to(x.device)

            nodeidx=torch.arange(node_num[i]).to(x.device)
            # hypervec=self.embedhy[i](hypidxc)
            # nodevec=self.embednod[i](nodeidx)

            # hyperen=torch.tanh(self.alpha*self.linhy[i](hypervec))
            # nodeec=torch.tanh(self.alpha*self.linnod[i](nodevec))
            hyperen=self.embedhy[i](hypidxc)
            nodeec=self.embednod[i](nodeidx)
            #dropout是新加的
            # hyperen=self.dropout(hyperen)
            # nodeec=self.dropout(nodeec)
            #生成点边关联矩阵
            # a=torch.mm(hyperen,nodeec.transpose(1,0))
            a = torch.mm(nodeec, hyperen.transpose(1, 0))
            adj=F.softmax(F.relu(self.alpha*a))
            # adj=F.relu(torch.tanh(self.alpha*a))

            # mask = torch.zeros(nodevec.size(0), hypervec.size(0)).to(x.device)
            ## mask=torch.zeros(hypervec.size(0),nodevec.size(0)).to(x.device)
            mask = torch.zeros(nodeec.size(0), hyperen.size(0)).to(x.device)
            mask.fill_(float('0'))
            # s1,t1=adj.topk(self.k,1)
            s1, t1 = adj.topk(min(adj.size(1),self.k), 1)
            mask.scatter_(1, t1, s1.fill_(1))
            adj = adj * mask
            adj = torch.where(adj > 0.5, torch.tensor(1).to(x.device), torch.tensor(0).to(x.device))
            # 去掉全为0的列
            adj = adj[:, (adj != 0).any(dim=0)]
            matrix_array = torch.tensor(adj, dtype=torch.int)
            result_list = [list(torch.nonzero(matrix_array[:, col]).flatten().tolist()) for col in
                           range(matrix_array.shape[1])]
            # 将list展平
            # sublist=[sublist for sublist in result_list if len(sublist) > 0]
            ##假设有四个节点，三条超边，则最终形成的矩阵形似如下,其中上面是节点集合，下面是超边集合
            # [1,2,3,1,2,4,2,3,4]
            # [1,1,1,2,2,2,3,3,3]
            node_list = torch.cat([torch.tensor(sublist) for sublist in result_list if len(sublist) > 0]).tolist()
            count_list = list(torch.sum(adj, dim=0).tolist())
            hperedge_list = torch.cat([torch.full((count,), idx) for idx, count in enumerate(count_list, start=0)]).tolist()

            # hyperedge_all+=hperedge_list
            # node_all+=node_list
            hypergraph=np.vstack((node_list,hperedge_list))
            # hypergraph=np.hstack((hypergraph,predf_hyper))
            hyperedge_all.append(hypergraph)





        a=hyperedge_all
        return a



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

        # 计算 attention 分数
        attention_scores = F.softmax(torch.matmul(q, k.transpose(1, 2)) / (k.shape[-1] ** 0.5), dim=-1)

        # 使用 attention 分数加权平均值
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

def pred_hyper(input_size, window_size, inner_size):
    all_size = []
    num_all = []
    intra_all = []
    j=0
    for i in range(input_size):
        if (i+1)%inner_size==0 or (i+1==input_size):
            left_side=max(i-inner_size+1,0)
            right_side=min(i+1,input_size)
            num=list(range(left_side,right_side))
            num_all+=num
            intra_edge = list(np.repeat(j, len(num)))
            intra_all += intra_edge
            j += 1


    intra_result6 = np.vstack((num_all, intra_all))
    # print(intra_result6)
    intra_result6 = torch.tensor(intra_result6, dtype=torch.long)
    # all_size = []
    # num_all = []
    # intra_all = []
    # j=0
    # all_size.append(input_size)
    # for i in range(len(window_size)):
    #     layer_size = math.floor(all_size[i] / window_size[i])
    #     all_size.append(layer_size)#####all_size=[169,7,1]
    #
    # for layer_idx in range(len(all_size)):
    #     start = sum(all_size[:layer_idx])
    #     for i in range(start, start + all_size[layer_idx]):
    #         if (i+1) % 4 == 0 or (i+1 == start + all_size[layer_idx]):
    #             left_side = max(i - inner_size, start)
    #             right_side = min(i+1 , start + all_size[layer_idx])
    #             num = list(range(left_side, right_side))
    #
    #             num_all += num
    #             intra_edge = list(np.repeat(j, len(num)))
    #             intra_all += intra_edge
    #             j += 1


    return intra_result6

