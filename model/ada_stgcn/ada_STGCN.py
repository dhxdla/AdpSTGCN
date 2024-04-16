from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import IPython

def asym_adj(adj):
    rowsum = adj.sum(3)
    d_inv =  rowsum.pow(-1)
    d_inv[torch.isinf(d_inv)] = 0.
    d_mat = torch.diag_embed(d_inv)
    return torch.einsum('bnij,bnjk->bnik',[d_mat,adj])

class GATv2(nn.Module):
    def __init__(self, c_in, c_out, nhead, bias, alpha, drop, concat):
        super(GATv2, self).__init__()
        self.concat = concat
        self.nhead = nhead
        if self.concat:
            self.c_hidden = c_out // self.nhead
        else:
            self.c_hidden = c_out
        self.l = nn.Linear(c_in, self.c_hidden*nhead, bias=bias)
        self.r = nn.Linear(c_in, self.c_hidden*nhead, bias=bias)
        self.attn = nn.Linear(self.c_hidden, 1)
        self.activation = nn.LeakyReLU(alpha)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(drop)
        # self.drop = drop

    def forward(self, x, adj):
        # x: (batch, spatial_dim, n_nodes, temporal_dim)
        batch, n_nodes, time_dim =  x.shape[0], x.shape[2], x.shape[3]
        # x: (batch, temporal_dim, n_nodes, spatial_dim)
        x = x.transpose(3,1)
        # x_l: (batch*temporal_dim, n_nodes, n_head, hidden)
        x_l = self.l(x).view(batch*time_dim, n_nodes, self.nhead, self.c_hidden)
        x_r = self.r(x).view(batch*time_dim, n_nodes, self.nhead, self.c_hidden)
        
        
        # x_sum = x_l.transpose(1,3).unsqueeze(-1) + x_r.transpose(1,3).unsqueeze(-2)
        # x_sum = x_sum.permute(0, 3, 4, 2, 1)
        
        
        x_l_repeat = x_l.repeat(1, n_nodes, 1, 1)
        x_r_repeat_interleave = x_r.repeat_interleave(n_nodes, dim=1)
        x_sum = x_l_repeat + x_r_repeat_interleave
        # n_nodes, n_nodes, self.n_heads, self.n_hidden
        x_sum = x_sum.view(batch*time_dim, n_nodes, n_nodes, self.nhead, self.c_hidden)
        
        e = self.attn(self.activation(x_sum))
        # e: (batch*temporal_dim, n_nodes, n_nodes, n_head)
        e = e.squeeze(-1)
        # 看一下
        # 原文中 adj (n_nodes, n_nodes, n_head)
        # e: (batch*temporal_dim, n_nodes, n_nodes, n_head)
        if len(adj.shape) == 2:
            adj = adj.repeat(1,self.nhead).view(n_nodes,n_nodes,self.nhead)
            e = e.masked_fill(adj==0,float('-inf'))
        else:
            adj = adj.repeat(1,1,self.nhead).view(batch,n_nodes,n_nodes,self.nhead)
            e = e.view(batch, time_dim, n_nodes, n_nodes, self.nhead)
            e = e.masked_fill(adj.unsqueeze(1)==0,-1e20)
            # e = e.masked_fill(adj.unsqueeze(1)==0,float('-inf'))
            e = e.view(batch*time_dim, n_nodes, n_nodes, self.nhead)
        a = self.softmax(e)
        a = self.dropout(a)
        # a = F.dropout(a, self.drop, training=self.training)
        # attn_res: (batch*time_dim, nodes, heads, hidden)
        attn_res = torch.einsum('bijh,bjhf->bihf', a, x_r)
        if self.concat:
            res = attn_res
        else:
            res = attn_res.mean(dim=-2)
        # res = res.view(batch, time_dim, n_nodes, -1)
        res = res.reshape(batch, time_dim, n_nodes, -1)
        return res.transpose(1,3)

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class GCN(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3):
        super(GCN,self).__init__()
        self.mlp = nn.ModuleList()
        self.mlp.append(linear(c_in,c_out))
        for _ in range(support_len):
            self.mlp.append( linear(4*c_in,c_out) )
        self.fus = linear(support_len*c_out+c_out, c_out)
        # self.fus = linear(4*support_len*c_in+c_in, c_out)
        self.dropout = dropout
        self.K = [10, 20, 40]

    def build_knn_neighbourhood(self, atten, topk, markoff_value):
        topk = min(topk, atten.size(-1))
        matrixs = []
        for attention in atten:
            knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
            weighted_adjacency_matrix = (markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val)
            matrixs.append(weighted_adjacency_matrix.unsqueeze(0))
        matrixs = torch.cat(matrixs, dim=0)
        return matrixs

    def forward(self, x, support):
        # out = [x]
        out = [self.mlp[0](x)]
        for i in range(support.shape[1]):
            a = support[:,i,:,:]
            out1 = []
            for j in range(3):
                a1 = F.softmax(self.build_knn_neighbourhood(a, self.K[j],-1e20), dim=-1 )
                x1 = torch.einsum('ncvl,nvw->ncwl', [x,a1])
                # out.append(x1)
                out1.append(x1)
            a1 = F.softmax(a, dim=-1)
            x1 = torch.einsum('ncvl,nvw->ncwl', [x,a1])
            
            out1.append(x1)
            out1 = torch.cat(out1, dim=1)
            out.append(self.mlp[i+1](out1))
            
            # out.append(x1)
            # x1 = torch.einsum('ncvl,nvw->ncwl', [x,support[:,i,:,:]])
            # out.append(self.mlp[i](x1))
        out = torch.cat(out, dim=1)
        out = self.fus(out)
        out = F.dropout(out,p=self.dropout, training=self.training)
        return out

class sample(nn.Module):
    def __init__(self):
        super(sample,self).__init__()
        self.K = self.K = [10, 20, 40, 500]
        # self.K = self.K = [5, 40, 100, 500]
   
    def build_knn_neighbourhood(self, atten, topk, markoff_value):
        topk = min(topk, atten.size(-1))
        # knn_val, knn_ind = torch.topk(atten, topk, dim=-1)
        # matrixs = (markoff_value * torch.ones_like(atten)).scatter_(-1, knn_ind, knn_val)
        matrixs = []
        for attention in atten:
            knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
            
            weighted_adjacency_matrix = (markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val)
            matrixs.append(weighted_adjacency_matrix.unsqueeze(0))
        matrixs = torch.cat(matrixs, dim=0)
        return matrixs

    def forward(self, attention):
        a = []
        for k in range(attention.shape[1]):
            a.append( F.softmax(self.build_knn_neighbourhood(attention[:,k,:,:], self.K[k], -1e20).unsqueeze(1), dim=-1) )
        # a.append( F.softmax(attention[:,-1,:,:], dim=-1) )
        # a = torch.cat(a, dim=2)
        a = torch.cat(a, dim=1)
        return a

#self.dyn.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.ndynadj,dyn=True))
class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2,dyn=False):
        super(gcn,self).__init__()
        self.nconv = nconv()
        # self.diffusion = True
        self.diffusion = False
        self.MLP = True
        # self.MLP = False
        self.dyn = dyn
        c_i = c_in
        if dyn:
            self.weight = nn.Parameter( torch.randn(support_len+1) )
            c_in = (support_len+1)*c_in
            # c_in = 2 * c_in
        else:
            if self.diffusion:
                c_in = (order*support_len+1)*c_in
            else:
                c_in = (order*support_len+1)*c_in
                # c_in = 96
                # c_in = (support_len + 1)*c_in
        if self.MLP:
            self.mlp = linear(c_in,c_out)
        else:
            self.mlp = nn.ModuleList()
            for _ in range(support_len + 1):
                self.mlp.append(linear(c_i, c_out))
            self.fus = linear(5*c_out, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        #x: batch, hidden, node, 13
        if not self.dyn:
            out = [x]
            if not self.diffusion:
                batch, fea, nodes, tem = x.shape
                x1 = torch.einsum('ncvl,bvw->nbcwl',(x,support))
                out.append( x1.reshape(batch, -1 ,nodes, tem) )
            else:
                for a in support:
                    x1 = self.nconv(x,a)
                    out.append(x1)
                    for _ in range(2, self.order + 1):
                        x2 = self.nconv(x1,a)
                        out.append(x2)
                        x1 = x2
        else:
            if self.MLP:
                # out = [x*self.weight[0]]
                out = [x]
            else:
                out = [self.mlp[0](x)]
            batch, fea, nodes, tem = x.shape
            # x1 = torch.einsum('ncvl,nvw->ncwl',(x,support))
            # x1 = x1.view(batch, fea, nodes, -1, tem)
            # out.append( x1.transpose(1,3).reshape(batch, -1 ,nodes, tem) )
            x1 = torch.einsum('ncvl,nbvw->nbcwl',(x,support))
            out.append( x1.reshape(batch, -1 ,nodes, tem) )
        if self.MLP:
            h = torch.cat(out,dim=1)

            h = self.mlp(h)
        else:
            h = torch.cat(out,dim=1)
            h = self.fus(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class Embedding(nn.Module):
    def __init__(self, nnodes, dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Parameter(torch.randn(1, nnodes, dim))
        self.embds = nn.Linear(dim, 1)
        # self.embds = nn.Linear(dim, 11)

    def forward(self, x):
        x_e = self.embds(x)

        # x_embedding = x + x_e + self.embedding
        x_embedding = x_e + self.embedding

        return x_embedding

class Graph_Generator(nn.Module):
    def __init__(self, method, c_in, c_hidden1, nheads, nnodes, drop):
        # 12, 1, 32, 16
        super(Graph_Generator, self).__init__()
        self.drop = drop
        self.method = method
        self.nheads = nheads
        
        if self.method == 'attention':
            self.wl = nn.Linear(c_in, c_hidden1*nheads)
            self.wr = nn.Linear(c_in, c_hidden1*nheads)
            # self.norm = 1 / sqrt(16)
            self.norm = 1 / sqrt(c_hidden1)

    def forward(self, features):
        #features: batch,1,ndoe,11
        feature = features.squeeze(1)

        if self.method == 'attention':
            l = self.wl(feature)
            # l: batch,node,hidden*head
            l = l.reshape(l.shape[0], l.shape[1], self.nheads, -1).permute(0,2,1,3)
            # l: batch, 
            r = self.wr(feature)
            r = r.reshape(r.shape[0], r.shape[1], self.nheads, -1).permute(0,2,1,3)
            # [64, n, 156, 156]

            attention = torch.matmul(l, r.transpose(3,2)) * self.norm
            # attention = torch.matmul(l, r.transpose(3,2))
            #attention: batch, head, node, node
        return attention

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = torch.abs(adj)
    adj[adj<10e-5] = 10e-5
    d = adj.sum(-1)
    d_inv_sqrt = torch.pow(d, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    e = torch.eye(adj.shape[1]).unsqueeze(0)
    e = e.repeat(adj.shape[0], 1, 1).to(adj.device)
    normalized_laplacian = e - torch.bmm( torch.bmm(adj,d_mat_inv_sqrt).transpose(1,2), d_mat_inv_sqrt)
    # normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    # normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def nl(adj):
    a = []
    for k in range(adj.shape[1]):
        a.append( calculate_normalized_laplacian(adj[:,k,:,:]).unsqueeze(1) )
    a = torch.cat(a, dim=1)
    return a

class ada_gcn(nn.Module):
    def __init__(self, configs, supports):
        super(ada_gcn, self).__init__()
        device = configs['device']
        num_nodes = configs['n_nodes']
        dropout = configs['dropout']
        blocks = configs['n_blocks']
        layers = configs['n_layers']
        self.blocks = configs['n_blocks']
        self.layers = configs['n_layers']
        self.addaptadj = configs['addaptadj']
        addaptadj = configs['addaptadj']
        aptinit = None
        gcn_bool = True
        self.gcn_bool = gcn_bool
        device = configs['device']
        num_nodes = configs['n_nodes']
        dropout=configs['dropout']
        in_dim = configs['in_dim']
        out_dim = configs['out_dim']
        residual_channels = configs['d_model']
        dilation_channels = configs['d_model']
        skip_channels = configs['skip_channels']
        end_channels = configs['end_channels']
        kernel_size=configs['kernel_size']
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.ndynadj = 4
        self.graph_generator = Graph_Generator('attention', 12, 32, self.ndynadj, num_nodes, self.dropout)
        # self.embedding = Embedding(num_nodes, 11)
        self.embedding = Embedding(num_nodes, 12)
        # self.embedding = nn.Linear(11, 1)
        self.sample = sample()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.dyn = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports
        self.alpha = nn.Parameter(torch.FloatTensor(1),requires_grad=True)
        self.alpha.data.fill_(0.5)
        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        for _ in range(blocks):
            #block=4,kernel_size=2,layers=2
            additional_scope = kernel_size - 1
            new_dilation = 1
            #new_dilation: 1->2
            #receptive_field: 1->13
            for _ in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                
                self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))
                self.dyn.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.ndynadj,dyn=True))
                # self.dyn.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.ndynadj,dyn=True))
                # self.dyn.append(GCN(dilation_channels,residual_channels,dropout,self.ndynadj))
                # GATv2
                # self.dyn.append(GATv2(dilation_channels,residual_channels, 2, bias=True, alpha=0.2, drop=dropout, concat=True))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):

        input = input.transpose(1,2).unsqueeze(1)
        #input: batch, hidden, node, step
        dyn_adj = None

        support = []

        #self.supports: 2,node,node
        for i in range(len(self.supports)):
            
            a = self.supports[i]

            support.append(a.unsqueeze(0))
            support.append(torch.mm(a,a).unsqueeze(0))
        support = torch.cat(support, dim=0)

        #support: 4,node,node


        graph_signal = input
        #graph_signal: batch,hidden,node,11

        graph_signal = self.embedding(graph_signal)
        #graph_signal: batch,hidden,node,11

        dyn_adj = self.graph_generator(graph_signal)
        #dyn_adj: batch, head, node, node

        dyn = self.sample(dyn_adj)
        #dyn: batch, head, node, node

        # dyn = nl(dyn_adj)
        in_len = input.size(3)
        #self.receptive_field: 13
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        #x: batch, 1, node, 13
        x = self.start_conv(x)
        #x: batch, hidden, node, 13
        skip = 0

        for i in range(self.blocks * self.layers):
            
            residual = x
            
            filter = self.filter_convs[i](residual)
            #dialation:12121212,融合时间信息

            filter = torch.tanh(filter)
            
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            #门控融合

            s = x
            s = self.skip_convs[i](s)
            #s: batch, hidden, node, 13-num
            #hidden:32->256
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            #x: batch,hidden(32),node,step  
            x = self.dyn[i](x, dyn)
            #x: batch,hidden(32),node,step

            x = self.gconv[i](x, support)
            # x = self.alpha * self.gconv[i](x, support) + (1-self.alpha) * self.dyn[i](x, dyn) 

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x.squeeze(-1)