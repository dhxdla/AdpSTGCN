import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Graph_Generator(nn.Module):
    def __init__(self,configs, topo_graph):
        super(Graph_Generator, self).__init__()
        # self.topo_graph = topo_graph
        # id_graph
        # id_graph = [torch.eye(configs['n_nodes']).unsqueeze(0) for _ in range(configs['order_topo_adj']*configs['num_topo_adj'])]
        # self.id_graph = torch.cat(id_graph,dim=0).to(configs['device'])
        # topo_graph = torch.cat(id_graph,dim=0)
        self.configs = configs
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(configs['dropout'])
        # hyper parameters
        self.gamma_lower = configs['gamma_lower']
        self.gamma_upper = configs['gamma_upper']
        # attn_graph
        fc_dim = configs['fc_dim_graph']
        self.attn_adj = attention_graph(configs)
        # negative filter projection
        c_in = c_out = configs['order_topo_adj']*configs['num_topo_adj']
        self.negative_projection = nn.Conv2d(c_in,c_out,(1,1))

    def graph_limite(self,A):
        A = F.sigmoid(A) # shoule be exist?
        A = A*(self.gamma_upper-self.gamma_lower)+self.gamma_lower
        A = torch.where(A < 0,torch.zeros_like(A),A) # be like relu activation
        A = torch.where(A > 1,torch.ones_like(A),A) # another style relu activation
        return A
    
    def forward(self,seq_x):
        # correlation and positive graph
        Adj_mt = self.attn_adj(seq_x)
        Adj_mt = self.graph_limite(Adj_mt)
        return Adj_mt
    
class attention_graph(nn.Module):
    def __init__(self,configs,model_type):
        super(attention_graph,self).__init__()
        self.model_type = model_type

        self.fc_dim = configs['fc_dim_graph']
        self.heads = configs['num_corr_adj'] if model_type=='positive' else configs['num_dist_adj']
        self.fc_q = nn.Linear(configs['seq_len'],self.fc_dim*self.heads)
        self.fc_k = nn.Linear(configs['seq_len'],self.fc_dim*self.heads)

    def forward(self,x):
        x = x.transpose(1,2)
        B,N,S = x.size()
        q = self.fc_q(x)
        q = q.reshape(B,N,self.heads,-1).transpose(1,2)
        k = self.fc_k(x)
        k = k.reshape(B,N,self.heads,-1).transpose(1,2)
        A_out = torch.matmul(q,k.transpose(2,3))
        A_out = A_out if self.model_type=='positive' else 1-A_out
        return A_out