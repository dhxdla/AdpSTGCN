import torch
import torch.nn as nn

class ST_DATA_EMBEDDING(nn.Module):
    def __init__(self,configs):
        super(ST_DATA_EMBEDDING,self).__init__()
        self.configs = configs
        self.value_embed = nn.Linear(1,configs['d_model'])
        # self.emb = nn.Parameter(torch.randn((configs['batch_size'],configs['d_model'],configs['seq_len'],configs['n_nodes'])))
        # self.temporal_embed = TemporalEmbedding(configs['d_model'])
        # self.alpha = configs['alpha']
        # self.dropout = nn.Dropout(configs['dropout'])

    def forward(self,inputs,inputs_mark):
        inputs = inputs.unsqueeze(-1)
        x1 = self.value_embed(inputs).permute(0,3,1,2)# + self.emb
        # x2 = self.temporal_embed(inputs_mark).transpose(1,2).unsqueeze(-1)
        return x1
    
class TemporalEmbedding(nn.Module):
    def __init__(self,d_model) -> None:
        super(TemporalEmbedding,self).__init__()
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        self.minute_embed = nn.Embedding(minute_size,d_model)
        self.hour_embed = nn.Embedding(hour_size,d_model)
        self.weekday_embed = nn.Embedding(weekday_size,d_model)
        self.day_embed = nn.Embedding(day_size,d_model)
        self.month_embed = nn.Embedding(month_size,d_model)
    
    def forward(self,inputs):
        inputs = inputs.to(torch.long)
        min_out = self.minute_embed(inputs[:,:,4])
        hour_out = self.hour_embed(inputs[:,:,3])
        weekday_out = self.weekday_embed(inputs[:,:,2])
        day_out = self.day_embed(inputs[:,:,1])
        month_out = self.month_embed(inputs[:,:,0])
        out = min_out+hour_out+weekday_out+day_out+month_out
        return out

class Graph_Gen_Embedding(nn.Module):
    def __init__(self, c_in, c_out, n_nodes):
        super(Graph_Gen_Embedding, self).__init__()
        self.FixEmbedding = nn.Parameter(torch.randn(1, c_out, n_nodes))
        self.ValueEmbedding = nn.Conv1d(c_in,1,1)

    def forward(self, x):
        x_e = self.ValueEmbedding(x)
        x_embedding = x + x_e + self.FixEmbedding
        return x_embedding