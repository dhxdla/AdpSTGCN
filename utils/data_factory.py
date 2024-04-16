import numpy as np
import scipy.sparse as sp
from torch.utils.data import DataLoader
from utils.data_process.dataset_generator import Dataset_with_Time_Stamp

def dist2adj(dist,edges,num_of_vertices,sigma):
    adj_mx = np.eye(int(num_of_vertices),dtype=np.float32)
    dist = dist/dist.max()
    for n,(i, j) in enumerate(edges):
        adj_mx[i, j] = np.exp(-dist[n]**2/sigma)
    return adj_mx

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat =  sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def generate_data(args, flag):
    '''
    args:
        batch_size:
        drop_last:
        root_path:
        data_path:
        flag: ['train','test','vali']
        size: ['label_len','pred_len']
        freq: '15min'
        period_type: ['week','month']
    '''
    batch_size = args.batch_size
    if flag  == 'test':
        shuffle_flag = False
        drop_last = True
    else:
        shuffle_flag = True
        drop_last = True

    data_set = Dataset_with_Time_Stamp(
        root_path = args.root_path,data_path = args.data_path,
        flag = flag,size = [args.label_len,args.pred_len],
        freq = args.freq,period_type = args.period_type)
    
    print(flag, len(data_set))
    
    data_loader = DataLoader(
        data_set,
        batch_size = batch_size,
        shuffle = shuffle_flag,
        drop_last = drop_last)
    return data_set, data_loader

