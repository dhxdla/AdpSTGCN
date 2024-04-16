import torch,pickle,os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils.data_factory import asym_adj,dist2adj

def load_data(args):
    print('Loading data from existence torch dataset.')
    dataset = {'train':None,'vali':None,'test':None}
    dataloader = {'train':None,'vali':None,'test':None}
    path = args.root_path + args.data_path
    for flag in ['train','vali','test']:
        dataset[flag],dataloader[flag]  = \
            load_existence_data(args,'{}_dataset'.format(flag),flag)
    scaler = dataset['train'].scaler
    return dataset,dataloader,scaler

def save_model(configs,model,epoch,exp_info, flag):
    '''
    saving model.
    '''
    if flag == False:
        return None
    dataset = configs['data_path'].split('/')[-2]
    saving_path = configs['root_path'] + configs['saving_path'] + configs['exp_start_time'] +'/' + dataset
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    log = 'Epoch_{}-Loss_{}'.format(str(epoch),str(round(exp_info.metrics_info['vali']['m_loss'][epoch-1],2)))
    torch.save(model.state_dict(), saving_path + '/' + log + '.pth')

def save_dict(path,name,dic):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save('{}{}.npy'.format(path,name),dic)

def load_best_model(configs,exp_info,current_epoch):
    bestid = np.argmin(list(exp_info.metrics_info['test']['m_loss']))
    best_epoch = bestid + 1
    if best_epoch == current_epoch:
        return None
    dataset = configs['data_path'].split('/')[-2]
    saving_path = configs['root_path'] + configs['saving_path'] + configs['exp_start_time'] +'/' + dataset
    log = 'Epoch_{}-Loss_{}'.format(str(best_epoch),str(round(exp_info.metrics_info['vali']['m_loss'][best_epoch-1],2)))
    engine = torch.load(saving_path + '/' + log + '.pth')
    log ='\n\t<loading best validation model,: {}>\n'.format(log)
    return engine,log

def load_existence_data(args,dataset_name,flag):
    '''
    root_path:
    data_path
    batch_size:
    drop_last:
    flag: ['train','test','vali']
    '''
    path = args.root_path + args.data_path
    data_set = torch.load(path+dataset_name)
    data_set.to_device(args.device,args.dtype)
    batch_size = args.batch_size
    if flag  == 'test':
        shuffle_flag = False
        drop_last = True
    else:
        shuffle_flag = True
        drop_last = True
    print('\t{:5}: {}'.format(flag, len(data_set)))
    data_loader = DataLoader(
        data_set,
        batch_size = batch_size,
        shuffle = shuffle_flag,
        drop_last = drop_last)
    return data_set, data_loader

def load_topo_adjs(path,n_nodes,order=1):
        adj_mx = load_adj(path, n_nodes)
        adjs = []
        for a in adj_mx:
            a = torch.tensor(a)
            adjs.append(a.unsqueeze(0))
            for _ in range(1,order):
                a = torch.matmul(a,a)
                adjs.append(a.unsqueeze(0))
        adjs = torch.cat(adjs,dim=0)
        return adjs.clone().detach()

def load_adj(adjdata,num_of_vertices=None):
    if adjdata[-3:]  == 'csv':
        dist_df = pd.read_csv(adjdata, header = 0)
        dist_df = dist_df.values
        edges = [(int(i[0]), int(i[1])) for i in dist_df]
        adj_mx = dist2adj(dist_df[:,2],edges,num_of_vertices,sigma=0.01)
    else:
        sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(adjdata)
    adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    return adj

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding = 'latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

