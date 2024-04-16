import copy,os,warnings
import numpy as np
import pandas as pd

def find_the_best(dic,flag_targets,metric_target,exclude = [None],max_try = 10000):
    dic = copy.deepcopy(dic)
    keys_flag = list(dic.keys())
    keys_metric = list(dic[keys_flag[0]].keys())
    if (not flag_targets in keys_flag) and (not metric_target in keys_metric):
        return None
    # n_epochs = len(dic[flag_targets][metric_target])
    for i in range(max_try):
        best = np.min(dic[flag_targets][metric_target])
        idx = np.argmin(dic[flag_targets][metric_target])
        if not metric_target in ['loss','mape','rmse']:
            if best in exclude:
                dic[flag_targets][metric_target][idx] = np.inf
            else:
                break
        else:
            n_iters = len(dic[flag_targets][metric_target][0])
            idx1 = idx // n_iters
            idx2 = idx - idx1 * n_iters
            idx = (idx1,idx2)
            if best in exclude:
                dic[flag_targets][metric_target][idx1][idx2] = np.inf
            else:
                break
    return best,idx,i

def get_title(metrics):
    # 'Notes',
    title = ['Notes', 
             'Curre Test MAE','Curre Test MAPE','Curre Test RMSE','Min Vali MAE Idx',
            'Min Train MAE','Min Vali MAE','Min Test MAE','Min Test MAPE','Min Test RMSE',]
    title = title + list(metrics['hyperparameters'].keys())
    title = title + ['Current Mae 12 Step'] + list(range(1,metrics['hyperparameters']['seq_len'][0]+1))
    title = title + ['Current mape 12 Step'] + list(range(1,metrics['hyperparameters']['seq_len'][0]+1))
    title = title + ['Current rmse 12 Step'] + list(range(1,metrics['hyperparameters']['seq_len'][0]+1))
    return title

def get_statistics(metrics):
    min_train_mae = np.min(metrics['train']['m_mae'])
    min_vali_idx = np.argmin(metrics['vali']['m_mae'])
    min_vali_mae = np.min(metrics['vali']['m_mae'])
    min_test_mae = np.min(metrics['test']['m_mae'])
    min_test_mape = np.min(metrics['test']['m_mape'])
    min_test_rmse = np.min(metrics['test']['m_rmse'])
    test_mae_idxed = metrics['test']['m_mae'][min_vali_idx]
    test_mape_idxed = metrics['test']['m_mape'][min_vali_idx]
    test_rmse_idxed = metrics['test']['m_rmse'][min_vali_idx]
    # metrics['notes']
    if not 'notes' in metrics.keys():
        metrics['notes'] = ''
    statistics = [metrics['notes'],
                  test_mae_idxed,test_mape_idxed,test_rmse_idxed,min_vali_idx+1,
                  min_train_mae,min_vali_mae,min_test_mae,min_test_mape,min_test_rmse,] +\
                    [v[0] for v in metrics['hyperparameters'].values()] +\
                    [min_vali_idx+1] + list(np.mean(metrics['test']['mae_all'][min_vali_idx],axis=0)) +\
                    [min_vali_idx+1] + list(np.mean(metrics['test']['mape_all'][min_vali_idx],axis=0)) +\
                    [min_vali_idx+1] + list(np.mean(metrics['test']['rmse_all'][min_vali_idx],axis=0))
    return statistics

def excel_statistics(files,path,saving_path):
    print('Processing.')
    files_name = os.listdir(path) if files  == 'all' else files
    files_name.sort()
    flag = True
    this_path = os.path.dirname(os.path.abspath(__file__))
    if saving_path == '':
        saving_path = this_path + '/'
    saving_name = 'metrics_info.xlsx'
    writer = pd.ExcelWriter(saving_path+saving_name)		# 写入Excel文件
    for i,name in enumerate(files_name):
        obj = np.load(path + name, allow_pickle = True)
        metrics = obj.item().metrics_info
        print('\t {:02d}/{:02d}: <{:s}>.'.format(i+1,len(files_name),name))
        # '''Train Validation Test mae every epoch'''
        index = [name, '', '']
        train_mae = [
            metrics['train']['m_mae'],
            metrics['vali']['m_mae'],
            metrics['test']['m_mae']]
        train_mae = pd.DataFrame(train_mae, index=index, columns=list(range(1,len(metrics['train']['m_mae'])+1)))
        if flag:
            train_mae.to_excel(writer, 'page_1', startrow=0)
        else:
            train_mae.to_excel(writer, 'page_1', startrow=1+(i*3), header=None)		# ‘page_1’是写入excel的sheet名

        # '''Statistics Infomation'''
        title = get_title(metrics)
        # state_info
        statistics_info = get_statistics(metrics)
        statistics_info = pd.DataFrame([statistics_info], index=[name], columns=title)
        if flag:
            statistics_info.to_excel(writer, 'page_2', startrow=0 ,float_format = '%.15f')
        else:
            statistics_info.to_excel(writer, 'page_2', startrow=1+i, header=None ,float_format = '%.15f')
        flag = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        writer.save()
    writer.close()
    print('\tSaving Path: \'{:s}\'.\nDone.'.format(saving_path+saving_name))

if __name__  == '__main__':
    names = 'all'
    path = './results/dict/'
    saving_path = ''
    excel_statistics(names,path,saving_path)