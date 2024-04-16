import os
import numpy as np
import pandas as pd

def get_timestamp(start_time,intervals,num):
    month_list = [31,28,31,30,31,30,31,31,30,31,30,31]
    time_list = [start_time]
    ts_now = '{:d}-{:d}-{:d} {:d}:{:d}:{:d}'.format(*start_time,0)
    ts_list = [ts_now]
    if intervals[-3:] == 'min':
        intervals = int(intervals[:-3])
        now_time = start_time
        for i in range(num-1):
            mins = now_time[-1] + intervals
            hours = now_time[-2]
            days = now_time[-3]
            months = now_time[-4]
            years = now_time[-5]
            if mins >= 60:
                mins -= 60
                hours += 1
            if hours >= 24:
                hours -= 24
                days += 1
            if days > month_list[months-1]:
                days -= month_list[months-1]
                months += 1
            if months > 12:
                months -= 12
                years += 1
            now_time = [years,months,days,hours,mins]
            ts_now = '{:d}-{:d}-{:d} {:d}:{:d}:{:d}'.format(*now_time,0)
            time_list.append(now_time)
            ts_list.append(ts_now)
    return time_list,ts_list

def transform_file_npz(raw_path,target_path,file_name,start_time,intervals):
    name_list = ['flow','occ','speed']
    target_name = ['{}_{}.csv'.format(file_name,i) for i in name_list]
    print('\t\"{}.npz\" -> \"{}\", \"{}\", \"{}\"'.format(file_name,*target_name),end='')
    np_raw = np.load(raw_path)['data']
    num,n_nodes,n_feats = np_raw.shape
    time_list,ts_list = get_timestamp(start_time,intervals,num)
    columns = list(range(n_nodes))
    df_list = {
        'flow':[],
        'occ':[],
        'speed':[]
    }
    for i,time in enumerate(ts_list):
        df_list['flow'].append(np_raw[i,:,0].tolist())
        df_list['occ'].append(np_raw[i,:,1].tolist())
        df_list['speed'].append(np_raw[i,:,2].tolist())
    df_new = {}
    for key in df_list.keys():
        df_new.update({key:pd.DataFrame([],columns = ['date'] + columns,index = range(len(ts_list)))})
        df_new[key]['date'] = ts_list
        df_new[key].iloc[:,1:] = df_list[key]
        df_new[key].to_csv('{}{}_{}.csv'.format(target_path,file_name,key),index = False)
    print('\tdone.')

def transform_file_h5(raw_path,target_path,file_name):
    print('\t\"{}.h5\" -> \"{}.csv\"'.format(file_name,file_name),end='')
    df = pd.read_hdf(raw_path)
    # 提取values,行列名
    num_samples, num_nodes = df.shape
    data = df.values
    column = df.columns
    index = df.index.values
    #  重新编写dataframe 防止编码错误
    df_data = pd.DataFrame(data,columns = column)
    # 将毫无逻辑的 numpy.timestamp 转换为 pandas.timestamp
    df_data.insert(loc = 0,column = 'date',value = [pd.Timestamp(e) for e in index])
    # save to csv
    df_data.to_csv('{}{}.csv'.format(target_path,file_name),index = False)
    print('\tdone.')

def main():
    # file path and names 
    root_path = '/data1/pp21/TP_store'
    target_path = 'raw/'
    file_name = ['metr-la.h5','pems-bay.h5']
    path_list = [(root_path + e) for e in file_name]
    transform_file_h5(target_path,file_name,path_list)

if __name__  == '__main__':
    main()