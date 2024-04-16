import torch
import pandas as pd,os
import numpy as np
from torch.utils.data import Dataset
import warnings

class DataScaler():
    def __init__(self):
        self.mean = 0
        self.std = None
        self.eps = 0.0

    def to_tensor(self,device,dtype = torch.float):
        self.mean = torch.tensor(self.mean,dtype = dtype).to(device)
        self.std = torch.tensor(self.std,dtype = dtype).to(device)
    
    def fit_transform(self,data):
        self.fit(data)
        return self.transform(data)
    
    def fit(self,data):
        self.mean = data.mean()
        self.std = data.std() + self.eps

    def transform(self,data):
        data = ((data - self.mean)/self.std)
        return data

    def inverse_transform(self,data):
        data = data * self.std + self.mean
        return data

class Dataset_with_Time_Stamp(Dataset):
    def __init__(self,
                 root_path, data_path,dtype = torch.float,
                 flag = 'train', size = None,scale = True,
                 freq = '15min',period_type = 'week'):
        
        self.period_type = period_type
        self.label_len = size[0]
        self.pred_len = size[1]
        self.seq_len = self.label_len +self.pred_len
        # init
        assert flag in ['train', 'test', 'vali']
        type_map = {'train': 0, 'vali': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.dtype = dtype
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path))
        freq = float("".join(list(filter(str.isdigit, self.freq))))
        if self.period_type  == 'week':
            period_len = df_raw.shape[0]/ (60 / freq * 24 * 7)
            num_unit_period = int(60/freq * 24 * 7)
        elif self.period_type  == 'month':
            period_len = df_raw.shape[0]/ (60 / freq * 24 * 30)
            num_unit_period = int(60/freq * 24 * 30)
        test_period = int(period_len * 0.2)
        val_period = test_period
        train_period = int(period_len - 2*test_period)
        border1s = np.array([
            0,
            train_period,
            (train_period + test_period)])* num_unit_period
        border2s = np.array([
            train_period,
            train_period + test_period,
            train_period + test_period + val_period]) * num_unit_period
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        if self.scale: 
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler = DataScaler()
            train_data = self.scaler.fit_transform(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            self.scaler = None
            data = df_data.values
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
        df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_stamp = df_stamp.drop('date', 1).values

        self.data = torch.tensor(data[border1:border2],dtype = self.dtype)
        self.data_stamp = torch.tensor(data_stamp,dtype = self.dtype)
    
    def to_device(self,device='cpu',dtype=None):
        if dtype is None:
            dtype = self.dtype
        self.data = self.data.to(device=device,dtype=dtype)
        self.data_stamp = self.data_stamp.to(device=device,dtype=dtype)
        return 
    
    def __getitem__(self, index):
        begin = index
        med = begin + self.label_len
        end = begin + self.seq_len
        seq_x = self.data[begin:med]
        seq_y = self.data[med:end]
        seq_x_mark = self.data_stamp[begin:med]
        seq_y_mark = self.data_stamp[med:end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def _set_dtype(self,dtype):
        self.data = self.data.to(dtype)
        self.data_stamp = self.data_stamp.to(dtype)

    def __len__(self):
        return len(self.data)-self.seq_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
if __name__  == '__main__':
    root_path = '/data1/traffic/'
    data_path = 'metr-la.csv'
    train_dataset = Dataset_with_Time_Stamp(root_path,data_path,flag = 'train',size = (12,12),scale = True,freq = '15min',period_type = 'month')
    print(None)