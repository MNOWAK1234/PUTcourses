import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, file_path, flag='train', size=(96, 24)):
        self.seq_len = size[0]
        self.pred_len = size[1]
        self.file_path = file_path
        self.flag = flag
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.file_path)

        if 'date' in df_raw.columns:
            df_data = df_raw.drop(columns=['date'])
        elif 'DateTime' in df_raw.columns and 'Junction' in df_raw.columns:
            df_data = df_raw.pivot(index='DateTime', columns='Junction', values='Vehicles').fillna(0)
        else:
            df_data = df_raw.select_dtypes(include=[np.number])

        num_train = int(len(df_data) * 0.7)
        num_test = int(len(df_data) * 0.2)
        
        border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + int(len(df_data) * 0.1), len(df_data)]
        
        idx = {'train': 0, 'val': 1, 'test': 2}[self.flag]
        border1, border2 = border1s[idx], border2s[idx]

        train_data = df_data.values[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        data = self.scaler.transform(df_data.values)
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        # Return as [Channels, Time]
        return torch.FloatTensor(seq_x).t(), torch.FloatTensor(seq_y).t()

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

def get_dataloader(file_path, batch_size, flag='train', size=(96, 24)):
    dataset = TimeSeriesDataset(file_path, flag=flag, size=size)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(flag == 'train'), 
        drop_last=True
    )
    return dataloader, dataset