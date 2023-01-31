import os
import glob
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


split_ = lambda x: x.split("/")[-1]

def find_tensor_files(root):
    return list(map(split_, glob.glob(root + "*")))


class MyDataset(Dataset):
    def __init__(self, files_folder, df):
        self.df = df
        self.files_folder = files_folder
        self.tensor_files = find_tensor_files(self.files_folder)
        self.audio_files = self.df['file'].values
        self.targets = self.df['target'].values
        
    def __len__(self):
        return len(self.tensor_files)
    
    def __getitem__(self, idx):
        tensor_file_name = self.tensor_files[idx]
        audio_file, target_idx = tensor_file_name[:-3].split("&*&")
        target_idx = int(target_idx)
        if target_idx >= len(self.df[self.df['file'] == audio_file]['target'].values[0]):
            print(audio_file, target_idx, tensor_file_name)
        target = int(self.df[self.df['file'] == audio_file]['target'].values[0][target_idx])
        x, y = torch.load(self.files_folder + tensor_file_name), torch.tensor(target)
        return x, y


class MyBalancedDataset(Dataset):
    def __init__(self, files_path, folder, df):
        self.df = df
        self.files = np.load(files_path)['arr_0']
        self.folder = folder
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_name = self.files[idx]
        audio_file, target_idx = file_name[:-3].split("&*&")
        target_idx = int(target_idx)
        target = int(self.df[self.df['file'] == audio_file]['target'].values[0][target_idx])
        x, y = torch.load(self.folder + file_name), torch.tensor(target)
        return x, y
    

