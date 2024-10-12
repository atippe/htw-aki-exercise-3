import torch
from torch.utils.data import Dataset
from os import listdir
import random
import pickle
import pandas

class BarCrawlDataset(Dataset):

    def __init__(self, seq_size):
        self.seq_size = seq_size
        self.paths = listdir('./Datasets/data/Samples/')
        random.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        #with open('Datasets/data/Samples/' + self.paths[idx], 'rb') as f:
        #    item = pickle.load(f)

        temp = pandas.read_pickle('./Datasets/data/Samples/' + self.paths[idx])
        item = dict(temp)

        readings = item['readings']
        x = torch.tensor(readings[['x','y','z']].values)
        if(len(x) < self.seq_size):
            x = torch.cat( (torch.zeros(self.seq_size-len(x), 3), x) )
        else:
            x = x[ len(x)-self.seq_size : len(x) ]
        y = item['tac']
        return x, y