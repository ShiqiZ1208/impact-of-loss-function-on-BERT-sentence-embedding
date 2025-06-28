import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from datasets import load_dataset


def get_sts_dataset(dataset_name, split=0.3, modes = "defualt"):
    if dataset_name == 'STS-B':
        dataset_name = 'stsbenchmark'
    elif dataset_name == 'SICK-R':
        dataset_name = 'sickr'
    dataset = load_dataset(f'mteb/{dataset_name.lower()}-sts', split='test')
    dataset = dataset.rename_column('score', 'labels')
    dataset_split = dataset.train_test_split(test_size=split)
    train_dataset, test_dataset = dataset_split['train'], dataset_split['test']
    if modes == "defualt":
      dataset_split_val = train_dataset.train_test_split(test_size=0.1)
      train_dataset, val_dataset = dataset_split_val['train'], dataset_split_val['test']
      return train_dataset, test_dataset, val_dataset
    elif modes == "eval":
      return train_dataset, test_dataset



class STSDataset(torch.utils.data.Dataset):
    def __init__(self, sentence1, sentence2, label):
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.sentence1[idx], self.sentence2[idx], self.label[idx]

