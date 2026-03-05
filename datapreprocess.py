import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from datasets import Value
from datasets import concatenate_datasets
from collections import defaultdict
import random

def remap_labels_batch(batch):
    batch["labels"] = [
        1 if label == 0 else 0
        for label in batch["labels"]
    ]
    return batch

def select_columns(example):
    columns_to_keep = ["sentence1", "sentence2", "labels"]
    return {col: example[col] for col in columns_to_keep}

def rename_columns(dataset):
    dataset = dataset.rename_column('premise', 'sentence1')
    dataset = dataset.rename_column('hypothesis', 'sentence2')
    dataset = dataset.rename_column('label', 'labels')
    dataset = dataset.cast_column("labels", Value("float32"))
    dataset_filtered = dataset.filter(
        lambda batch: [label in {0, 2} for label in batch["labels"]],
        batched=True
        )
    dataset = dataset_filtered.map(remap_labels_batch, batched=True)
    return dataset

def Triplet_dataset(dataset_name):
    #print(dataset_name)
    if dataset_name == "snli":
        dataset = load_dataset(f'stanfordnlp/{dataset_name.lower()}', split='train')
        dataset = rename_columns(dataset)
    elif dataset_name == "multi_nli":
        dataset = load_dataset(f'nyu-mll/{dataset_name.lower()}', split='train')
        dataset = rename_columns(dataset)
        dataset = dataset.map(select_columns,
        remove_columns=[col for col in dataset.column_names if col not in ["sentence1", "sentence2", "labels"]])
    elif dataset_name == "nli":
        snli = load_dataset(f'stanfordnlp/snli', split='train')
        snli = rename_columns(snli)
        multinli = load_dataset(f'nyu-mll/multi_nli', split='train')
        multinli = rename_columns( multinli)
        multinli = multinli.map(select_columns,
        remove_columns=[col for col in multinli.column_names if col not in ["sentence1", "sentence2", "labels"]])
        dataset = concatenate_datasets([snli, multinli])
    pairs = defaultdict(lambda: {"positive": [], "negative": []})

    for data in dataset:
      if data["labels"] == 1:  # entailment
          pairs[data["sentence1"]]["positive"].append(data["sentence2"])
      elif data["labels"] == 0:  # contradiction
          pairs[data["sentence1"]]["negative"].append(data["sentence2"])
    valid_triplet = []

    for premise, data in pairs.items():
      if len(data["positive"]) > 0 and len(data["negative"]) > 0:  # only keep groups with both pos & neg
          for pos in data["positive"]:
              neg = random.choice(data["negative"])
              valid_triplet.append({
                  "anchor": premise,
                  "positive": pos,
                  "negative": neg
              })
    print(f"Total triplets: {len(valid_triplet)}")

    triplet_dataset = Dataset.from_list(valid_triplet)
    return triplet_dataset

def get_sts_dataset(dataset_name, is_triplet = False):
    '''
    get sts datasets using huggingface datasets.
    input: dataset_name(string)
    output: dataset (dictionary)
    '''
    #check all names of datasets
    if dataset_name == 'STS-B':
        dataset_name = 'stsbenchmark'
    elif dataset_name == 'SICK-R':
        dataset_name = 'sickr'

    if is_triplet == True:
        print("true")
        dataset = Triplet_dataset(dataset_name)
    elif dataset_name == "snli":
        dataset = load_dataset(f'stanfordnlp/{dataset_name.lower()}', split='train')
        dataset = rename_columns(dataset)
    elif dataset_name == "multi_nli":
        dataset = load_dataset(f'nyu-mll/{dataset_name.lower()}', split='train')
        dataset = rename_columns(dataset)
        dataset = dataset.map(select_columns,
         remove_columns=[col for col in dataset.column_names if col not in ["sentence1", "sentence2", "labels"]])
    elif dataset_name == "nli":
        snli = load_dataset(f'stanfordnlp/snli', split='train')
        snli = rename_columns(snli)
        multinli = load_dataset(f'nyu-mll/multi_nli', split='train')
        multinli = rename_columns( multinli)
        multinli = multinli.map(select_columns,
         remove_columns=[col for col in multinli.column_names if col not in ["sentence1", "sentence2", "labels"]])
        dataset = concatenate_datasets([snli, multinli])
    else:
        # load datasets from huggingface
        dataset = load_dataset(f'mteb/{dataset_name.lower()}-sts', split='test')
        # rename datasets columns score to labels
        dataset = dataset.rename_column('score', 'labels')

    return dataset



def prepare_dataset(dataset_name, split=0.3, is_triplet = False):
    '''
    split sts dataset into train test and validation (if needed)
    input: dataset_name (string), split (float), modes.
    output: train_set, test_set, and val_set (if needed)
    mode:
    1, defualt: split test and train set in splite ratio, and random get 1/10 set for validation.
    2, eval: split test and train set in splite ratio.
    '''
    # use get sts dataset to get dataset from huggingface datasets
    if is_triplet == False:
        dataset = get_sts_dataset(dataset_name)
    else:
        print("Triplet")
        dataset =  get_sts_dataset(dataset_name, is_triplet = True)

    # use dataset.train_test_split to split datasets into test and train
    if dataset_name not in ["snli","multi_nli","nli","triplet"]:
        dataset_split = dataset.train_test_split(test_size=split)
        train_dataset, test_dataset = dataset_split['train'], dataset_split['test']
    else:
        test_dataset = []
        train_dataset = dataset
        datasets = ['STS-B', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'SICK-R']
        for item in datasets:
            test_dataset.append(get_sts_dataset(item))
        
    
    return train_dataset, test_dataset


class STSDataset(torch.utils.data.Dataset):
    '''
    an class for STSDataset
    have function len and getitem for training process
    '''
    def __init__(self, sentence1, sentence2, label):
        self.label = label
        self.sentence1 = sentence1
        self.sentence2 = sentence2

    def __len__(self):

        # use the len of label to get the length of dataset
        return len(self.label)

    def __getitem__(self, idx):

        # return the Setnence1, Sentence2 and the label for each data
        return self.sentence1[idx], self.sentence2[idx], self.label[idx]

class TripDataset(torch.utils.data.Dataset):
      '''
      an class for triplet_Dataset
      have function len and getitem for training process
      '''
      def __init__(self, anchor, positive, negative):
          self.anchor = anchor
          self.positive = positive
          self.negative = negative

      def __len__(self):
        return len(self.anchor)

      def __getitem__(self, idx):
        return self.anchor[idx], self.positive[idx], self.negative[idx]
