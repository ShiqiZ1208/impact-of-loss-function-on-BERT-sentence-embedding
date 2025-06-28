import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate
from torch.optim import AdamW
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from scipy.stats import spearmanr
import argparse
from prepare_STS import get_sts_dataset, STSDataset
from Loss_Functions import cosine_similarity_mse_loss, cosine_similarity_mse_norm, cosent_loss, in_batch_negative_loss, angle_loss, cosent_ibn_angle

# check the device the code use
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# set up some global variable for the program

parser = argparse.ArgumentParser(description="TrainSE.py -b Batch_size -r total_runs -e epoches of each fine-tuning")
parser.add_argument("-b", "--batch_size", )
parser.add_argument("-r", "--total_runs", )
parser.add_argument("-e", "--total_epochs", )
parser.add_argument("-l", "--learning_rate", )
parser.add_argument("-mode","--modes")
args = parser.parse_args()
#b_size = int(args.batch_size)
#total_run  = int(args.total_runs)
#epochs = int(args.total_epochs)
if args.batch_size == None:
  print("use default batch_size of 60")
  b_size = 60
else:
  b_size = int(args.batch_size)
if args.total_runs == None:
  print("use default total run of 3")
  total_run = 3
else:
  total_run  = int(args.total_runs)
if args.total_epochs == None:
  epochs = 10
else:
  epochs = int(args.total_epochs)
if args.learning_rate == None:
  lrate = 5e-5
else:
  lrate = float(args.learning_rate)
if args.modes == None:
  modes = "default"
elif args.modes.lower() == "eval":
  modes = "eval"
else:
  print("no such modes")

# use BERT-base-uncased model for the base model
model_id = 'bert-base-uncased'
# The list of name of the datasets used in this experiments
sts_datasets = ['STS-B', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'SICK-R']
# list of dict of loss name and loss keyword, loss type is not used
losses = [
    #{'loss_name':'without_ft', 'loss_type': 'emb', 'loss_kwargs': {}},
    {'loss_name': 'cosine_similarity_mse_loss', 'loss_type': 'emb', 'loss_kwargs': {}},
    {'loss_name': 'cosine_similarity_mse_norm', 'loss_type': 'emb', 'loss_kwargs': {'norm': 'divided_by_maximum'}},
    {'loss_name': 'cosent_loss', 'loss_type': 'emb', 'loss_kwargs': {'tau': 20.0}},
    {'loss_name': 'in_batch_negative_loss', 'loss_type': 'emb', 'loss_kwargs': {'tau': 20.0}},
    {'loss_name': 'angle_loss', 'loss_type': 'emb', 'loss_kwargs': {'tau': 1.0}},
    {'loss_name': 'cosent_ibn_angle', 'loss_type': 'emb', 'loss_kwargs': {'w_cosent': 1, 'w_ibn': 1, 'w_angle': 1, 'tau_cosent': 20.0, 'tau_ibn': 20.0, 'tau_angle': 1.0}}
]
print(modes)
def get_model_tokenizer(model_id):
    '''
    get the base model and tokenizer for sentence embedding.
    input: model_id (string)
    output: model and tokenizer
    '''
    #get pretrain model on huggingface
    model = AutoModel.from_pretrained(model_id)
    #get the tokenizer for pretrain BERT
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    #load model to the device
    model.to(device)
    return model, tokenizer

def extract_embeddings(model, tokenizer, device, sentences, to_numpy=False):
    '''
    extract the embedding layers from the model, The embedding is the CLS token on the right.
    input: model, tokenizer, device, sentences (string)
    output: the embedding score of sentences
    '''
    encodings = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(device)
    # get the CLS tokens in the last hidden layer of the model
    embeddings = model(**encodings).last_hidden_state[:, 0, :]

    #detach and covert from tensor to numpy if needed
    if to_numpy:
        embeddings = embeddings.cpu().detach().numpy()
    return embeddings

def train(model, tokenizer, dataset, val_dataset, batch_size, epochs=10, lrate = 5e-5, loss_name='cosine_similarity_mse_loss', **loss_kwargs):
    # use optimizer AdamW with learning rate 5e-5
    optimizer = AdamW(model.parameters(), lrate)

    loss_list = []
    # start the training loop
    num_epochs = epochs
    for epoch in range(num_epochs):
        #print(epoch)
        #load the data points from dataloader
        model.train()
        data_loader = DataLoader(STSDataset(dataset['sentence1'], dataset['sentence2'], dataset['labels']), batch_size=batch_size, shuffle=True)
        #l_list=[]
        for sentence1_texts, sentence2_texts, labels in tqdm(data_loader, desc="Training", leave=False):
            labels = labels.to(device)

            # extract the CLS token embedding
            sentence1_embeddings = extract_embeddings(model, tokenizer, device, sentence1_texts)
            sentence2_embeddings = extract_embeddings(model, tokenizer, device, sentence2_texts)

            # compute the embedding loss
            loss = globals()[loss_name](sentence1_embeddings, sentence2_embeddings, labels, **loss_kwargs)
            if abs(loss) < 1e-8:
                continue
            #l_list.append(loss)
            # Backpropagation
            loss.backward()

            # Updating
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        val_list=[]
        val_data_loader = DataLoader(STSDataset(val_dataset['sentence1'], val_dataset['sentence2'], val_dataset['labels']), batch_size=batch_size, shuffle=True)
        with torch.no_grad():
            for sentence1_texts, sentence2_texts, labels in tqdm(val_data_loader, desc="validation", leave=False):
                labels = labels.to(device)

                sentence1_embeddings = extract_embeddings(model, tokenizer, device, sentence1_texts)
                sentence2_embeddings = extract_embeddings(model, tokenizer, device, sentence2_texts)

                # compute the embedding loss
                loss = globals()[loss_name](sentence1_embeddings, sentence2_embeddings, labels, **loss_kwargs)
                if abs(loss) < 1e-8:
                    continue
                val_list.append(loss)
        average = sum(val_list)/len(val_list)
        loss_list.append(average)
        
    return model, loss_list

def calculate_cosine_similarity(embeddings_1, embeddings_2):
    # use torch.nn calculate the cosine_similarity from two embeddings
    cosine_similarity = F.cosine_similarity(embeddings_1, embeddings_2, dim=1)
    return cosine_similarity

def calculate_Spearman_rank_correlation_coefficient(scores, scores_actual):
    #calculate the spearman correlation from the predice score and actual score
    sc, _ = spearmanr(scores, scores_actual)
    return sc

def evaluate_sts(model, tokenizer, test_dataset, batch_size):
    # set mode to evaluation for evaluate
    model.eval()

    # load the testset from dataloader
    test_dataset = STSDataset(test_dataset['sentence1'], test_dataset['sentence2'], test_dataset['labels'])
    data_loader = DataLoader(test_dataset, batch_size=batch_size)
    all_embeddings1 = []
    all_embeddings2 = []
    all_labels = []

    with torch.no_grad():
        for sentences1, sentences2, labels in tqdm(data_loader, desc="Extracting embeddings", leave=False):
            #for every pair extract the embedding and labels append to the empty list created
            embeddings1 = extract_embeddings(model, tokenizer, device, sentences1)
            embeddings2 = extract_embeddings(model, tokenizer, device, sentences2)
            all_embeddings1.append(embeddings1.cpu())
            all_embeddings2.append(embeddings2.cpu())
            all_labels.append(labels.cpu())

    # convert the list of tensor to single tensor
    # list([tensor([1,2,3]),tensor([4,5,6])])
    # covert to 
    # tensor([1, 2, 3],[4, 5, 6])
    data_embeddings1 = torch.cat(all_embeddings1)
    data_embeddings2 = torch.cat(all_embeddings2)
    data_labels = torch.cat(all_labels)
    data_labels_np = data_labels.numpy()

    # calculate the cosine_similarity score and spearman score then return the spearman
    cosine_similarities = calculate_cosine_similarity(data_embeddings1, data_embeddings2)
    spearman = calculate_Spearman_rank_correlation_coefficient(cosine_similarities, data_labels_np)
    return spearman

def run(total_run, b_size, epochs, lrate, modes="defualt"):
  total_runs = total_run
  batch_size = b_size
  lr = lrate
  spearman_list = []
  for loss in losses:
      loss_name = loss['loss_name']
      loss_type = loss['loss_type']
      loss_kwargs = loss['loss_kwargs']

      for dataset in sts_datasets:
          print(f'Running: {loss_name} on {dataset}')
          total_spearman = 0.
          all_lists = []
          average_list = []
          for loop_count in range(0, total_runs):
              # Dataset Preparation...
              if modes == "defualt":
                train_dataset, test_dataset, val_dataset = get_sts_dataset(dataset)
              elif modes == "eval":
                train_dataset, test_dataset = get_sts_dataset(dataset, 0.3, modes)
              #print(len(train_dataset),len(test_dataset))
              # Model Preparation...
              model, tokenizer = get_model_tokenizer(model_id)

              # Training Loop...
              if loss_name != 'without_ft':
                  if modes == "defualt":
                    model, val_list = train(model, tokenizer, train_dataset, val_dataset, batch_size, epochs, lr, loss_name=loss_name, **loss_kwargs)
                  elif modes == "eval":
                    model, val_list = train(model, tokenizer, train_dataset, test_dataset, batch_size, epochs, lr, loss_name=loss_name, **loss_kwargs)
                  all_lists.append(val_list)
                  #print(val_list)
              # Evaluation loop...
              spearman = evaluate_sts(model, tokenizer, test_dataset, batch_size)
              print(f'Loop {loop_count} spearman - {spearman}')
              total_spearman += spearman
          for elements in zip(*all_lists):
            avg = sum(elements) / len(elements)
            average_list.append(avg)
            
          print(average_list)
          min_value = min(average_list)
          min_index = average_list.index(min_value)
          print(f"suggest to stop at {min_index} epochs with best performance.\n")
          spearman_list.append({'loss': loss_name, 'dataset': dataset, 'spearman': total_spearman / total_runs, 'loss_list': average_list})
  np.save('bert_sts_results.npy', np.array(spearman_list, dtype=object))

run(total_run, b_size, epochs, lrate, modes)