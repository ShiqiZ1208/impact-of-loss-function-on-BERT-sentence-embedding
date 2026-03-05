import torch
import os
import numpy as np
from transformers import logging

from Label_similarity import generate_distribution
from TrainerSE import TrainerSE
from lossfunc import get_loss
from datapreprocess import prepare_dataset
from transformers import AutoModel, AutoTokenizer
import argparse
import yaml

os.environ["ACCELERATE_DISABLE_PROGRESS_BAR"] = "true"
os.environ["DISABLE_TQDM"] = "1"

logging.set_verbosity_error()
logging.disable_progress_bar()
# check the device the code use
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description="TrainSE.py -b Batch_size -r total_runs -e epoches of each fine-tuning")
parser.add_argument("-b", "--batch_size", )
parser.add_argument("-r", "--total_runs", )
parser.add_argument("-e", "--total_epochs", )
parser.add_argument("-l", "--learning_rate", )
parser.add_argument("-g","--graph")
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
  lrate = 3e-5
else:
  lrate = float(args.learning_rate)
if args.graph == None:
  is_graph = False
else:
  is_graph = True

# use BERT-base-uncased model for the base model
model_id = 'bert-base-uncased'
#model_id = 'FacebookAI/roberta-base'
#model_id = 'mistralai/Mistral-7B-v0.1'
# The list of name of the datasets used in this experiments
sts_datasets = ['STS-B', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'SICK-R']
#sts_datasets = ["snli"]
# list of dict of loss name and loss keyword, loss type is not used
#file_path = 'losses.yaml'
#with open(file_path, 'r') as file:
  #data = yaml.safe_load(file)

#losses = data['losses']
losses = [
  {'loss_name': 'without_ft', 'loss_type': 'emb', 'loss_kwargs': {}},
  {'loss_name': 'cosine_similarity_mse_norm', 'loss_type': 'emb', 'loss_kwargs': {'norm': 'divided_by_maximum'}},
  {'loss_name': 'cosine_similarity_mse_norm_adj', 'loss_type': 'emb', 'loss_kwargs': {'norm': 'divided_by_maximum'}},
  {'loss_name': 'Batch_JS_div', 'loss_type': 'emb', 'loss_kwargs': {'norm': 'divided_by_maximum'}},
  {'loss_name': 'cosent_loss', 'loss_type': 'emb', 'loss_kwargs': {'tau': 20.0}},
  {'loss_name': 'in_batch_negative_loss', 'loss_type': 'emb', 'loss_kwargs': {'tau': 20.0}},
  {'loss_name': 'angle_loss', 'loss_type': 'emb', 'loss_kwargs': {'tau': 1.0}},
  {'loss_name': 'ibn_JSD', 'loss_type': 'emb', 'loss_kwargs': {'w_ibn': 1, 'w_JSD': 1, 'tau_ibn': 20.0}},
  {'loss_name': 'cosent_ibn_angle', 'loss_type': 'emb', 'loss_kwargs': {'w_cosent': 1, 'w_ibn': 1, 'w_angle': 1, 'tau_cosent': 20.0, 'tau_ibn': 20.0, 'tau_angle': 1.0}},
  #{'loss_name': 'triplet', 'loss_type': 'emb', 'loss_kwargs': {'margin': 1.0, 'minimum': 0.0, 'eps': 1e-6, 'distance': 'Eucliden'}},
  #{'loss_name': 'triplet', 'loss_type': 'emb', 'loss_kwargs': {'margin': 0.3, 'minimum': 0.0, 'eps': 1e-6, 'distance': 'cos_sim'}}
  #{'loss_name': 'IB','loss_type': 'emb','loss_kwargs': {'tau': 20.0}}
  ]

def get_model_tokenizer(model_id):
    '''
    get the base model and tokenizer for sentence embedding.
    input: model_id (string)
    output: model and tokenizer
    '''
    #get pretrain model on huggingface
    if model_id == 'mistralai/Mistral-7B-v0.1':
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    else:
        model = AutoModel.from_pretrained(model_id, output_attentions=True)
    #get the tokenizer for pretrain BERT
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    #load model to the device
    if model_id != 'mistralai/Mistral-7B-v0.1':
      model.to(device)
    return model, tokenizer


def run(total_run, b_size, epochs, lrate):
  total_runs = total_run
  batch_size = b_size
  lr = lrate
  spearman_list = []
  return_Batch_loss_list = []
  for loss in losses:
      loss_name = loss['loss_name']
      loss_type = loss['loss_type']
      loss_kwargs = loss['loss_kwargs']

      for dataset in sts_datasets:
          print(f'Running: {loss_name} on {dataset}')
          dataset_name = dataset
          total_spearman = 0.
          total_spearman_list = [0, 0, 0, 0, 0, 0, 0]
          total_mean = 0.
          total_kl = 0.
          all_lists = []
          average_list = []
          total_Batch_loss_list_run = []
          if loss_name == "triplet":
            train_dataset, test_datasets = prepare_dataset(dataset, 0.3, is_triplet = True)
          elif dataset not in["snli","nli", "multi_nli"]:
            train_dataset, test_dataset = prepare_dataset(dataset, 0.3)
          else:
            train_dataset, test_datasets = prepare_dataset(dataset, 0.3)
          for runs in range(total_runs):
              model, tokenizer = get_model_tokenizer(model_id)
              trainer = TrainerSE(model, device, tokenizer, model_id, mode='cls')
              model = trainer.train(train_dataset, loss_name, loss_kwargs, epochs, batch_size)
              
              if dataset_name not in ["snli","nli", "multi_nli","triplet"]:
                  spearman = trainer.evaluate(test_dataset, batch_size, dataset_name, loss_name)
                  print(f'run - {runs} dataset_name - {dataset_name} spearman - {spearman}')
                  total_spearman = total_spearman + spearman
              else:
                  spearmans = trainer.evaluate(test_datasets, batch_size, dataset_name, loss_name)
                  i = 0
                  for i, (item, spearman) in enumerate(zip(['STS-B', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'SICK-R'], spearmans)):
                    print(f'run - {runs} dataset_name - {item} spearman - {spearman}')
                    total_spearman_list[i] += spearman
                    #i = i + 1
          generate_distribution(model, tokenizer, dataset, loss_name, display_label = False, is_graph = False)
          if dataset_name not in ["nli", "snli", "multi_nli"]:
            spearman_list.append({'loss': loss_name, 'dataset': dataset, 'spearman': total_spearman / total_runs})
          else:
            for i in range(7):
                avg_spearman = total_spearman_list[i] / total_runs
            spearman_list.append({'loss': loss_name, 'dataset': dataset, 'spearman': avg_spearman})
  directory = "run_results"
  if not os.path.exists(directory):
    os.makedirs(directory)  # Create the directory (including parent dirs if needed)
    print(f"Created directory: {directory}")
  else:
    print(f"Directory already exists: {directory}")
  np.save('./run_results/bert_sts_results.npy', np.array(spearman_list, dtype=object))
  #if loss_name != 'triplet':
    #np.save('./run_results/batch_loss_different.npy', np.array(return_Batch_loss_list, dtype=object))

run(total_run, b_size, epochs, lrate)