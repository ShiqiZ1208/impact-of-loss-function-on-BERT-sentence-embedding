import torch
import os
import numpy as np
from transformers import logging
import random
from TrainerSE import TrainerSE
from datapreprocess import prepare_train_dataset, prepare_eval_datasets, prepare_SP_eval_datasets, prepare_CL_eval_datasets, prepare_CT_eval_datasets
from transformers import AutoModel, AutoTokenizer
import yaml
import torch, datasets
from transformers.utils import logging
import huggingface_hub
import warnings

os.environ["ACCELERATE_DISABLE_PROGRESS_BAR"] = "true"
os.environ["DISABLE_TQDM"] = "1"
datasets.disable_progress_bars()
datasets.logging.set_verbosity_error()
huggingface_hub.utils.logging.set_verbosity_error()
logging.set_verbosity_error()
logging.disable_progress_bar()
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", message="n_jobs value")
warnings.filterwarnings("ignore", message="Spectral initialisation failed")
warnings.filterwarnings("error", category=RuntimeWarning)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_training_config():
    """
    Reads training_config.yaml and returns the training hyperparameters
    as a dict (model_name, pooling_strategy, batch_size, learning_rate,
    num_epochs).
    """
    with open("config.yaml", "r") as file:
        data = yaml.safe_load(file)
    return data['training']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def mean_loss_across_runs(all_loss_logs):
    lengths = [len(log) for log in all_loss_logs]
    if len(set(lengths)) != 1:
        min_len = min(lengths)
        print(f"Warning: unequal lengths {lengths} -- truncating all to {min_len} steps")
        all_loss_logs = [log[:min_len] for log in all_loss_logs]
    loss_array = np.array(all_loss_logs)
    return loss_array.mean(axis=0), loss_array.std(axis=0)

def generate_seeds(total_runs, base_seed=42):
    rng = random.Random(base_seed)
    return [rng.randint(0, 2**32 - 1) for _ in range(total_runs)]

def get_model_tokenizer(model_id):
    '''
    get the base model and tokenizer for sentence embedding.
    input: model_id (string)
    output: model and tokenizer
    '''
    model = AutoModel.from_pretrained(model_id, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    #load model to the device
    model.to(device)

    return model, tokenizer

def rescale_dataset_labels(train_dataset, target_mean, target_variance, eps=1e-8):
    labels = torch.tensor(train_dataset['labels'], dtype=torch.float32)
    
    target_std = target_variance ** 0.5
    labels_mean = labels.mean()
    labels_std = labels.std()
    labels_var = labels.var()
    
    z = (labels - labels_mean) / (labels_std + eps)
    scaled_labels = z * target_std + target_mean
    
    scaled_labels_list = scaled_labels.tolist()
    train_dataset = train_dataset.map(
        lambda example, idx: {'labels': scaled_labels_list[idx]},
        with_indices=True
    )
    original_mean = labels_mean.item()
    original_variance = labels_var.item()
    return train_dataset, original_mean, original_variance

def to_named(results, names, metrics, skip=()):
    named = {n: {m: r[m] for m in metrics if m not in skip} for n, r in zip(names, results)}
    kept = [m for m in metrics if m not in skip]
    if names:
        named['Average'] = {m: float(np.mean([named[n][m] for n in names])) for m in kept}
    return named

def aggregate_runs(all_runs, group):
    datasets = all_runs[0][group].keys()
    out = {}
    for d in datasets:
        metrics = all_runs[0][group][d].keys()
        out[d] = {m: (float(np.mean([run[group][d][m] for run in all_runs])),
                      float(np.std([run[group][d][m] for run in all_runs])))
                  for m in metrics}
    return out

def print_aggregate(agg, title):
    print(f'\n===== {title}: mean ± std over runs =====')
    for d, metrics in agg.items():
        print(f' {d}: ', end='  ')
        for m, (mean, std) in metrics.items():
            print(f'{m}: {mean:.4f} ± {std:.4f}', end='  ')
        print()

def run(config, seeds, is_seed):
  lrate = config['learning_rate']
  batch_size = config['batch_size']
  epochs = config['num_epochs']
  mode = config['pooling']
  model_id = config['model_id']
  train_dataset_name = config['train_name']
  eval_sts_datasets_name = config['test_sts_name']
  eval_sp_datasets_name = config['test_sp_name']
  eval_cl_datasets_name = config['test_cl_name']
  eval_ct_datasets_name = config['test_ct_name']
  evaluation_metric = config['evaluation_metric']
  evaluation_sp_metric = config['evaluation_metric_sp']
  evaluation_cl_metric = config['evaluation_metric_cl']
  evaluation_ct_metric = config['evaluation_metric_ct']
  print(f'train on {train_dataset_name}\n test:{eval_sts_datasets_name}\n test:{eval_sp_datasets_name}\n test:{eval_cl_datasets_name}\n test:{eval_ct_datasets_name}\n metric:{evaluation_metric}')
  output_dir = f'results/{model_id}'
  os.makedirs(output_dir, exist_ok=True)
  if config['graph'] == 1:
     is_graph = True
  else:
     is_graph = False

  spearman_list = []
  all_summaries = {} 
  for loss in config['losses']:
      loss_name = loss['loss_name']
      loss_kwargs = loss['loss_kwargs']
      
      for dataset in train_dataset_name:
          print(f'Fine-tuning model: {model_id} using: {loss_name} on {dataset}')

          train_dataset, val_dataset = prepare_train_dataset(dataset, seed = seeds)
          test_datasets = prepare_eval_datasets(eval_sts_datasets_name)

          sp_datasets= prepare_SP_eval_datasets(eval_sp_datasets_name)
          sp_val_datasets  = {n: val  for n, (val, test) in sp_datasets.items()}
          sp_test_datasets = {n: test for n, (val, test) in sp_datasets.items()}

          cl_datasets = prepare_CL_eval_datasets(eval_cl_datasets_name)
          cl_val_datasets  = {n: val  for n, (val, test) in cl_datasets.items()}
          cl_test_datasets = {n: test for n, (val, test) in cl_datasets.items()}

          ct_test_datasets = prepare_CT_eval_datasets(eval_ct_datasets_name)
          all_runs = [] 
          for runs in range(config['total_runs']):
            if is_seed == True:
              set_seed(seeds[runs])
            model, tokenizer = get_model_tokenizer(model_id)
            trainer = TrainerSE(model, device, tokenizer, model_id, loss_name, dataset, evaluation_metric, evaluation_sp_metric, evaluation_cl_metric, evaluation_ct_metric, lrate, mode, is_seed, is_graph)
                  
            if loss_name == 'mean_adjust_MSE':
                    mean, variance = trainer.cal_mean_variance(train_dataset, batch_size, seed=seeds[runs])
                    train_dataset, o_mu, o_var = rescale_dataset_labels(train_dataset, mean, variance)
            model, loss_log = trainer.train(train_dataset, val_dataset, loss_kwargs, epochs, batch_size, seed=seeds[runs], is_eval =True)
            STS_result = trainer.evaluate_all_sts(test_datasets)
            for i, name in enumerate(test_datasets):
                print(f' evaluate on dataset {name}: ', end='  ')
                for metric in evaluation_metric:
                    print(f'{metric}: {STS_result[i][metric]:.4f}', end='  ')
                print()
            SP_result = trainer.evaluate_all_sp(sp_val_datasets, sp_test_datasets)
            for i, name in enumerate(sp_test_datasets):
                print(f' evaluate on dataset {name}: ', end='  ')
                for metric in evaluation_sp_metric:
                    print(f'{metric}: {SP_result[i][metric]:.4f}', end='  ')
                print()
            CL_result = trainer.evaluate_all_cl(cl_val_datasets, cl_test_datasets)
            for i, name in enumerate(cl_test_datasets):
                print(f' evaluate on dataset {name}: ', end='  ')
                for metric in evaluation_cl_metric:
                    print(f'{metric}: {CL_result[i][metric]:.4f}', end='  ')
                print()
            CT_result = trainer.evaluate_all_ct(ct_test_datasets)
            for i, name in enumerate(ct_test_datasets):
                print(f' evaluate on dataset {name}: ', end='  ')
                for metric in evaluation_ct_metric:
                    print(f'{metric}: {CT_result[i][metric]:.4f}', end='  ')
                print()

            all_runs.append({
                'STS': to_named(STS_result, list(test_datasets), evaluation_metric),
                'SP':  to_named(SP_result, list(sp_test_datasets), evaluation_sp_metric,
                                skip=('f1_threshold', 'ac_threshold')),
                'CL':  to_named(CL_result, list(cl_test_datasets), evaluation_cl_metric),
                'CT':  to_named(CT_result, list(ct_test_datasets), evaluation_ct_metric),
            })
  
      key = f'{loss_name}|{dataset}'
      all_summaries[key] = {}
      print(f'\n########## {loss_name} on {dataset}: {len(all_runs)} runs ##########')
      for group in ('STS', 'SP', 'CL', 'CT'):
          agg = aggregate_runs(all_runs, group)
          print_aggregate(agg, group)
          all_summaries[key][group] = agg

      np.save(f'{output_dir}/{model_id.replace("/", "_")}_{loss_name}_{dataset}_runs.npy',
                  np.array(all_runs, dtype=object))   # raw per-run numbers
  out_dir = './run_results'
  os.makedirs(out_dir, exist_ok=True)
  np.save('./run_results/bert_sts_results.npy', np.array(all_summaries, dtype=object))   # mean ± std for every loss

if __name__ == "__main__":
    config = load_training_config()
    seeds = generate_seeds(config['total_runs'])
    #seeds = [3163119785, 1812140441, 127978094, 939042955, 2340505846]
    print(f"Seeds: {seeds}")
    run(config, seeds, is_seed = False)

