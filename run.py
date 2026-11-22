import os
# must be set before CUDA/cuBLAS is first used, otherwise deterministic mode has no effect
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import random
import sys
import warnings
from datetime import datetime

import numpy as np
import torch
import datasets
import yaml
import huggingface_hub
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging

import Label_similarity
from TrainerSE import (TrainerSE, RunRecorder, build_probe_set, summarize,
                       load_or_compute_similarity, print_gain_by_bin)
from datapreprocess import prepare_train_dataset, prepare_eval_datasets, STS_train_test_split, prepare_SP_eval_datasets, prepare_CL_eval_datasets, prepare_CT_eval_datasets, prepare_RT_eval_datasets

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
    with open("config.yaml", "r", encoding="utf-8-sig") as file:   # utf-8-sig: tolerate a BOM
        data = yaml.safe_load(file)
    return data['training']
 
class Tee:
    """Sends prints to the terminal and to the log .txt file (progress bars stay out of the file)."""
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, s):
        self.stream.write(s)
        if not s.startswith('\r'):          # tqdm progress-bar updates start with '\r'
            self.file.write(s)
            self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)

def start_log(log_dir='./logs'):
    """Everything printed after this call is also written to log_dir/run_<time>.txt,
    starting with the full config.yaml of this run."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"run_{datetime.now():%Y%m%d_%H%M%S}.txt")
    log_file = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    print(f"Log file: {log_path}\nStarted:  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 30 + " config.yaml " + "=" * 30)
    with open("config.yaml", "r", encoding="utf-8-sig") as f:
        print(f.read())
    print("=" * 73 + "\n")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only: an op without a deterministic kernel warns instead of crashing the run
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['PYTHONHASHSEED'] = str(seed)
 
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
 
def get_model_tokenizer(model_id, pooling='mean'):
    '''
    get the base model and tokenizer for sentence embedding.
    input: model_id (string), pooling (string)
    output: model and tokenizer
    '''
    # attention weights are only needed for 'attention' pooling, and only the eager
    # implementation returns them; everything else uses the faster default (SDPA)
    if pooling.lower() == 'attention':
        model = AutoModel.from_pretrained(model_id, attn_implementation='eager')
    else:
        model = AutoModel.from_pretrained(model_id)
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
        runs_with_d = [run for run in all_runs if d in run[group]]   # MTEB exists only in run 0
        out[d] = {m: (float(np.mean([run[group][d][m] for run in runs_with_d])),
                      float(np.std([run[group][d][m] for run in runs_with_d])))
                  for m in metrics}
    return out
 
def print_aggregate(agg, title):
    print(f'\n===== {title}: mean ± std over runs =====')
    for d, metrics in agg.items():
        print(f' {d}: ', end='  ')
        for m, (mean, std) in metrics.items():
            print(f'{m}: {mean:.4f} ± {std:.4f}', end='  ')
        print()
 
def run(config, seeds, is_seed, verbose=False):
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
  eval_rt_datasets_name = config['test_rt_name']
  evaluation_metric = config['evaluation_metric']
  evaluation_sp_metric = config['evaluation_metric_sp']
  evaluation_cl_metric = config['evaluation_metric_cl']
  evaluation_ct_metric = config['evaluation_metric_ct']
  evaluation_rt_metric = config['evaluation_metric_rt']
  print(f'train on {train_dataset_name}\n test:{eval_sts_datasets_name}\n test:{eval_sp_datasets_name}\n test:{eval_cl_datasets_name}\n test:{eval_ct_datasets_name}\n metric:{evaluation_metric}')
  # graph folder for this model's settings: graphs/<evaluation>/<model>/<RUN_TAG>/...
  Label_similarity.RUN_TAG = f"{mode}_lr{lrate}_bs{batch_size}_ep{epochs}"
  output_dir = f'results/{model_id}'
  os.makedirs(output_dir, exist_ok=True)
  if config['graph'] == 1:
     is_graph = True
  else:
     is_graph = False
 
  spearman_list = []
  all_summaries = {}
  # name -> HF Dataset with sentence1, sentence2 and a score/labels column
  test_sets = prepare_eval_datasets(eval_sts_datasets_name)
  # where the mechanism-study runs are written; set mechanism_dir in config.yaml
  # (e.g. a Google Drive folder on Colab). Created if missing.
  OUT = config.get('mechanism_dir', './mechanism_runs')
  os.makedirs(OUT, exist_ok=True)
  tag = f"{model_id.replace('/', '_')}_lr{lrate}"   # one folder/file per (model, lr), so a sweep doesn't overwrite
  sims = {}          # train dataset name -> similarity of each test pair to its nearest training pair
  run_dirs = {}      # (loss_name, train dataset) -> run folders, for the gain-by-bin table at the end
  for loss in config['losses']:
      loss_name = loss['loss_name']
      loss_kwargs = loss['loss_kwargs']

      for dataset in train_dataset_name:
          print(f'Fine-tuning model: {model_id} using: {loss_name} on {dataset}')

          # split seed is fixed (42) so every loss/run sees the same train/test partition
          train_dataset, val_dataset = prepare_train_dataset(dataset, seed = 42)
          # in-domain: STS13-16 / SICK-R are trained on their 70% split, so they must be
          # tested on the held-out 30% (same seed), not on the full set with the training pairs
          # (STS12 uses its official train/test split, so its full test set is already correct)
          test_datasets = dict(test_sets)
          if dataset in test_datasets and dataset in ('STS13', 'STS14', 'STS15', 'STS16', 'SICK-R'):
              test_datasets[dataset] = STS_train_test_split(dataset, split=0.3, seed=42)['test']
          probe = build_probe_set(test_datasets, n_per_set=200)   # same probe for every run of this training set
          if dataset not in sims:   # computed once per training set, then cached on disk
              sims[dataset] = load_or_compute_similarity(
                  f"{OUT}/pair_similarity_{dataset}.npy", train_dataset, test_datasets)

          sp_datasets= prepare_SP_eval_datasets(eval_sp_datasets_name)
          sp_val_datasets  = {n: val  for n, (val, test) in sp_datasets.items()}
          sp_test_datasets = {n: test for n, (val, test) in sp_datasets.items()}
 
          cl_datasets = prepare_CL_eval_datasets(eval_cl_datasets_name)
          cl_val_datasets  = {n: val  for n, (val, test) in cl_datasets.items()}
          cl_test_datasets = {n: test for n, (val, test) in cl_datasets.items()}
 
          ct_test_datasets = prepare_CT_eval_datasets(eval_ct_datasets_name)
          rt_test_datasets = prepare_RT_eval_datasets(eval_rt_datasets_name)
          all_runs = []
          if loss_name == 'without_ft':
              n_run = 1
          else:
              n_run = config['total_runs']
          for runs in range(n_run):
            if is_seed == True:
              set_seed(seeds[runs])
            print(f"runs: {runs}\n")
            model, tokenizer = get_model_tokenizer(model_id, mode)
            trainer = TrainerSE(model, device, tokenizer, model_id, loss_name, dataset, evaluation_metric, evaluation_sp_metric, evaluation_cl_metric, evaluation_ct_metric, evaluation_rt_metric, lrate, mode, is_seed, is_graph)

            run_train = train_dataset
            if loss_name == 'mean_adjust_MSE':
                    # rescale a per-run copy, so each run starts from the original labels
                    mean, variance = trainer.cal_mean_variance(train_dataset, batch_size, seed=seeds[runs])
                    run_train, o_mu, o_var = rescale_dataset_labels(train_dataset, mean, variance)
                    print(f"prediction:{mean:.4f}, {variance:.4f}")
            rec = None
            if loss_name != 'without_ft':
                run_dir = f"{OUT}/{tag}/{dataset}/{loss_name}/run{runs}"
                rec = RunRecorder(run_dir, trainer.model, trainer.tokenizer, probe, train_dataset,
                                  sims=sims[dataset], pooling=mode)
                run_dirs.setdefault((loss_name, dataset), []).append(run_dir)
            # is_eval (config.yaml): keep the epoch with the best validation Spearman instead of the
            # last one; only possible for training sets that have a validation split (STS-B, SemRel)
            use_val = bool(config.get('is_eval', False)) and len(val_dataset) > 0
            if config.get('is_eval', False) and not use_val and runs == 0:
                print(f'is_eval: {dataset} has no validation split, using the final checkpoint')
            # the trainer calls rec.start() before the first step and rec.log_step() after each step
            model, loss_log = trainer.train(run_train, val_dataset, loss_kwargs, epochs,
                                            batch_size, seed=seeds[runs], is_eval=use_val,
                                            recorder=rec)
            if runs == 0 and is_graph and rec is not None and rec.epoch_log:   # validation score per epoch
                Label_similarity.plot_epoch_curve(rec.epoch_log, model_id, mode, loss_name, dataset)
            if rec is not None:
                final = rec.finish(test_datasets, save_checkpoint=(runs == 0))
                summarize(final)
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
            RT_result = trainer.evaluate_all_rt(rt_test_datasets)
            for i, name in enumerate(rt_test_datasets):
                print(f' evaluate on dataset {name}: ', end='  ')
                for metric in evaluation_rt_metric:
                    print(f'{metric}: {RT_result[i][metric]:.4f}', end='  ')
                print()
            MTEB_result = {}
            if config.get('test_mteb_tasks') and runs == 0:   # MTEB is slow: only on the first run
                MTEB_result = trainer.evaluate_mteb(config['test_mteb_tasks'])
                for name, score in MTEB_result.items():
                    print(f' MTEB {name}: main_score: {score:.4f}')

            all_runs.append({
                'STS': to_named(STS_result, list(test_datasets), evaluation_metric),
                'SP':  to_named(SP_result, list(sp_test_datasets), evaluation_sp_metric,
                                skip=('f1_threshold', 'ac_threshold')),
                'CL':  to_named(CL_result, list(cl_test_datasets), evaluation_cl_metric),
                'CT':  to_named(CT_result, list(ct_test_datasets), evaluation_ct_metric),
                'RT':  to_named(RT_result, list(rt_test_datasets), evaluation_rt_metric),
                'MTEB': {name: {'main_score': s} for name, s in MTEB_result.items()},
            })

          # summary per (loss, training set) -- inside the dataset loop so no set is dropped
          key = f'{loss_name}|{dataset}'
          all_summaries[key] = {}
          print(f'\n########## {loss_name} on {dataset}: {len(all_runs)} runs ##########')
          for group in ('STS', 'SP', 'CL', 'CT', 'RT', 'MTEB'):
              agg = aggregate_runs(all_runs, group)
              print_aggregate(agg, group)
              all_summaries[key][group] = agg

          np.save(f'{output_dir}/{tag}_{loss_name}_{dataset}_runs.npy',
                      np.array(all_runs, dtype=object))   # raw per-run numbers

  # gain of Mean-Adjust over MSE-norm in each similarity bin (needs both losses in config).
  # Runs are paired by index; with use_seed: true both losses use the same seed per index.
  a, b = 'cosine_similarity_mse_norm', 'mean_adjust_MSE'
  for dataset in train_dataset_name:
      ra, rb = run_dirs.get((a, dataset)), run_dirs.get((b, dataset))
      if ra and rb and len(ra) == len(rb):
          print(f'\n(trained on {dataset})')
          print_gain_by_bin(sims[dataset], ra, rb, 'MSE-norm', 'Mean-Adjust')
 
  out_dir = './run_results'
  os.makedirs(out_dir, exist_ok=True)
  # one summary file per model (mean ± std for every loss), so models don't overwrite each other
  np.save(f'{out_dir}/{tag}_sts_results.npy', np.array(all_summaries, dtype=object))
  return all_summaries


def expand_models(config):
    '''
    model_id in config.yaml may be one model or a list of models. Each list entry is either
    a model name, or a dict with model_id plus any training settings to override for that
    model only (e.g. pooling, batch_size, learning_rate). Returns one config per model.
    '''
    entries = config['model_id']
    if not isinstance(entries, list):
        entries = [entries]
    configs = []
    for entry in entries:
        overrides = dict(entry) if isinstance(entry, dict) else {'model_id': entry}
        if 'model_id' not in overrides:
            raise ValueError(f"model entry {entry} has no model_id")
        configs.append({**config, **overrides})
    return configs


if __name__ == "__main__":
    config = load_training_config()
    start_log(config.get('log_dir', './logs'))     # terminal output + config.yaml -> logs/run_<time>.txt
    seeds = generate_seeds(config['total_runs'])   # same seeds for every model
    #seeds = [3163119785, 1812140441, 127978094, 939042955, 2340505846]
    print(f"Seeds: {seeds}")
    # use_seed: true makes run i of every loss use the same seed (needed for a paired comparison)
    is_seed = bool(config.get('use_seed', False))

    model_configs = expand_models(config)
    print(f"Models to run: {[c['model_id'] for c in model_configs]}")
    all_models = {}
    for i, model_config in enumerate(model_configs):
        model_id = model_config['model_id']
        print(f"\n==================== model {i + 1}/{len(model_configs)}: {model_id} "
              f"(pooling={model_config['pooling']}, batch_size={model_config['batch_size']}, "
              f"lr={model_config['learning_rate']}) ====================")
        try:
            all_models[f"{model_id}_lr{model_config['learning_rate']}"] = run(model_config, seeds, is_seed=is_seed)
        except Exception as e:
            # one failing model (e.g. out of memory, gated model) does not stop the others
            print(f"!! model {model_id} failed: {type(e).__name__}: {e}")
            if not config.get('continue_on_error', True):
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()   # release GPU memory before the next model

    os.makedirs('./run_results', exist_ok=True)
    np.save('./run_results/all_models_sts_results.npy', np.array(all_models, dtype=object))
    print(f"\nFinished {len(all_models)}/{len(model_configs)} models: {list(all_models)}")