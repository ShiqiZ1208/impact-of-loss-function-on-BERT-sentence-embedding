from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import re
import time
import warnings
import numpy as np
from tqdm import tqdm
from lossfunc import get_loss, TRACE
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr
from torch.optim import AdamW
from datapreprocess import STSDataset, TripDataset, CLDataset, RTDataset
from IsoScore.IsoScore import IsoScore
from Label_similarity import generate_random_pair_distribution, generate_distribution, plot_clusters, plot_confusion, plot_SP_threshold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import v_measure_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.feature_extraction.text import TfidfVectorizer

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class TrainerSE:
    def __init__(self, model, device, tokenizer, model_id, loss_n, dataset_n, evaluate_metric, evaluation_sp_metric, evaluation_cl_metric, evaluation_ct_metric, evaluation_rt_metric, lrate = 5e-5, pooling = "cls", is_seed = False, is_graph = False):
        #print(lrate)
        self.is_seed = is_seed
        self.model = model
        self.optimizer = AdamW(model.parameters(), lrate)
        self.device = device
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.is_llama = "llama" in model_id.lower()
        self.loss_name = loss_n
        self.train_dataset_name = dataset_n
        self.pooling = pooling
        self.evaluate_metric = evaluate_metric
        self.evaluate_sp_metric = evaluation_sp_metric
        self.evaluate_cl_metric = evaluation_cl_metric
        self.evaluate_ct_metric = evaluation_ct_metric
        self.evaluate_rt_metric = evaluation_rt_metric
        self.is_graph = is_graph




    def extract_embeddings(self, model, tokenizer, sentences, device):
        if self.is_llama:
          tokenizer.pad_token = tokenizer.eos_token
          model.config.pad_token_id = tokenizer.pad_token_id
        encodings = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(device)
        #output = model(**encodings, output_attentions=True, output_hidden_states=True, return_dict=True)
        need_attentions = self.pooling.lower() == 'attention'
        need_hidden_states = self.pooling.lower() == 'mid_mean'
        output = model(**encodings, output_attentions=need_attentions, output_hidden_states=need_hidden_states, return_dict=True)
        if self.pooling.lower() =="cls":
          embeddings = output.last_hidden_state[:, 0, :]
        elif self.pooling.lower() == "mean":
          token_embeddings = output.last_hidden_state
          attention_mask = encodings['attention_mask'].unsqueeze(-1)
          embeddings = (token_embeddings * attention_mask).sum(1) / attention_mask.sum(1)
        elif self.pooling.lower() == "max":
          token_embeddings = output.last_hidden_state
          attention_mask = encodings['attention_mask'].unsqueeze(-1).expand(token_embeddings.size())
          token_embeddings = token_embeddings.masked_fill(attention_mask == 0, -1e9)
          embeddings = token_embeddings.max(1).values
        elif self.pooling.lower() == "attention":
          token_embeddings = output.last_hidden_state
          attention_mask = encodings['attention_mask'].unsqueeze(-1)
          last_attention = output.attentions[-1]
          cls_attention = last_attention[:, :, 0, :]
          weights = cls_attention.mean(dim=1)
          weights = weights * attention_mask.squeeze(-1)
          weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
          embeddings = (token_embeddings * weights.unsqueeze(-1)).sum(dim=1)
        elif self.pooling.lower() == "mid_mean":
          hidden_states = output.hidden_states          # tuple: (embeddings, layer_1, ..., layer_N)
          num_layers = len(hidden_states) - 1            # exclude embedding layer at index 0
          layer_idx = getattr(self, 'layer_idx', num_layers // 2)  # defaults to middle transformer layer
          token_embeddings = hidden_states[layer_idx]
          attention_mask = encodings['attention_mask'].unsqueeze(-1)
          embeddings = (token_embeddings * attention_mask).sum(1) / attention_mask.sum(1)

        return embeddings.float()

    def cal_mean_variance(self, train_dataset, batch_size, seed = 42, if_max = False):
        if self.train_dataset_name == "snli":
          if_max = True
          print("max 1000 sample")
        sample = 0
        g = torch.Generator()
        g.manual_seed(seed)
        self.model.eval()
        data_loader = DataLoader(STSDataset(train_dataset['sentence1'], train_dataset['sentence2'], train_dataset['labels']), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        all_embeddings1 = []
        all_embeddings2 = []

        with torch.no_grad():
            for sentences1, sentences2, labels in tqdm(data_loader, desc="calculating", leave=False):
                #for every pair extract the embedding and labels append to the empty list created
                embeddings1 = self.extract_embeddings(self.model, self.tokenizer, sentences1, self.device)
                embeddings2 = self.extract_embeddings(self.model, self.tokenizer, sentences2, self.device)
                all_embeddings1.append(embeddings1.cpu())
                all_embeddings2.append(embeddings2.cpu())
                sample = sample + 1
                if if_max:
                  if batch_size * sample > 1000:
                    break

        data_embeddings1 = torch.cat(all_embeddings1)
        data_embeddings2 = torch.cat(all_embeddings2)

        cosine_similarities = self.calculate_cosine_similarity(data_embeddings1, data_embeddings2)
        mean = cosine_similarities.mean().item()
        variance = cosine_similarities.var().item()

        return mean, variance


    def train_triplet(self, train_dataset, val_dataset, loss_kwargs, num_epochs, batch_size, seed = 42):
        loss_function = get_loss(self.loss_name, **loss_kwargs)
        if self.is_seed == True:
            g = torch.Generator()
            g.manual_seed(seed)
        else:
            g = None
        loss_log = []
        for epoch in range(num_epochs):
            self.model.train()
            data_loader = DataLoader(TripDataset(train_dataset['anchor'], train_dataset['positive'], train_dataset['negative']), batch_size=batch_size, shuffle=True , worker_init_fn=seed_worker, generator=g)
            for sentence1_texts, sentence2_texts, sentence3_texts in tqdm(data_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False):

                self.optimizer.zero_grad()

                anchor = self.extract_embeddings(self.model, self.tokenizer, sentence1_texts, self.device)
                positive = self.extract_embeddings(self.model, self.tokenizer, sentence2_texts, self.device)
                negative = self.extract_embeddings(self.model, self.tokenizer, sentence3_texts, self.device)
                
                loss = loss_function(anchor, positive, negative)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

        return self.model, loss_log

    def train_base(self, train_dataset, val_dataset, loss_kwargs, num_epochs, batch_size, seed=42, is_eval = False, recorder=None):
        loss_log = []
        best_spearman = float('-inf')
        loss_function = get_loss(self.loss_name, **loss_kwargs)
        sts_name = self.train_dataset_name
        best_model_state = None
        if self.is_seed == True:
            g = torch.Generator()
            g.manual_seed(seed)
        else:
            g = None 
        i = 0
        if recorder is not None:                           # <- before the first epoch
            recorder.model = self.model                    #    the model actually being trained
            recorder.start()
        global_step = 0
        for epoch in range(num_epochs):
            self.model.train()
            data_loader = DataLoader(STSDataset(train_dataset['sentence1'], train_dataset['sentence2'], train_dataset['labels']), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
            for sentence1_texts, sentence2_texts, labels in tqdm(data_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False):
              
                self.optimizer.zero_grad()
                labels = labels.to(self.device)

                sentence1_embeddings = self.extract_embeddings(self.model, self.tokenizer, sentence1_texts, self.device)
                sentence2_embeddings = self.extract_embeddings(self.model, self.tokenizer, sentence2_texts, self.device)
                
                loss = loss_function(sentence1_embeddings, sentence2_embeddings, labels)
                loss_log.append(loss.item())

                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                global_step += 1
                if recorder is not None:                   
                  recorder.log_step(global_step)
            if is_eval == True:
                results = self.evaluate_sts(val_dataset, sts_name, graph = False)
                spearman = results[0]
                if recorder is not None:
                  recorder.log_epoch(epoch, {'val_spearman': spearman})
                if spearman > best_spearman:
                  best_spearman = spearman
                  best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        if is_eval == True:
            self.model.load_state_dict(best_model_state)
        return self.model, loss_log


    def train(self, train_dataset, val_dataset, loss_kwargs, num_epochs, batch_size, seed = 42, is_eval = False, recorder = None):
        if self.loss_name == "triplet":
            return self.train_triplet(train_dataset, val_dataset, loss_kwargs, num_epochs, batch_size, seed)
        elif self.loss_name == "without_ft":
            return self.model, []
        else:
            return self.train_base(train_dataset, val_dataset, loss_kwargs, num_epochs, batch_size, seed, is_eval, recorder)

    def calculate_cosine_similarity(self, data_embeddings1, data_embeddings2):
        cosine_similarity = F.cosine_similarity(data_embeddings1, data_embeddings2, dim=1)
        return cosine_similarity
    
    def AnglE_similarity(self, data_embeddings1, data_embeddings2):       
        y_pred_re1, y_pred_im1 = torch.chunk(data_embeddings1, 2, dim=1)
        y_pred_re2, y_pred_im2 = torch.chunk(data_embeddings2, 2, dim=1)

        a = y_pred_re1
        b = y_pred_im1
        c = y_pred_re2
        d = y_pred_im2

        z = torch.sum(c**2 + d**2, dim=1, keepdim=True)
        re = (a * c + b * d) / z
        im = (b * c - a * d) / z

        dz = torch.sum(a**2 + b**2, dim=1, keepdim=True)**0.5
        dw = torch.sum(c**2 + d**2, dim=1, keepdim=True)**0.5
        re /= (dz / dw)
        im /= (dz / dw)

        y_pred = torch.concat((re, im), dim=1)
        y_pred = torch.abs(torch.sum(y_pred, dim=1))

        return y_pred

    def calculate_Spearman_rank_correlation_coefficient(self, data_embeddings1, data_embeddings2, scores_actual):
        if self.loss_name != 'angle_loss':
            cosine_similarities = self.calculate_cosine_similarity(data_embeddings1, data_embeddings2)
        else:
            cosine_similarities = self.AnglE_similarity(data_embeddings1, data_embeddings2)
        if self.is_llama:
          cosine_similarities = cosine_similarities.detach().cpu().float().numpy()
          cosine_similarities = np.asarray(cosine_similarities, dtype=np.float32)
          scores_actual = np.asarray(scores_actual, dtype=np.float32)
        sc, _ = spearmanr(cosine_similarities, scores_actual)
        return sc

    def calculate_Person_rank_correlation_coefficient(self, data_embeddings1, data_embeddings2, scores_actual):
        if self.loss_name != 'angle_loss':
            cosine_similarities = self.calculate_cosine_similarity(data_embeddings1, data_embeddings2)
        else:
            cosine_similarities = self.AnglE_similarity(data_embeddings1, data_embeddings2)
        if self.is_llama:
          cosine_similarities = cosine_similarities.detach().cpu().float().numpy()
          cosine_similarities = np.asarray(cosine_similarities, dtype=np.float32)
          scores_actual = np.asarray(scores_actual, dtype=np.float32)
        pc, _  = pearsonr(cosine_similarities, scores_actual)
        return pc

    def random_embedding_similarity(self, embeddings1, embeddings2, scores_actual, n_samples=10000):
        all_embeddings = torch.cat([embeddings1, embeddings2], dim=0)
        
        all_embeddings = torch.unique(all_embeddings, dim=0)

        n = len(all_embeddings)
        n_samples = min(n_samples, n * (n - 1))

        idx1 = torch.randint(0, n, (n_samples,))
        idx2 = torch.randint(0, n, (n_samples,))

        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]
        all_embeddings = all_embeddings.float()
        # reuse your existing function
        sims = self.calculate_cosine_similarity(
            all_embeddings[idx1], 
            all_embeddings[idx2]
        )
        if self.is_graph == True:
            test_name = getattr(self, '_current_test_name', self.train_dataset_name)
            generate_random_pair_distribution(sims, self.loss_name, self.model_id, self.pooling, self.train_dataset_name, test_name)
        return sims.mean().item()

    def measure_isoscore(self, embeddings1, embeddings2, scores_actual):
        all_embeddings = torch.cat([embeddings1, embeddings2], dim=0)
        all_embeddings_np = all_embeddings.float().detach().numpy()
        score = IsoScore(all_embeddings_np)
        return score

    def measure_discriminability(self, embeddings1, embeddings2, labels, 
                              range_percentiles=None):
        label_min = float(min(labels))
        range_label = float(max(labels)) - label_min
        # thresholds are offset by the label minimum (e.g. SICK-R runs 1-5, not 0-5)
        pos_threshold = label_min + (2/3) * range_label
        neg_threshold = label_min + (1/3) * range_label
        #print(neg_threshold)
        if self.is_llama:
            embeddings1 = embeddings1.float()
            embeddings2 = embeddings2.float()
        sims = self.calculate_cosine_similarity(embeddings1, embeddings2)
        
        # convert labels to tensor if numpy
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels)
        
        # split by human labels
        positive_mask = labels >= pos_threshold  # clearly similar
        negative_mask = labels <= neg_threshold  # clearly dissimilar
        
        # check enough samples exist
        n_pos = positive_mask.sum().item()
        n_neg = negative_mask.sum().item()
        
        if n_pos == 0 or n_neg == 0:
            print(f"Warning: not enough samples (pos={n_pos}, neg={n_neg})")
            print(f"Consider lowering pos_threshold or raising neg_threshold")
            return float('nan')   # not None: callers format this with :.4f and average it
        
        pos_sims = sims[positive_mask]
        neg_sims = sims[negative_mask]
        
        # compute gap
        pos_mean = pos_sims.mean().item()
        neg_mean = neg_sims.mean().item()
        gap = pos_mean - neg_mean  # larger = better

        if range_percentiles is None:
            sim_min, sim_max = sims.min().item(), sims.max().item()
        else:
            lo, hi = range_percentiles
            q = torch.tensor([lo / 100, hi / 100], device=sims.device, dtype=sims.dtype)
            sim_min, sim_max = torch.quantile(sims, q).tolist()

        sim_range = sim_max - sim_min
        relative_gap = gap / sim_range if sim_range > 1e-12 else float('nan')
        
        return relative_gap

    def evaluate_sts(self, test_dataset, test_name, graph = True):
        self.model.eval()
        self._current_test_name = test_name   # used to name the random-pair plot
        metric_fns = {
            'spearman': self.calculate_Spearman_rank_correlation_coefficient,
            'pearson': self.calculate_Person_rank_correlation_coefficient,
            'rand_mean': self.random_embedding_similarity,
            'isoscore': self.measure_isoscore,
            'disc_gap': self.measure_discriminability
        }

        test_dataloader = DataLoader(STSDataset(test_dataset['sentence1'], test_dataset['sentence2'], test_dataset['labels']), batch_size=90)
        all_embeddings1 = []
        all_embeddings2 = []
        all_labels = []

        all_sentences1 = []
        all_sentences2 = []
        i = 0
        with torch.no_grad():
            for sentences1, sentences2, labels in tqdm(test_dataloader, disable=False, desc="Extracting", leave=False):
                #for every pair extract the embedding and labels append to the empty list created
                embeddings1 = self.extract_embeddings(self.model, self.tokenizer, sentences1, self.device)
                embeddings2 = self.extract_embeddings(self.model, self.tokenizer, sentences2, self.device)
                all_embeddings1.append(embeddings1.cpu())
                all_embeddings2.append(embeddings2.cpu())
                all_sentences1.extend(sentences1)
                all_sentences2.extend(sentences2)
                all_labels.append(labels.cpu())

        data_embeddings1 = torch.cat(all_embeddings1)
        data_embeddings2 = torch.cat(all_embeddings2)
        data_labels = torch.cat(all_labels)
        data_labels_np = data_labels.numpy()

        if self.loss_name != 'angle_loss':
            cosine_similarities = self.calculate_cosine_similarity(data_embeddings1, data_embeddings2)
        else:
            cosine_similarities = self.AnglE_similarity(data_embeddings1, data_embeddings2)
        results = []
        for m in self.evaluate_metric:
            fn = metric_fns[m]
            results.append(fn(data_embeddings1, data_embeddings2, data_labels))
        if self.is_graph and graph:
            generate_distribution(self.model_id, self.pooling, self.loss_name, self.train_dataset_name, test_name, cosine_similarities, data_labels)
        return results

    def evaluate_all_sts(self, test_datasets):
        evaluation_result = []
        for name, test_dataset in test_datasets.items():
            results = self.evaluate_sts(test_dataset, name)
            result_set = {}
            for i, metric in enumerate(self.evaluate_metric):
                result_set[metric] = results[i]
            evaluation_result.append(result_set)
        return evaluation_result

    def sweep(self, sims, labels):
        sims, labels = np.asarray(sims, float), np.asarray(labels, int)
        order = np.argsort(-sims, kind="mergesort")
        s, y = sims[order], labels[order]
        tp = np.cumsum(y)
        fp = np.cumsum(1 - y)
        last = np.r_[np.diff(s) != 0, True]
        return s[last], tp[last], fp[last], y.sum(), len(y)

    def calculate_F1(self, sims, labels):
        t, tp, fp, n_pos, n = self.sweep(sims, labels)
        fn = n_pos - tp
        tn = (n - n_pos) - fp
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        return f1

    def calculate_accuracy(self, sims, labels):
        t, tp, fp, n_pos, n = self.sweep(sims, labels)
        fn = n_pos - tp
        tn = (n - n_pos) - fp
        acc = (tp + tn) / n
        return acc

    def best_acc_thresholds(self, sims, labels):
      t, _, _, _, _ = self.sweep(sims, labels)
      acc = self.calculate_accuracy(sims, labels)
      return t[np.argmax(acc)]

    def best_f1_thresholds(self, sims, labels):
      t, _, _, _, _ = self.sweep(sims, labels)
      f1 = self.calculate_F1(sims, labels)
      return t[np.argmax(f1)]

    def calculate_AP(self, sims, labels):
        _, tp, fp, n_pos, _ = self.sweep(sims, labels)
        if n_pos == 0:
            return float("nan")
        precision = tp / (tp + fp)
        recall = tp / n_pos
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    def calculate_ROC_AUC(self, sims, labels):

        _, tp, fp, n_pos, n = self.sweep(sims, labels)
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        tpr = np.r_[0.0, tp / n_pos]
        fpr = np.r_[0.0, fp / n_neg]
        return float(np.trapezoid(tpr, fpr))

    def f1_at(self, sims, labels, thr):
        sims = np.asarray(sims)
        labels = np.asarray(labels)

        pred = (sims >= thr).astype(int)

        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))  

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def accuracy_at(self, sims, labels, thr):
        sims = np.asarray(sims)
        labels = np.asarray(labels)

        pred = (sims >= thr).astype(int)

        correct = np.sum(pred == labels)
        total = len(labels)

        return correct / total

    def evaluate_sp(self, val_dataset, test_dataset, test_name):
        self.model.eval()
        #print(val_dataset)
        test_dataloader = DataLoader(STSDataset(test_dataset['sentence1'], test_dataset['sentence2'], test_dataset['labels']), batch_size=90)
        val_dataloader = DataLoader(STSDataset(val_dataset['sentence1'], val_dataset['sentence2'], val_dataset['labels']), batch_size=90)
        all_embeddings1 = []
        all_embeddings2 = []
        val_embeddings1 = []
        val_embeddings2 = []
        all_labels = []
        val_labels = []

        all_sentences1 = []
        all_sentences2 = []
        val_sentences1 = []
        val_sentences2 = []
        i = 0
        with torch.no_grad():
            for sentences1, sentences2, labels in tqdm(val_dataloader, disable=False, desc="finding threshold", leave=False):
                #for every pair extract the embedding and labels append to the empty list created
                embeddings1 = self.extract_embeddings(self.model, self.tokenizer, sentences1, self.device)
                embeddings2 = self.extract_embeddings(self.model, self.tokenizer, sentences2, self.device)
                val_embeddings1.append(embeddings1.cpu())
                val_embeddings2.append(embeddings2.cpu())
                val_sentences1.extend(sentences1)
                val_sentences2.extend(sentences2)
                val_labels.append(labels.cpu())

        val_embeddings1 = torch.cat(val_embeddings1)
        val_embeddings2 = torch.cat(val_embeddings2)
        val_labels = torch.cat(val_labels)
        val_labels_np = val_labels.numpy()

        if self.loss_name != 'angle_loss':
            val_cosine_similarities = self.calculate_cosine_similarity(val_embeddings1, val_embeddings2)
        else:
            val_cosine_similarities = self.AnglE_similarity(val_embeddings1, val_embeddings2)

        f1_threhold = self.best_f1_thresholds(val_cosine_similarities, val_labels)
        acc_threshold = self.best_acc_thresholds(val_cosine_similarities, val_labels)

        with torch.no_grad():
            for sentences1, sentences2, labels in tqdm(test_dataloader, disable=False, desc="Extracting", leave=False):
                #for every pair extract the embedding and labels append to the empty list created
                embeddings1 = self.extract_embeddings(self.model, self.tokenizer, sentences1, self.device)
                embeddings2 = self.extract_embeddings(self.model, self.tokenizer, sentences2, self.device)
                all_embeddings1.append(embeddings1.cpu())
                all_embeddings2.append(embeddings2.cpu())
                all_sentences1.extend(sentences1)
                all_sentences2.extend(sentences2)
                all_labels.append(labels.cpu())

        data_embeddings1 = torch.cat(all_embeddings1)
        data_embeddings2 = torch.cat(all_embeddings2)
        data_labels = torch.cat(all_labels)
        data_labels_np = data_labels.numpy()

        if self.loss_name != 'angle_loss':
            cosine_similarities = self.calculate_cosine_similarity(data_embeddings1, data_embeddings2)
        else:
            cosine_similarities = self.AnglE_similarity(data_embeddings1, data_embeddings2)
        results = []
        ROC_AUC = self.calculate_ROC_AUC(cosine_similarities, data_labels)
        AP = self.calculate_AP(cosine_similarities, data_labels)
        F1 = self.f1_at(cosine_similarities, data_labels, f1_threhold)
        acc = self.accuracy_at(cosine_similarities, data_labels, acc_threshold)
        results.append(AP)
        results.append(ROC_AUC)
        results.append(F1)
        results.append(acc)
        results.append(f1_threhold)
        results.append(acc_threshold)
        if self.is_graph:
            plot_SP_threshold(cosine_similarities, data_labels, f1_threhold,
                              self.model_id, self.pooling, self.loss_name,
                              self.train_dataset_name, test_name)
        return results
        
    def evaluate_all_sp(self, val_dataset, test_datasets):
        evaluation_result = []
        for name, test_dataset in test_datasets.items():
            results = self.evaluate_sp(val_dataset[name], test_dataset, name)
            result_set = {}
            for i, metric in enumerate(self.evaluate_sp_metric):
                result_set[metric] = results[i]
            evaluation_result.append(result_set)
        return evaluation_result



    def evaluate_sc(self, train_embedding, train_label, test_embedding, test_label, test_name):

        scaler = StandardScaler().fit(train_embedding)
        clf = LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10, 100], cv=5, max_iter=2000)
        clf.fit(scaler.transform(train_embedding), train_label)

        pred = clf.predict(scaler.transform(test_embedding))
        if self.is_graph:
            plot_confusion(test_label, pred, self.model_id, self.pooling,
                          self.loss_name, self.train_dataset_name, test_name)
        return [accuracy_score(test_label, pred), f1_score(test_label, pred, average="macro"), clf.C_[0]]

    def evaluate_cl(self, val_dataset, test_dataset, test_name):
        self.model.eval()
        #print(val_dataset)
        test_dataloader = DataLoader(CLDataset(test_dataset['text'], test_dataset['labels']), batch_size=90)
        val_dataloader = DataLoader(CLDataset(val_dataset['text'], val_dataset['labels']), batch_size=90)
        all_embeddings1 = []
        val_embeddings1 = []
        all_labels = []
        val_labels = []

        all_sentences1 = []
        val_sentences1 = []
        i = 0
        with torch.no_grad():
            for text, labels in tqdm(val_dataloader, disable=False, desc="training_classfier", leave=False):
                #for every pair extract the embedding and labels append to the empty list created
                embeddings1 = self.extract_embeddings(self.model, self.tokenizer, text, self.device)
                val_embeddings1.append(embeddings1.cpu())
                val_sentences1.extend(text)
                val_labels.append(labels.cpu())

        val_embeddings1 = torch.cat(val_embeddings1)
        val_labels = torch.cat(val_labels)

        with torch.no_grad():
            for text, labels in tqdm(test_dataloader, disable=False, desc="Extracting", leave=False):
                #for every pair extract the embedding and labels append to the empty list created
                embeddings1 = self.extract_embeddings(self.model, self.tokenizer, text, self.device)
                all_embeddings1.append(embeddings1.cpu())
                all_sentences1.extend(text)
                all_labels.append(labels.cpu())

        data_embeddings1 = torch.cat(all_embeddings1)
        data_labels = torch.cat(all_labels)
        data_labels_np = data_labels.numpy()
        acc_score, f1_score, _ = self.evaluate_sc(val_embeddings1, val_labels, data_embeddings1, data_labels, test_name)
        results = []
        results.append(acc_score)
        results.append(f1_score)
        return results

    def evaluate_all_cl(self, val_dataset, test_datasets):
        evaluation_result = []
        for name, test_dataset in test_datasets.items():
            results = self.evaluate_cl(val_dataset[name], test_dataset, name)
            result_set = {}
            for i, metric in enumerate(self.evaluate_cl_metric):
                result_set[metric] = results[i]
            evaluation_result.append(result_set)
        return evaluation_result

    def evaluate_clustering(self, embeddings, labels, test_name, seed=0):
        X = np.asarray(embeddings, dtype=np.float32)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        y = np.asarray(labels)
        k = len(np.unique(y))

        pred = MiniBatchKMeans(n_clusters=k, batch_size=500, n_init="auto",
                              random_state=seed).fit_predict(X)
        if self.is_graph:
            plot_clusters(embeddings, labels, pred, self.model_id, self.pooling, self.loss_name, self.train_dataset_name, test_name, n_points=5000, seed=seed)
        return v_measure_score(y, pred), normalized_mutual_info_score(y, pred), adjusted_rand_score(y, pred)


    def evaluate_ct(self, test_dataset, test_name):
        self.model.eval()
        #print(val_dataset)
        test_dataloader = DataLoader(CLDataset(test_dataset['sentences'], test_dataset['labels']), batch_size=90)
        all_embeddings1 = []
        all_labels = []

        all_sentences1 = []
        i = 0
        with torch.no_grad():
            for sentences, labels in tqdm(test_dataloader, disable=False, desc="Extracting", leave=False):
                #for every pair extract the embedding and labels append to the empty list created
                embeddings1 = self.extract_embeddings(self.model, self.tokenizer, sentences, self.device)
                all_embeddings1.append(embeddings1.cpu())
                all_sentences1.extend(sentences)
                all_labels.append(labels.cpu())

        data_embeddings1 = torch.cat(all_embeddings1)
        data_labels = torch.cat(all_labels)
        data_labels_np = data_labels.numpy()
        v_measure, nmi, ari = self.evaluate_clustering(data_embeddings1, data_labels, test_name, seed=0)
        results = []
        results.append(v_measure)
        results.append(nmi)
        results.append(ari)
        
        return results

    def evaluate_all_ct(self, test_datasets):
        evaluation_result = []
        for name, test_dataset in test_datasets.items():
            results = self.evaluate_ct(test_dataset, name)
            result_set = {}
            for i, metric in enumerate(self.evaluate_ct_metric):
                result_set[metric] = results[i]
            evaluation_result.append(result_set)
        return evaluation_result

    def ndcg_at_k(self, ranked_doc_ids, rel_dict, k=10):
        dcg = sum(rel_dict.get(d, 0) / np.log2(i + 2) for i, d in enumerate(ranked_doc_ids[:k]))
        ideal = sorted(rel_dict.values(), reverse=True)[:k]
        idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
        return dcg / idcg if idcg > 0 else 0.0

    def calculate_retrieval_metrics(self, Q, D, q_ids, doc_ids, rel, k=10, top_n=100):

        ndcgs, recalls = [], []
        for start in range(0, len(Q), 256):
            S = Q[start:start + 256] @ D.T
            top = np.argsort(-S, axis=1)[:, :top_n + 1]
            for qi, row in enumerate(top):
                q = q_ids[start + qi]
                relevant = rel[q]
                ranked = [doc_ids[j] for j in row if doc_ids[j] != q][:top_n]   
                ndcgs.append(self.ndcg_at_k(ranked, relevant, k=k))
                recalls.append(len(set(ranked) & set(relevant)) / len(relevant))
        return float(np.mean(ndcgs)), float(np.mean(recalls))

    def evaluate_rt(self, test_dataset, test_name):
        self.model.eval()
        doc_dataloader   = DataLoader(RTDataset(test_dataset['doc_texts']), batch_size=90, shuffle=False)
        query_dataloader = DataLoader(RTDataset(test_dataset['q_texts']),   batch_size=90, shuffle=False)
        all_doc = []
        all_query = []

        with torch.no_grad():
            for sentences in tqdm(doc_dataloader, disable=False, desc="Extracting Doc", leave=False):
                #for every pair extract the embedding and labels append to the empty list created
                doc_embeddings = self.extract_embeddings(self.model, self.tokenizer, sentences, self.device)
                all_doc.append(doc_embeddings.cpu())
            
            for sentences in tqdm(query_dataloader, disable=False, desc="Extracting query", leave=False):
                #for every pair extract the embedding and labels append to the empty list created
                query_embeddings = self.extract_embeddings(self.model, self.tokenizer, sentences, self.device)
                all_query.append(query_embeddings.cpu())


        D = torch.cat(all_doc).numpy()
        Q = torch.cat(all_query).numpy()
        D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
        Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
        ndcg, recall = self.calculate_retrieval_metrics(
            Q, D, test_dataset['q_ids'], test_dataset['doc_ids'], test_dataset['rel'])
        results = []
        results.append(ndcg)
        results.append(recall)
        return results

    def evaluate_all_rt(self, test_datasets):
        evaluation_result = []
        for name, test_dataset in test_datasets.items():
            results = self.evaluate_rt(test_dataset, name)
            result_set = {}
            for i, metric in enumerate(self.evaluate_rt_metric):
                result_set[metric] = results[i]
            evaluation_result.append(result_set)
        return evaluation_result

    def evaluate_mteb(self, task_names):
        """Official MTEB evaluation (pip install mteb, v2) on the given task names, e.g. the
        SimCSE/AnglE STS protocol ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
        'STSBenchmark', 'SICK-R'] (main score = Spearman of cosine). Returns {task name: main score}."""
        import mteb
        from mteb.models.abs_encoder import AbsEncoder
        from mteb.models.model_meta import ModelMeta, ScoringFunction

        class MTEBEncoder(AbsEncoder):
            """Lets mteb.evaluate call the fine-tuned model with the same pooling as training.
            mteb only uses a passed object as the model if it has `mteb_model_meta`."""
            def __init__(self, trainer):
                self.trainer = trainer
                self.model = trainer.model
                name = trainer.model_id if '/' in trainer.model_id else f'local/{trainer.model_id}'
                self.mteb_model_meta = ModelMeta.create_empty(overwrites=dict(
                    name=name, revision=f'{trainer.loss_name}_{trainer.pooling}',
                    similarity_fn_name=ScoringFunction.COSINE))

            @torch.no_grad()
            def encode(self, inputs, *, task_metadata=None, hf_split=None, hf_subset=None,
                       prompt_type=None, **kwargs):
                t = self.trainer
                t.model.eval()
                out = [t.extract_embeddings(t.model, t.tokenizer, list(batch['text']), t.device).cpu()
                       for batch in inputs]                 # DataLoader of {'text': [...]}
                return torch.cat(out).numpy()

        tasks = mteb.get_tasks(tasks=task_names)
        encoder = MTEBEncoder(self)
        scores = {}
        # one printed line per task instead of a tqdm bar: nested bars (mteb has its own)
        # cannot be redrawn in Colab and print a new line on every update
        for i, task in enumerate(tasks, 1):
            name = task.metadata.name
            print(f' MTEB [{i}/{len(tasks)}] {name} ({task.metadata.type}) ...', flush=True)
            start = time.time()
            # cache=None: never reuse results from an earlier run/loss
            results = mteb.evaluate(encoder, tasks=[task], cache=None,
                                    encode_kwargs={'batch_size': 90})
            scores.update({r.task_name: float(r.main_score) for r in results.task_results})
            print(f' MTEB [{i}/{len(tasks)}] {name} done: main_score {scores.get(name, float("nan")):.4f} '
                  f'({(time.time() - start) / 60:.1f} min)', flush=True)
        return scores


# ############################################################################
#  MECHANISM STUDY  (formerly instrumentation.py + similarity_bins.py)
#
#  Records, without changing how training works:
#    Test 2  calibration vs ranking split of the MSE during training  (log_step)
#    Test 3  representation drift: linear CKA (overall + per layer),
#            relative weight distance, random-pair mean cosine        (start / finish)
#    Test 4  mean-centred evaluation                                 (finish)
#    Bins    Spearman per closeness bin: how close each test pair is to its nearest
#            training pair. 'exact' = the pair itself is in the training set (seen);
#            the other bins split the unseen pairs from near-duplicate to novel.
#
#  Usage (see run.py):
#    sims = load_or_compute_similarity(path, train_dataset, test_sets)
#    rec  = RunRecorder(out_dir, model, tokenizer, probe, train_dataset, sims=sims, pooling='mean')
#    trainer.train(..., recorder=rec)          # calls rec.start() and rec.log_step()
#    summarize(rec.finish(test_sets))
#    print_gain_by_bin(sims, runs_mse_norm, runs_mean_adjust, 'MSE-norm', 'Mean-Adjust')
#
#  Closeness uses character n-gram TF-IDF, which does not depend on any trained model
#  (using the fine-tuned model's own embeddings would be circular).
#  Prediction if Mean-Adjust's advantage is a generalization advantage:
#    the gain is ~0 on exact copies and grows as pairs get further from training.
# ############################################################################

# ---------------------------- text helpers ----------------------------------

def clean_text(s):
    """Lower-case and collapse punctuation/whitespace, so trivial differences don't count."""
    return re.sub(r'\W+', ' ', s.lower()).strip()


def pair_key(a, b):
    """Order-independent key of a sentence pair."""
    return frozenset((clean_text(a), clean_text(b)))


def build_train_pairs(train_dataset):
    return {pair_key(a, b) for a, b in zip(train_dataset['sentence1'], train_dataset['sentence2'])}


def seen_mask(dataset, train_pairs):
    """True where the (unordered) test pair also occurs in the training set ('exact' bin)."""
    return np.array([pair_key(a, b) in train_pairs
                     for a, b in zip(dataset['sentence1'], dataset['sentence2'])], dtype=bool)


def get_labels(ds):
    for col in ('labels', 'score', 'label', 'similarity_score'):
        if col in ds.column_names:
            return np.asarray(ds[col], dtype=float)
    raise KeyError(f"No label column in {ds.column_names}")


def build_probe_set(test_sets, n_per_set=200, seed=0):
    """Fixed list of sentences used to compare representations before / after training.
    Same seed => same probe across all runs."""
    rng = random.Random(seed)
    probe = []
    for name in sorted(test_sets):
        sents = list(dict.fromkeys(list(test_sets[name]['sentence1'])))  # unique, ordered
        rng.shuffle(sents)
        probe.extend(sents[:n_per_set])
    return probe


# ---------------------------- closeness bins --------------------------------

SIM_BINS = [  # (label, low, high); 'exact' = seen pairs, other bins contain unseen pairs only
    ('exact',      None, None),
    ('near>=.90',  0.90, 1.01),
    ('sim.70-.90', 0.70, 0.90),
    ('rel.50-.70', 0.50, 0.70),
    ('novel<.50',  -1.0, 0.50),
]
BIN_LABELS = [b[0] for b in SIM_BINS]
MIN_BIN_PAIRS = 30   # bins smaller than this are reported as empty


def bin_masks(sim, seen):
    """dict bin label -> boolean mask over the test pairs of one dataset."""
    sim, seen = np.asarray(sim), np.asarray(seen, dtype=bool)
    return {label: (seen if lo is None else (~seen) & (sim >= lo) & (sim < hi))
            for label, lo, hi in SIM_BINS}


def pair_similarity(train_dataset, test_sets, chunk=500):
    """For each test pair (a, b): max over training pairs (a', b') of
    min(sim(a, a'), sim(b, b')), also trying the swapped order. A pair is only 'close'
    if BOTH of its sentences are close to the two sentences of the same training pair.
    Returns dict name -> float32 array in dataset order (1.0 = identical after cleaning)."""
    t1 = [clean_text(s) for s in train_dataset['sentence1']]
    t2 = [clean_text(s) for s in train_dataset['sentence2']]
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True)
    vec.fit(t1 + t2)
    T1, T2 = vec.transform(t1), vec.transform(t2)          # rows are L2-normalised
    out = {}
    for name, ds in test_sets.items():
        a_all = [clean_text(s) for s in ds['sentence1']]
        b_all = [clean_text(s) for s in ds['sentence2']]
        best = np.empty(len(a_all), dtype=np.float32)
        for i in range(0, len(a_all), chunk):
            A = vec.transform(a_all[i:i + chunk])
            B = vec.transform(b_all[i:i + chunk])
            same = np.minimum((A @ T1.T).toarray(), (B @ T2.T).toarray())
            swap = np.minimum((A @ T2.T).toarray(), (B @ T1.T).toarray())
            best[i:i + chunk] = np.maximum(same, swap).max(1)
        out[name] = np.clip(best, 0.0, 1.0)
    return out


def load_or_compute_similarity(path, train_dataset, test_sets):
    """Cached pair_similarity: computed once, then read from `path`.
    Recomputed automatically if the cache doesn't match the test sets' names/sizes."""
    if os.path.exists(path):
        sims = np.load(path, allow_pickle=True).item()
        if set(sims) == set(test_sets) and all(len(sims[k]) == len(test_sets[k]) for k in sims):
            return sims
    sims = pair_similarity(train_dataset, test_sets)
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, sims, allow_pickle=True)
    return sims


def safe_spearman(pred, gold, min_n=10):
    """Spearman, or None if fewer than min_n pairs or either side is constant.
    Never raises, even when RuntimeWarnings are turned into errors (run.py)."""
    pred, gold = np.asarray(pred, dtype=float), np.asarray(gold, dtype=float)
    if len(pred) < min_n or np.ptp(pred) == 0 or np.ptp(gold) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = spearmanr(pred, gold).correlation
    return None if np.isnan(r) else float(r)


# ---------------------------- representation measures -----------------------

def _pool(h, mask, pooling):
    """h: [b, t, d], mask: [b, t, 1]. Matches TrainerSE.extract_embeddings."""
    if pooling == 'cls':
        return h[:, 0]
    if pooling == 'max':
        return h.masked_fill(mask == 0, -1e9).max(1).values
    return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)   # 'mean' (and fallback)


@torch.no_grad()
def encode(model, tokenizer, sentences, batch_size=64, max_length=None,
           all_layers=False, pooling='mean'):
    """Sentence embeddings with the same pooling and truncation as training.
    Returns a CPU float tensor [n, d], or a list per layer (embedding layer first) if
    all_layers=True. Restores the model's train/eval mode."""
    if tokenizer.pad_token is None:                       # e.g. Llama
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    pooling = pooling.lower()
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    outs = None
    for i in range(0, len(sentences), batch_size):
        batch = tokenizer(sentences[i:i + batch_size], padding=True, truncation=True,
                          max_length=max_length, return_tensors='pt').to(device)
        out = model(**batch, output_hidden_states=all_layers, output_attentions=False)
        mask = batch['attention_mask'].unsqueeze(-1).float()
        states = out.hidden_states if all_layers else (out.last_hidden_state,)
        pooled = [_pool(h, mask, pooling).float().cpu() for h in states]
        if outs is None:
            outs = [[] for _ in pooled]
        for k, p in enumerate(pooled):
            outs[k].append(p)
    if was_training:
        model.train()
    layers = [torch.cat(o) for o in outs]
    return layers if all_layers else layers[0]


def calib_rank_split(p, y, eps=1e-12):
    """Test 2: split MSE(p, y) into ranking error (left after the best affine fit a*p+b)
    and calibration error (the rest). Returns (total, rank, calib)."""
    p, y = p.double().flatten(), y.double().flatten()
    total = ((p - y) ** 2).mean()
    pm, ym = p.mean(), y.mean()
    a = ((p - pm) * (y - ym)).mean() / (((p - pm) ** 2).mean() + eps)
    b = ym - a * pm
    rank = ((a * p + b - y) ** 2).mean()
    return total.item(), rank.item(), (total - rank).item()


def linear_cka(X, Y):
    """Test 3: linear CKA between two [n, d] representations of the same inputs.
    Centring removes the shared offset, so this compares structure, not rand_mean."""
    X = X.double() - X.double().mean(0, keepdim=True)
    Y = Y.double() - Y.double().mean(0, keepdim=True)
    hsic = (X.T @ Y).norm() ** 2
    return (hsic / ((X.T @ X).norm() * (Y.T @ Y).norm())).item()


def snapshot_weights(model):
    return {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()
            if v.is_floating_point()}


def weight_distance(model, theta0):
    """Test 3: relative L2 distance ||theta - theta0|| / ||theta0|| over all float parameters."""
    num, den = 0.0, 0.0
    for k, v in model.state_dict().items():
        if k in theta0:
            v0 = theta0[k]
            num += ((v.detach().float().cpu() - v0) ** 2).sum().item()
            den += (v0 ** 2).sum().item()
    return (num / max(den, 1e-12)) ** 0.5


def random_pair_mean_cos(E, n_pairs=5000, seed=0):
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, len(E), (n_pairs,), generator=g)
    j = torch.randint(0, len(E), (n_pairs,), generator=g)
    keep = i != j
    return F.cosine_similarity(E[i[keep]], E[j[keep]]).mean().item()


@torch.no_grad()
def evaluate_extended(model, tokenizer, test_sets, train_pairs, mu, sims=None,
                      batch_size=64, pooling='mean', max_length=None):
    """Test 4 + bins. For every test set: Spearman with plain cosine and with mean-centred
    cosine (mu = mean embedding of training sentences), and Spearman inside each closeness
    bin if sims is given. Returns (results, per-pair arrays for predictions.npz)."""
    results, per_pair = {}, {}
    for name, ds in test_sets.items():
        gold = get_labels(ds)
        e1 = encode(model, tokenizer, list(ds['sentence1']), batch_size, max_length, pooling=pooling)
        e2 = encode(model, tokenizer, list(ds['sentence2']), batch_size, max_length, pooling=pooling)
        cos = F.cosine_similarity(e1, e2).numpy()
        cos_c = F.cosine_similarity(e1 - mu, e2 - mu).numpy()
        seen = seen_mask(ds, train_pairs)
        results[name] = {
            'n': int(len(gold)), 'n_seen': int(seen.sum()),
            'spearman': safe_spearman(cos, gold),
            'spearman_centered': safe_spearman(cos_c, gold),
        }
        if sims is not None and name in sims:
            sim = np.asarray(sims[name])
            assert len(sim) == len(gold), f'{name}: similarity array does not match test set'
            results[name]['bins'] = {
                label: [int(m.sum()), safe_spearman(cos[m], gold[m], MIN_BIN_PAIRS)]
                for label, m in bin_masks(sim, seen).items()}
            per_pair[f'{name}__sim'] = sim
        # per-pair predictions in dataset order, so binning can be redone offline
        per_pair[f'{name}__cos'] = cos
        per_pair[f'{name}__cos_centered'] = cos_c
        per_pair[f'{name}__gold'] = gold
        per_pair[f'{name}__seen'] = seen
    return results, per_pair


# ---------------------------- recorder --------------------------------------

class RunRecorder:
    """Collects everything for one (backbone, loss, run) and writes it to out_dir:
        predictions.npz      per-pair cos / cos_centered / gold / seen / sim for every test set
        E0.npy, E1.npy       probe embeddings before / after training
        step_log.json        Test 2, every `log_every` optimizer steps
        epoch_log.json       anything passed to log_epoch
        final_results.json   CKA, weight distance, rand_mean, extended evaluation
        best_fp16.pt         (optional) half-precision state dict
    """

    def __init__(self, out_dir, model, tokenizer, probe_sentences, train_dataset,
                 sims=None, pooling='mean', max_length=None,
                 log_every=20, n_mu_sentences=4000, seed=0):
        self.out_dir = out_dir
        self.model, self.tokenizer = model, tokenizer
        self.probe = probe_sentences
        self.sims = sims
        self.pooling, self.max_length = pooling, max_length
        self.log_every = log_every
        self.train_pairs = build_train_pairs(train_dataset)
        sents = list(dict.fromkeys(list(train_dataset['sentence1']) + list(train_dataset['sentence2'])))
        random.Random(seed).shuffle(sents)
        self.mu_sentences = sents[:n_mu_sentences]
        self.step_log, self.epoch_log = [], []
        self.started = False
        os.makedirs(out_dir, exist_ok=True)

    def _encode(self, sentences, all_layers=False):
        return encode(self.model, self.tokenizer, sentences, max_length=self.max_length,
                      all_layers=all_layers, pooling=self.pooling)

    def start(self):
        """Before the first training step (called by TrainerSE.train_base)."""
        TRACE.clear()
        self.step_log, self.epoch_log = [], []
        self.theta0 = snapshot_weights(self.model)
        self.E0_layers = self._encode(self.probe, all_layers=True)
        self.E0 = self.E0_layers[-1]
        self.started = True

    def log_step(self, step):
        """After every optimizer step; reads what the loss wrote into TRACE."""
        if step % self.log_every or 'p' not in TRACE:
            return
        p, y = TRACE['p'], TRACE['y']
        total, rank, calib = calib_rank_split(p, y)
        self.step_log.append({
            'step': step, 'total': total, 'rank': rank, 'calib': calib,
            'calib_share': calib / total if total > 0 else 0.0,
            'p_mean': p.float().mean().item(), 'p_std': p.float().std().item(),
            'y_mean': y.float().mean().item(), 'y_std': y.float().std().item(),
        })

    def log_epoch(self, epoch, metrics):
        self.epoch_log.append({'epoch': epoch, **metrics})

    def finish(self, test_sets, save_checkpoint=False):
        """After training (and after the best checkpoint is loaded)."""
        if not self.started:
            raise RuntimeError('RunRecorder.finish() called before start(); '
                               'pass recorder=rec to trainer.train().')
        E1_layers = self._encode(self.probe, all_layers=True)
        E1 = E1_layers[-1]
        mu = self._encode(self.mu_sentences).mean(0, keepdim=True)
        eval_results, per_pair = evaluate_extended(
            self.model, self.tokenizer, test_sets, self.train_pairs, mu, sims=self.sims,
            pooling=self.pooling, max_length=self.max_length)
        final = {
            'cka': linear_cka(self.E0, E1),
            'cka_per_layer': [linear_cka(a, b) for a, b in zip(self.E0_layers, E1_layers)],
            'weight_distance': weight_distance(self.model, self.theta0),
            'rand_mean_before': random_pair_mean_cos(self.E0),
            'rand_mean_after': random_pair_mean_cos(E1),
            'rand_mean_after_centered': random_pair_mean_cos(E1 - mu),
            'n_logged_steps': len(self.step_log),
            'eval': eval_results,
        }
        np.savez_compressed(os.path.join(self.out_dir, 'predictions.npz'), **per_pair)
        np.save(os.path.join(self.out_dir, 'E0.npy'), self.E0.numpy())
        np.save(os.path.join(self.out_dir, 'E1.npy'), E1.numpy())
        for fname, obj in [('step_log.json', self.step_log), ('epoch_log.json', self.epoch_log),
                           ('final_results.json', final)]:
            with open(os.path.join(self.out_dir, fname), 'w') as f:
                json.dump(obj, f, indent=1)
        if save_checkpoint:
            torch.save({k: (v.half() if v.is_floating_point() else v)
                        for k, v in self.model.state_dict().items()},
                       os.path.join(self.out_dir, 'best_fp16.pt'))
        del self.theta0                                    # free the CPU weight copy
        TRACE.clear()
        return final


# ---------------------------- reporting -------------------------------------

def summarize(final):
    """Short printout of one RunRecorder.finish() result."""
    print(f"CKA {final['cka']:.3f} | weight dist {final['weight_distance']:.4f} | "
          f"rand_mean {final['rand_mean_before']:.3f} -> {final['rand_mean_after']:.3f} "
          f"(centred {final['rand_mean_after_centered']:.3f}) | "
          f"logged steps {final['n_logged_steps']}")
    fmt = lambda v: '  -  ' if v is None else f'{v:.4f}'
    print(f"{'dataset':8s} {'seen%':>6s} {'all':>7s} {'centred':>7s}")
    for name, r in final['eval'].items():
        print(f"{name:8s} {100 * r['n_seen'] / r['n']:6.1f} {fmt(r['spearman']):>7s} "
              f"{fmt(r['spearman_centered']):>7s}")
    if any('bins' in r for r in final['eval'].values()):
        print(f"Spearman by closeness to nearest training pair  (- = under {MIN_BIN_PAIRS} pairs)")
        print(f"{'dataset':8s}" + ''.join(f'{l:>17s}' for l in BIN_LABELS))
        for name, r in final['eval'].items():
            if 'bins' in r:
                cells = ''.join(f'{"-":>17s}' if sp is None else f'{f"{sp:.4f} (n={n})":>17s}'
                                for n, sp in (r['bins'][l] for l in BIN_LABELS))
                print(f'{name:8s}{cells}')


def binned_gain(sims, runs_a, runs_b, key='cos', min_pairs=MIN_BIN_PAIRS):
    """Gain = Spearman(B) - Spearman(A) inside each bin, per dataset, averaged over runs.
    runs_a / runs_b: run folders paired by position. key: 'cos' or 'cos_centered'.
    Spearman is computed WITHIN a dataset (label scales differ); the 'ALL' row averages
    the per-dataset gains weighted by bin size. Returns {dataset: {bin: (n, mean, std)}}."""
    assert len(runs_a) == len(runs_b), 'pair the runs: same number of runs for both losses'
    A = [np.load(os.path.join(r, 'predictions.npz')) for r in runs_a]
    B = [np.load(os.path.join(r, 'predictions.npz')) for r in runs_b]
    table = {}
    for name, sim in sims.items():
        if f'{name}__gold' not in A[0]:
            continue
        gold, seen = A[0][f'{name}__gold'], A[0][f'{name}__seen'].astype(bool)
        assert len(gold) == len(sim), f'{name}: test set does not match saved predictions'
        row = {}
        for label, m in bin_masks(sim, seen).items():
            n, gains = int(m.sum()), []
            if n >= min_pairs:
                for a, b in zip(A, B):
                    sa = safe_spearman(a[f'{name}__{key}'][m], gold[m], min_pairs)
                    sb = safe_spearman(b[f'{name}__{key}'][m], gold[m], min_pairs)
                    if sa is not None and sb is not None:
                        gains.append(sb - sa)
            row[label] = (n, float(np.mean(gains)) if gains else None,
                          float(np.std(gains, ddof=1)) if len(gains) > 1 else None)
        table[name] = row
    all_row = {}
    for label in BIN_LABELS:
        cells = [table[d][label] for d in table if table[d][label][1] is not None]
        n = sum(c[0] for c in cells)
        all_row[label] = (n, float(sum(c[0] * c[1] for c in cells) / n) if n else None, None)
    table['ALL'] = all_row
    return table


def print_gain_by_bin(sims, runs_a, runs_b, name_a='A', name_b='B', key='cos'):
    """After both losses finish: gain (B - A) per dataset and closeness bin."""
    table = binned_gain(sims, runs_a, runs_b, key=key)
    print(f'\n===== Spearman gain by closeness bin: {name_b} minus {name_a} '
          f'({len(runs_a)} runs, {key}) =====')
    print(f"{'dataset':8s}" + ''.join(f'{l:>22s}' for l in BIN_LABELS))
    for name, row in table.items():
        print(f'{name:8s}' + ''.join(f'{"-":>22s}' if g is None else f'{f"{g:+.4f} (n={n})":>22s}'
                                     for n, g, _ in (row[l] for l in BIN_LABELS)))
