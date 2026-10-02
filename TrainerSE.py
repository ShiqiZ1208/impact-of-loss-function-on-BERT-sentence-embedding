from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from lossfunc import get_loss
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

    def train_base(self, train_dataset, val_dataset, loss_kwargs, num_epochs, batch_size, seed=42, reval_datasets = [], is_eval = False):
        loss_log = []
        loss_function = get_loss(self.loss_name, **loss_kwargs)
        sts_name = self.train_dataset_name
        best_spearman = -1
        best_model_state = None
        if self.is_seed == True:
            g = torch.Generator()
            g.manual_seed(seed)
        else:
            g = None 
        i = 0

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
            
            if is_eval == True:
                results = self.evaluate_sts(val_dataset, sts_name, graph = False)
                spearman = results['spearman']
                if spearman > best_spearman:
                  print(f"current spearman{spearman}")
                  best_spearman = spearman
                  best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        if is_eval == True:
            self.model.load_state_dict(best_model_state)
        return self.model, loss_log


    def train(self, train_dataset, val_dataset, loss_kwargs, num_epochs, batch_size, seed = 42, is_eval = False):
        if self.loss_name == "triplet":
            return self.train_triplet(train_dataset, val_dataset, loss_kwargs, num_epochs, batch_size, seed, is_eval)
        elif self.loss_name == "without_ft":
            return self.model, []
        else:
            return self.train_base(train_dataset, val_dataset, loss_kwargs, num_epochs, batch_size, seed, is_eval)

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
            generate_random_pair_distribution(sims, self.loss_name, self.model_id, self.pooling, self.train_dataset_name, self.train_dataset_name)
        return sims.mean().item()

    def measure_isoscore(self, embeddings1, embeddings2, scores_actual):
        all_embeddings = torch.cat([embeddings1, embeddings2], dim=0)
        all_embeddings_np = all_embeddings.float().detach().numpy()
        score = IsoScore(all_embeddings_np)
        return score

    def measure_discriminability(self, embeddings1, embeddings2, labels, 
                              range_percentiles=None):
        range_label = max(labels) - min(labels)
        pos_threshold = (2/3) * range_label
        #print(pos_threshold)
        neg_threshold = (1/3) * range_label
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
            return None
        
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
        for start in range(0, len(Q), 256):                 # chunks to save memory
            S = Q[start:start + 256] @ D.T                   # cosine similarity
            top = np.argsort(-S, axis=1)[:, :top_n]
            for qi, row in enumerate(top):
                relevant = rel[q_ids[start + qi]]
                ranked = [doc_ids[j] for j in row]
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
