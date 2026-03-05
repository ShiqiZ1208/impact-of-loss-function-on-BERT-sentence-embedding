from torch._C import parse_schema
from tqdm import tqdm
from lossfunc import get_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from torch.optim import AdamW
from datapreprocess import prepare_dataset, STSDataset, TripDataset


class TrainerSE:
    def __init__(self, model, device, tokenizer, model_id, mode='cls', lrate = 5e-5):
        self.model = model
        self.optimizer = AdamW(model.parameters(), lrate)
        self.device = device
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.ib = InformationBottleneck().to(self.device)
    

    def extract_embeddings(self, model, tokenizer, sentences, device, mode):
        if self.model_id == 'mistralai/Mistral-7B-v0.1':
          tokenizer.pad_token = tokenizer.eos_token
          model.config.pad_token_id = tokenizer.pad_token_id
        encodings = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(device)
        output = model(**encodings, output_attentions=True, return_dict=True)
        if mode.lower() =="cls":
          embeddings = output.last_hidden_state[:, 0, :]
        elif mode.lower() == "mean":
          token_embeddings = output.last_hidden_state
          attention_mask = encodings['attention_mask'].unsqueeze(-1)
          embeddings = (token_embeddings * attention_mask).sum(1) / attention_mask.sum(1)
        elif mode.lower() == "max":
          token_embeddings = output.last_hidden_state
          attention_mask = encodings['attention_mask'].unsqueeze(-1).expand(token_embeddings.size())
          token_embeddings = token_embeddings.masked_fill(attention_mask == 0, -1e9)
          embeddings = token_embeddings.max(1).values
        elif mode.lower() == "attention":
          token_embeddings = output.last_hidden_state
          attention_mask = encodings['attention_mask'].unsqueeze(-1)
          last_attention = output.attentions[-1]
          cls_attention = last_attention[:, :, 0, :]
          weights = cls_attention.mean(dim=1)
          weights = weights * attention_mask.squeeze(-1)
          weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
          embeddings = (token_embeddings * weights.unsqueeze(-1)).sum(dim=1)
        return embeddings


    def train_triplet(self, train_dataset, loss_name, loss_kwargs, num_epochs, batch_size, mode):
        loss_function = get_loss(loss_name, **loss_kwargs)
        total_loss = 0
        num_batches = 0

        for epoch in range(num_epochs):
            self.model.train()
            data_loader = DataLoader(TripDataset(train_dataset['anchor'], train_dataset['positive'], train_dataset['negative']), batch_size=batch_size, shuffle=True)
            for sentence1_texts, sentence2_texts, sentence3_texts in tqdm(data_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False):

                self.optimizer.zero_grad()

                anchor = self.extract_embeddings(self.model, self.tokenizer, sentence1_texts, self.device, mode)
                positive = self.extract_embeddings(self.model, self.tokenizer, sentence2_texts, self.device, mode)
                negative = self.extract_embeddings(self.model, self.tokenizer, sentence3_texts, self.device, mode)
                
                loss = loss_function(anchor, positive, negative)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                avg_loss = total_loss / max(num_batches, 1)
        #print(f"Epoch {epoch+1}: Average loss = {avg_loss:.6f}")

        return self.model

    def train_base(self, train_dataset, loss_name, loss_kwargs, num_epochs, batch_size, mode):
        loss_function = get_loss(loss_name, **loss_kwargs)
        total_loss = 0
        num_batches = 0

        for epoch in range(num_epochs):
            self.model.train()
            data_loader = DataLoader(STSDataset(train_dataset['sentence1'], train_dataset['sentence2'], train_dataset['labels']), batch_size=batch_size, shuffle=True)
            for sentence1_texts, sentence2_texts, labels in tqdm(data_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False):
              
                self.optimizer.zero_grad()
                labels = labels.to(self.device)

                sentence1_embeddings = self.extract_embeddings(self.model, self.tokenizer, sentence1_texts, self.device, mode)
                sentence2_embeddings = self.extract_embeddings(self.model, self.tokenizer, sentence2_texts, self.device, mode)

                loss = loss_function(sentence1_embeddings, sentence2_embeddings, labels)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                avg_loss = total_loss / max(num_batches, 1)
            #print(f"Epoch {epoch+1}: Average loss = {avg_loss:.6f}")

        return self.model

    def train_IB(self, train_dataset, loss_name, loss_kwargs, num_epochs, batch_size, mode):
        loss_name = 'in_batch_negative_loss'
        loss_function = get_loss(loss_name, **loss_kwargs)
        total_loss_IB = 0
        total_loss_cl = 0
        total_loss = 0
        num_batches = 0
        i = 0
        beta = 0
        gamma = 0
        #beta = 0.001
        #gamma = 0.0005       

        for epoch in range(num_epochs):
            self.model.train()
            data_loader = DataLoader(STSDataset(train_dataset['sentence1'], train_dataset['sentence2'], train_dataset['labels']), batch_size=batch_size, shuffle=True)
            for sentence1_texts, sentence2_texts, labels in tqdm(data_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False):
              
                self.optimizer.zero_grad()
                labels = labels.to(self.device)

                sentence1_embeddings, mu1, logvar1, kl_loss1 = self.ib(self.extract_embeddings(self.model, self.tokenizer, sentence1_texts, self.device, mode))
                sentence2_embeddings, mu2, logvar2, kl_loss2 = self.ib(self.extract_embeddings(self.model, self.tokenizer, sentence2_texts, self.device, mode))

                BA_loss = barlow_twins_loss(sentence1_embeddings, sentence2_embeddings)
                loss = loss_function(sentence1_embeddings, sentence2_embeddings, labels) + beta * (kl_loss1 + kl_loss2) + gamma * BA_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss_IB = beta * (kl_loss1 + kl_loss2).item()
                total_loss_cl = (loss - beta * (kl_loss1 + kl_loss2)-gamma * BA_loss).item()
                total_loss_BA = gamma * BA_loss.item()
                total_loss += loss.item()
                num_batches += 1
                #avg_loss_IB = total_loss_IB / max(num_batches, 1)
                #avg_loss_cl = total_loss_cl / max(num_batches, 1)
                i = i + 1
                if (i % 50) == 0:
                  #beta = min(beta + i*0.001, 0.03)
                  print(f"\nEpoch {epoch+1}: Average loss IB = {total_loss_IB:.6f}")
                  print(f"Epoch {epoch+1}: Average loss CL = {total_loss_cl:.6f}")
                  print(f"Epoch {epoch+1}: Average loss BA = {total_loss_BA:.6f}")
                  print(f"Epoch {epoch+1}: Average loss total = {(total_loss/(i+1)):.6f}")
        return self.model
    
    def train(self, train_dataset, loss_name, loss_kwargs, num_epochs, batch_size, mode = 'attention'):
        if loss_name == "triplet":
            return self.train_triplet(train_dataset, loss_name, loss_kwargs, num_epochs, batch_size, mode)
        elif loss_name == "without_ft":
            return self.model
        elif loss_name == "IB":
            return self.train_IB(train_dataset, loss_name, loss_kwargs, num_epochs, batch_size, mode)
        else:
            return self.train_base(train_dataset, loss_name, loss_kwargs, num_epochs, batch_size, mode)

    def calculate_cosine_similarity(self, data_embeddings1, data_embeddings2):
        cosine_similarity = F.cosine_similarity(data_embeddings1, data_embeddings2, dim=1)
        return cosine_similarity

    def calculate_Spearman_rank_correlation_coefficient(self, scores, scores_actual):
        sc, _ = spearmanr(scores, scores_actual)
        return sc

    def evaluate_sts(self, test_dataset, batch_size, mode):
        self.model.eval()

        test_dataloader = DataLoader(STSDataset(test_dataset['sentence1'], test_dataset['sentence2'], test_dataset['labels']), batch_size=batch_size)
        all_embeddings1 = []
        all_embeddings2 = []
        all_labels = []

        with torch.no_grad():
            for sentences1, sentences2, labels in tqdm(test_dataloader, desc="Extracting", leave=False):
                #for every pair extract the embedding and labels append to the empty list created
                embeddings1 = self.extract_embeddings(self.model, self.tokenizer, sentences1, self.device, mode)
                embeddings2 = self.extract_embeddings(self.model, self.tokenizer, sentences2, self.device, mode)
                all_embeddings1.append(embeddings1.cpu())
                all_embeddings2.append(embeddings2.cpu())
                all_labels.append(labels.cpu())

        data_embeddings1 = torch.cat(all_embeddings1)
        data_embeddings2 = torch.cat(all_embeddings2)
        data_labels = torch.cat(all_labels)
        data_labels_np = data_labels.numpy()

        cosine_similarities = self.calculate_cosine_similarity(data_embeddings1, data_embeddings2)
        spearman = self.calculate_Spearman_rank_correlation_coefficient(cosine_similarities, data_labels_np)
        return spearman

    def evaluate_all_sts(self, test_datasets, batch_size, mode):
        self.model.eval()
        spearman_list = []
        for test_dataset in test_datasets:
            test_dataloader = DataLoader(STSDataset(test_dataset['sentence1'], test_dataset['sentence2'], test_dataset['labels']), batch_size=batch_size)
            #data_loader = DataLoader(test_dataset, batch_size=batch_size)
            all_embeddings1 = []
            all_embeddings2 = []
            all_labels = []

            with torch.no_grad():
                for sentences1, sentences2, labels in tqdm(test_dataloader, desc="Extracting", leave=False):
                    embeddings1= self.extract_embeddings(self.model, self.tokenizer, sentences1, self.device, mode)
                    embeddings2= self.extract_embeddings(self.model, self.tokenizer, sentences2, self.device, mode)
                    all_embeddings1.append(embeddings1.cpu())
                    all_embeddings2.append(embeddings2.cpu())
                    all_labels.append(labels.cpu())

            data_embeddings1 = torch.cat(all_embeddings1)
            data_embeddings2 = torch.cat(all_embeddings2)        
            data_labels = torch.cat(all_labels)
            data_labels_np = data_labels.numpy()

            cosine_similarities = self.calculate_cosine_similarity(data_embeddings1, data_embeddings2)
            spearman = self.calculate_Spearman_rank_correlation_coefficient(cosine_similarities, data_labels_np)
            spearman_list.append(spearman)
        return spearman_list

    def evaluate_sts_IB(self, test_dataset, batch_size, mode):
        self.model.eval()

        test_dataloader = DataLoader(STSDataset(test_dataset['sentence1'], test_dataset['sentence2'], test_dataset['labels']), batch_size=batch_size)
        all_embeddings1 = []
        all_embeddings2 = []
        all_labels = []

        with torch.no_grad():
            for sentences1, sentences2, labels in tqdm(test_dataloader, desc="Extracting", leave=False):
                #for every pair extract the embedding and labels append to the empty list created
                embeddings1, _, _, _ = self.ib(self.extract_embeddings(self.model, self.tokenizer, sentences1, self.device, mode), is_train = False)
                embeddings2, _, _, _ = self.ib(self.extract_embeddings(self.model, self.tokenizer, sentences2, self.device, mode), is_train = False)
                all_embeddings1.append(embeddings1.cpu())
                all_embeddings2.append(embeddings2.cpu())
                all_labels.append(labels.cpu())

        data_embeddings1 = torch.cat(all_embeddings1)
        data_embeddings2 = torch.cat(all_embeddings2)
        data_labels = torch.cat(all_labels)
        data_labels_np = data_labels.numpy()

        cosine_similarities = self.calculate_cosine_similarity(data_embeddings1, data_embeddings2)
        spearman = self.calculate_Spearman_rank_correlation_coefficient(cosine_similarities, data_labels_np)
        return spearman

    def evaluate_all_sts_IB(self, test_datasets, batch_size, mode):
        self.model.eval()
        spearman_list = []
        for test_dataset in test_datasets:
            test_dataloader = DataLoader(STSDataset(test_dataset['sentence1'], test_dataset['sentence2'], test_dataset['labels']), batch_size=batch_size)
            #data_loader = DataLoader(test_dataset, batch_size=batch_size)
            all_embeddings1 = []
            all_embeddings2 = []
            all_labels = []

            with torch.no_grad():
                for sentences1, sentences2, labels in tqdm(test_dataloader, desc="Extracting", leave=False):
                    embeddings1, _, _, _= self.ib(self.extract_embeddings(self.model, self.tokenizer, sentences1, self.device, mode))
                    embeddings2, _, _, _= self.ib(self.extract_embeddings(self.model, self.tokenizer, sentences2, self.device, mode))
                    all_embeddings1.append(embeddings1.cpu())
                    all_embeddings2.append(embeddings2.cpu())
                    all_labels.append(labels.cpu())

            data_embeddings1 = torch.cat(all_embeddings1)
            data_embeddings2 = torch.cat(all_embeddings2)        
            data_labels = torch.cat(all_labels)
            data_labels_np = data_labels.numpy()

            cosine_similarities = self.calculate_cosine_similarity(data_embeddings1, data_embeddings2)
            spearman = self.calculate_Spearman_rank_correlation_coefficient(cosine_similarities, data_labels_np)
            spearman_list.append(spearman)
        return spearman_list

    def evaluate(self, test_dataset, batch_size, dataset_name, loss_name):
        if dataset_name == 'snli':
            if loss_name != 'IB':
              return self.evaluate_all_sts(test_dataset, batch_size, mode = 'attention')
            else:
              return self.evaluate_all_sts_IB(test_dataset, batch_size, mode = 'attention')
        else:
            if loss_name !="IB":
              return self.evaluate_sts(test_dataset, batch_size, mode = 'attention')
            else:
              return self.evaluate_sts_IB(test_dataset, batch_size, mode = 'attention')

   

class InformationBottleneck(nn.Module):
    """
    Simple Variational Information Bottleneck Layer
    Input: embedding vector [batch_size, hidden_dim]
    Output: compressed representation [batch_size, bottleneck_dim]
    """
    def __init__(self, input_dim = 768, bottleneck_dim = 768, beta = 1):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.beta = beta  # Trade-off parameter: higher β = more compression
        self.logvar_min = -10
        self.logvar_max = 10
        
        # Encoder: q(z|x) ~ N(μ(x), σ²(x))
        self.fc_mu = nn.Linear(input_dim, bottleneck_dim)      # Mean
        self.fc_logvar = nn.Linear(input_dim, bottleneck_dim)  # Log variance
        
    def encode(self, x):
        """Convert input to mean and log-variance"""
        mu = self.fc_mu(x)          # μ = f_μ(x)
        logvar = self.fc_logvar(x)  # log(σ²) = f_logvar(x)
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = μ + ε·σ, ε ~ N(0, I)"""
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log(σ²))
        eps = torch.randn_like(std)     # ε ~ N(0, I)
        z = mu + eps * std             # z = μ + ε·σ
        return z
    
    def kl_divergence(self, mu, logvar):
        """Compute KL(q(z|x) || N(0, I)) = -½ Σ(1 + log(σ²) - μ² - σ²)"""
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()  # Average over batch
    
    def forward(self, x, is_train = True):
        """
        Forward pass through IB bottleneck
        
        Args:
            x: Input embeddings [batch_size, input_dim]
            return_stats: Whether to return statistics
            
        Returns:
            z: Compressed representation [batch_size, bottleneck_dim]
            If return_stats=True, also returns (mu, logvar, kl_loss)
        """
        mu, logvar = self.encode(x)       # Step 1: Get distribution parameters
        z = self.reparameterize(mu, logvar)  # Step 2: Sample z
        kl_loss = self.kl_divergence(mu, logvar)
        if is_train:
            return z, mu, logvar, self.beta * kl_loss
        else:
            return z, mu, logvar, self.beta * kl_loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def barlow_twins_loss(z1, z2, lambda_coeff=5e-3):
    # Normalize batch
    z1 = (z1 - z1.mean(0)) / z1.std(0)
    z2 = (z2 - z2.mean(0)) / z2.std(0)

    N, D = z1.size()

    # Cross-correlation
    c = torch.mm(z1.T, z2) / N

    # On-diagonal loss
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

    # Off-diagonal loss
    off_diag = off_diagonal(c).pow_(2).sum()

    loss = on_diag + lambda_coeff * off_diag
    return loss
        
