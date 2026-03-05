from datasets import load_dataset
from torch.utils.data import DataLoader, default_collate
from datapreprocess import get_sts_dataset, STSDataset
from lossfunc import divided_by_maximum
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

# check the device use cuda if possible
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def extract_embedding(model, tokenizer, device, sentences, to_numpy=False):
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


def cosine_similarity_mse(embedding1, embedding2):
    '''
    extract cosine similarity score from two embeddings
    input: embedding1, embedding2
    return: cosine_sim score
    '''
    #cosine similarity between the pairs of embeddings using torch.nn
    cos_sim = F.cosine_similarity(embedding1, embedding2)
    return cos_sim


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

def get_mean_from_dataset(datas, model, tokenizer):
    dataset = get_sts_dataset(datas)
    train_labels = np.array(dataset["labels"])
    train_labels_means = (train_labels.mean())/5

    label_list = []
    dd = STSDataset(dataset['sentence1'], dataset['sentence2'], dataset['labels'])
    length = len(dd)
    sim_data_loader = DataLoader(dd, batch_size=10, shuffle=False)
    with torch.no_grad():
      for sentence1_texts, sentence2_texts, labels in tqdm(sim_data_loader, desc="sim_cal", leave=False):
          labels = labels.to(device)

          sentence1_embeddings = extract_embedding(model, tokenizer, device, sentence1_texts)
          sentence2_embeddings = extract_embedding(model, tokenizer, device, sentence2_texts)

          cos_simi = F.cosine_similarity(sentence1_embeddings, sentence2_embeddings)
          cos_sim = cos_simi.detach().cpu()
          label_list.append(cos_sim)
          #label_list.append(labels.detach().cpu().float().numpy())
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    all_labels = np.concatenate(label_list)
    mean_diff = all_labels.mean() - train_labels_means
    #print(mean_diff)

    return mean_diff
    
def generate_distribution(model, tokenizer, datas, loss_name, display_label = False, is_graph =True):
    '''
    generate distribution graph using matplot
    input: model, tokenizer, data, spearman, loop num. loss name, display_label = False(if true will generate true label distribution)
    '''
    is_snli = False
    if datas == "snli":
       datas = "SICK-R"
       is_snli = True
    if display_label == False:
        model = model
        tokenizer = tokenizer
        model.eval()
        cosli = []
        d = get_sts_dataset(datas)
        dd = STSDataset(d['sentence1'], d['sentence2'], d['labels'])
        length = len(dd)
        sim_data_loader = DataLoader(dd, batch_size=10, shuffle=False)
        with torch.no_grad():
            for sentence1_texts, sentence2_texts, labels in tqdm(sim_data_loader, desc="sim_cal", leave=False):
              labels = labels.to(device)

              sentence1_embeddings = extract_embedding(model, tokenizer, device, sentence1_texts)
              sentence2_embeddings = extract_embedding(model, tokenizer, device, sentence2_texts)

              cos_simi = cosine_similarity_mse(sentence1_embeddings, sentence2_embeddings)
              cos_sim = cos_simi.detach().tolist()
              cosli.append(cos_sim)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Check if the directory exists
        directory = "datasets_disttest"
        if not os.path.exists(directory):
          os.makedirs(directory)  # Create the directory (including parent dirs if needed)
          print(f"Created directory: {directory}")

        coslo = [item for sublist in cosli for item in sublist]
        coslo_array = np.array(coslo)
        if is_snli:
          np.save(f"{directory}/cos_sim_{datas}_{loss_name}_snli.npy", coslo_array)
        else:
          np.save(f"{directory}/cos_sim_{datas}_{loss_name}.npy", coslo_array)
        bin_width = 0.05
        bins = np.arange(0, max(coslo) + bin_width, bin_width)


        if is_graph == True:
          plt.hist(coslo, bins=bins, edgecolor='black')
          plt.title(f'cosine_similarity_{datas} with {loss_name}')
          plt.xlabel('cosine_similarity_score')
          plt.ylabel('Frequency')
          plt.savefig(f"{directory}/{loss_name}_{datas}.png")
          plt.close()
        else:
          print("without graph")
    else:
        directory = "datasets_distribution"
        # Check if the directory exists
        if not os.path.exists(directory):
          os.makedirs(directory)  # Create the directory (including parent dirs if needed)
          #print(f"Created directory: {directory}")
          
        label_list = []
        d = get_sts_dataset(datas)
        dd = STSDataset(d['sentence1'], d['sentence2'], d['labels'])
        length = len(dd)
        sim_data_loader = DataLoader(dd, batch_size=60, shuffle=False)
        with torch.no_grad():
            for sentence1_texts, sentence2_texts, labels in tqdm(sim_data_loader, desc="sim_cal", leave=False):
                labels = divided_by_maximum(labels)
                label_list.append(labels.detach().cpu().float().numpy())
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        coslo = [item for sublist in label_list for item in sublist]
        coslo_array = np.array(coslo)
        np.save(f"{directory}/label_{datas}.npy", coslo_array)
        bin_width = 0.05
        bins = np.arange(0, max(coslo) + bin_width, bin_width)


        if is_graph == True:
          plt.hist(coslo, bins=bins, edgecolor='black')
          plt.title(f"{datas}'s distributions")
          plt.xlabel('label score')
          plt.ylabel('Frequency')
          plt.savefig(f'./datasets_distribution/{datas} no nom label distribution.png')
          plt.close()
