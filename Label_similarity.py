from datasets import load_dataset
from torch.utils.data import DataLoader, default_collate
from prepare_STS import STSDataset
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--datasets", )
args = parser.parse_args()
datas = args.datasets

def get_sts1_dataset(dataset_name):
    if dataset_name == 'STS-B':
        dataset_name = 'stsbenchmark'
    elif dataset_name == 'SICK-R':
        dataset_name = 'sickr'
    dataset = load_dataset(f'mteb/{dataset_name.lower()}-sts', split='test')
    dataset = dataset.rename_column('score', 'labels')
    return dataset

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


def cosine_similarity_mse(embedding1, embedding2, labels):
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
    
def run(datas):
    model_id = 'bert-base-uncased'
    model, tokenizer = get_model_tokenizer(model_id)
    model.eval()
    cosli = []
    label_list = []
    d = get_sts1_dataset(datas)
    dd = STSDataset(d['sentence1'], d['sentence2'], d['labels'])
    length = len(dd)
    sim_data_loader = DataLoader(dd, batch_size=100, shuffle=False)
    for sentence1_texts, sentence2_texts, labels in tqdm(sim_data_loader, desc="sim_cal", leave=False):
      labels = labels.to(device)

      sentence1_embeddings = extract_embedding(model, tokenizer, device, sentence1_texts)
      sentence2_embeddings = extract_embedding(model, tokenizer, device, sentence2_texts)

      cos_simi = cosine_similarity_mse(sentence1_embeddings, sentence2_embeddings, labels)
      cos_sim = cos_simi.tolist()
      cosli.append(cos_sim)
      label_list.append(labels.cpu().float().numpy())
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    flat = [item for sublist in cosli for item in sublist]
    labelss = [item for sublist in label_list for item in sublist]
    return flat, labelss

coslo, labelss = run(datas)

bin_width = 0.01
bins = np.arange(0, max(coslo) + bin_width, bin_width)

plt.hist(coslo, bins=bins, edgecolor='black')
plt.title('cosine_similarity_{datas}')
plt.xlabel('cosine_similarity_score')
plt.ylabel('Frequency')
plt.savefig(f'./datasets_distribution/cos_sim_{datas}.png')
plt.close()

bin_widths = 0.5
bins = np.arange(0, max(labelss) + bin_widths, bin_widths)

plt.hist(labelss, bins=bins, edgecolor='black')
plt.title(f'Labels_{datas}')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig(f'./datasets_distribution/labels_{datas}.png')
plt.close()