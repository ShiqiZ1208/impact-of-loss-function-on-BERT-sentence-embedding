import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from datasets import Value
from datasets import concatenate_datasets
from collections import defaultdict, Counter
import random
 
 
# ---------------------------------------------------------------------------
# NLI helpers
# ---------------------------------------------------------------------------
 
def remap_labels_batch(batch):
    '''
    Remaps NLI's {entailment: 0, contradiction: 2} (after filtering out neutral)
    to binary {contradiction: 0, entailment: 1}.
    '''
    batch["labels"] = [
        1 if label == 0 else 0
        for label in batch["labels"]
    ]
    return batch
 
 
def select_columns(example):
    columns_to_keep = ["sentence1", "sentence2", "labels"]
    return {col: example[col] for col in columns_to_keep}
 
 
def rename_columns(dataset, type = 'nli'):
    '''
    Standardize dataset column names via 'rename':
      - STS and SP tasks: rename columns to 'sentence1', 'sentence2', and 'labels'.
      - CL tasks: rename columns to 'text' and 'labels'.

    For SNLI and NLI datasets, labels are additionally binarized:
    0 = negative, 1 = positive.
    '''
    if type == 'nli':
        # Rename to match the sentence1/sentence2 schema used by STS datasets
        dataset = dataset.rename_column('premise', 'sentence1')
        dataset = dataset.rename_column('hypothesis', 'sentence2')
        dataset = dataset.rename_column('label', 'labels')
    elif type == 'qqp':
        dataset = dataset.rename_column('question1', 'sentence1')
        dataset = dataset.rename_column('question2', 'sentence2')
        dataset = dataset.rename_column('label', 'labels')
    elif type == 'general':
        dataset = dataset.rename_column('label', 'labels')
    elif type == 'sms_spam':
        dataset = dataset.rename_column('sms', 'text')
        dataset = dataset.rename_column('label', 'labels')

    # NLI labels arrive as ints (0=entailment, 1=neutral, 2=contradiction);
    # cast to float32 to match the label dtype used by STS datasets
    dataset = dataset.cast_column("labels", Value("float32"))

    # Drop neutral: only entailment/contradiction give an unambiguous
    # positive/negative pair, which is what contrastive training needs.
    # Follows the SimCSE paper's supervised NLI setup.
    if type == 'nli':
        dataset_filtered = dataset.filter(
            lambda batch: [label in {0, 2} for label in batch["labels"]],
            batched=True
        )

        # Remap to binary: entailment(0) -> 1 (similar), contradiction(2) -> 0 (dissimilar)
        dataset_remapped = dataset_filtered.map(remap_labels_batch, batched=True)

        return dataset_remapped
    else:
        return dataset
 
 
def Triplet_dataset(dataset_name):
    '''
    Builds (anchor, positive, negative) triplets from SNLI / MultiNLI / combined NLI.
    For each premise, every entailing hypothesis is paired with a randomly chosen
    contradicting hypothesis from the same premise. Premises with no negative
    (or no positive) are dropped.
    '''
    if dataset_name == "snli":
        dataset = load_dataset('stanfordnlp/snli', split='train')
        dataset = rename_columns(dataset)
    elif dataset_name == "multi_nli":
        dataset = load_dataset('nyu-mll/glue', 'mnli', split='train')
        dataset = rename_columns(dataset)
        # MultiNLI carries extra columns (e.g. genre, promptID) that SNLI doesn't;
        # drop everything except sentence1/sentence2/labels so the schema matches
        # SNLI's exactly (needed below for concatenate_datasets in the "nli" branch)
        dataset = dataset.map(
            select_columns,
            remove_columns=[c for c in dataset.column_names if c not in ["sentence1", "sentence2", "labels"]]
        )
    elif dataset_name == "nli":
        snli = load_dataset('stanfordnlp/snli', split='train')
        snli = rename_columns(snli)
        multinli = load_dataset('nyu-mll/glue', 'mnli', split='train')
        multinli = rename_columns(multinli)
        # Same column alignment as the multi_nli branch above -- concatenate_datasets
        # requires both datasets to have identical column schemas
        multinli = multinli.map(
            select_columns,
            remove_columns=[c for c in multinli.column_names if c not in ["sentence1", "sentence2", "labels"]]
        )
        dataset = concatenate_datasets([snli, multinli])
    else:
        raise ValueError(f"Triplet_dataset: unsupported dataset_name '{dataset_name}' "
                          f"(expected 'snli', 'multi_nli', or 'nli').")

    # Group hypotheses by premise. Labels are already binary from rename_columns:
    # 1 = entailment (positive), 0 = contradiction (negative); neutral was
    # already dropped upstream, so no other label values should appear here.
    pairs = defaultdict(lambda: {"positive": [], "negative": []})
    for data in dataset:
        if data["labels"] == 1:  # entailment
            pairs[data["sentence1"]]["positive"].append(data["sentence2"])
        elif data["labels"] == 0:  # contradiction
            pairs[data["sentence1"]]["negative"].append(data["sentence2"])

    valid_triplet = []
    for premise, data in pairs.items():
        # A triplet needs both a positive and a negative; premises missing
        # either side (e.g. only ever entailed, or only ever contradicted)
        # are silently skipped here rather than raising or logging
        if len(data["positive"]) > 0 and len(data["negative"]) > 0:
            for pos in data["positive"]:
                # Fresh random draw per positive, not once per premise --
                # so two positives from the same premise can end up with
                # different negatives, and the same negative may be reused
                neg = random.choice(data["negative"])
                valid_triplet.append({
                    "anchor": premise,
                    "positive": pos,
                    "negative": neg
                })
    print(f"Total triplets: {len(valid_triplet)}")

    return Dataset.from_list(valid_triplet)
 
 
# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
 
def get_sts_dataset(dataset_name, split='test', is_triplet=False):
    '''
    Takes a dataset_name and returns the corresponding dataset object from Hugging Face,
    including STS datasets, NLI datasets, and some evaluation datasets.
    For STS-B, it has train/validation/test splits, while the other STS datasets only have evaluation sets.
    For the SemRel dataset, it has train/validation/test splits like STS-B and is used for the cross-domain task.
    For SNLI, MultiNLI, and NLI datasets, it uses the NLI helper to return an STS-like training dataset with discrete labels.
    For other evaluation datasets, it returns the evaluation set.
    '''
    if is_triplet:
        return Triplet_dataset(dataset_name)

    if dataset_name == "snli":
        dataset = load_dataset('stanfordnlp/snli', split='train')
        return rename_columns(dataset)

    if dataset_name == "multi_nli":
        dataset = load_dataset('nyu-mll/glue', 'mnli', split='train')
        dataset = rename_columns(dataset)
        return dataset.map(
            select_columns,
            remove_columns=[c for c in dataset.column_names if c not in ["sentence1", "sentence2", "labels"]]
        )

    if dataset_name == "nli":
        snli = rename_columns(load_dataset('stanfordnlp/snli', split='train'))
        multinli = load_dataset('nyu-mll/glue', 'mnli', split='train')
        multinli = rename_columns(multinli)
        multinli = multinli.map(
            select_columns,
            remove_columns=[c for c in multinli.column_names if c not in ["sentence1", "sentence2", "labels"]]
        )
        return concatenate_datasets([snli, multinli])

    if dataset_name == 'STS-B':
        dataset = load_dataset('mteb/stsbenchmark-sts', split=split)
        return dataset.rename_column('score', 'labels')
    
    if dataset_name == "SemRel":
        dataset = load_dataset("SemRel/SemRel2024", 'eng', split=split)
        return dataset.rename_column('label', 'labels')

    if dataset_name == 'BIOSSES':
        dataset = load_dataset("mteb/biosses-sts", split="test")
        return dataset.rename_column('score', 'labels')
    
    if dataset_name == 'STS17':
        dataset = load_dataset("mteb/sts17-crosslingual-sts","en-en", split="test")
        return dataset.rename_column('score', 'labels')
    
    # STS12-16, SICK-R: test-only usage, carved into train/test by prepare_dataset
    if split != 'test':
        raise ValueError(
            f"{dataset_name} only has a usable 'test' split on the hub — "
            f"use prepare_dataset('{dataset_name}') to carve train/test out of it, "
            f"rather than requesting split='{split}' here directly."
        )
    hub_name = 'sickr' if dataset_name == 'SICK-R' else dataset_name.lower()
    dataset = load_dataset(f'mteb/{hub_name}-sts', split='test')
    return dataset.rename_column('score', 'labels')
 
def get_SP_dataset(dataset_name, val_size = 6000, test_size = 10000):
    '''
    Load a sentence-pair classification dataset.

    Uses 'load_dataset' with the given dataset name to retrieve the
    validation and test splits. The validation split is used to
    determine the decision threshold for separating the classes.
    '''
    if dataset_name == "snli":
        val_dataset = load_dataset('stanfordnlp/snli', split= f'validation[:{val_size}]')
        eval_dataset = load_dataset('stanfordnlp/snli', split= f'test[:{test_size}]')
        return rename_columns(val_dataset), rename_columns(eval_dataset)
    if dataset_name == "multi_nli":
        val_dataset_nmap = load_dataset('nyu-mll/glue', 'mnli', split=f'validation_matched[:{val_size}]')
        eval_dataset_nmap = load_dataset('nyu-mll/glue', 'mnli', split=f'test_matched[:{test_size}]')
        val_dataset = rename_columns(val_dataset_nmap)
        eval_dataset = rename_columns(eval_dataset_nmap)
        val_dataset.map(select_columns, remove_columns=[c for c in val_dataset.column_names if c not in ["sentence1", "sentence2", "labels"]])
        eval_dataset.map(select_columns, remove_columns=[c for c in eval_dataset.column_names if c not in ["sentence1", "sentence2", "labels"]])
        return val_dataset, eval_dataset
    if dataset_name == "QQP":
        val_dataset = load_dataset('nyu-mll/glue', 'qqp', split=f"train[:{val_size}]")
        eval_dataset = load_dataset('nyu-mll/glue', 'qqp', split=f"validation[:{test_size}]")
        return rename_columns(val_dataset, type = 'qqp'), rename_columns(eval_dataset, type = 'qqp')
    if dataset_name == "MRPC":
        val_dataset = load_dataset('nyu-mll/glue', 'mrpc', split=f"train[:{val_size}]")
        eval_dataset = load_dataset('nyu-mll/glue', 'mrpc', split=f"test[:{test_size}]")
        return rename_columns(val_dataset, type = 'general'), rename_columns(eval_dataset, type = 'general')
    if dataset_name == "RTE":
        val_dataset = load_dataset('nyu-mll/glue', 'rte', split=f"train[:{val_size}]")
        eval_dataset = load_dataset('nyu-mll/glue', 'rte', split=f"validation[:{test_size}]")
        return rename_columns(val_dataset, type = 'general'), rename_columns(eval_dataset, type = 'general')
         

def get_CL_dataset(dataset_name, val_size = 6000, test_size = 10000):
    '''
    Load a single sentence classification dataset.

    Uses 'load_dataset' with the given dataset name to retrieve the
    validation and test splits. The validation split is used to
    training the classification head for separating the classes.
    '''
    if dataset_name == 'MR':
        val_dataset = load_dataset('cornell-movie-review-data/rotten_tomatoes', split= f'train[:{val_size}]')
        eval_dataset = load_dataset('cornell-movie-review-data/rotten_tomatoes', split= f'test[:{test_size}]')
        return rename_columns(val_dataset, type = 'general'), rename_columns(eval_dataset, type = 'general')
    if dataset_name == 'CR':
        val_dataset = load_dataset('SetFit/CR', split= f'train[:{val_size}]')
        eval_dataset = load_dataset('SetFit/CR', split= f'test[:{test_size}]')
        return rename_columns(val_dataset, type = 'general'), rename_columns(eval_dataset, type = 'general')
    if dataset_name == 'subj':
        val_dataset = load_dataset('SetFit/subj', split= f'train[:{val_size}]')
        eval_dataset = load_dataset('SetFit/subj', split= f'test[:{test_size}]')
        return rename_columns(val_dataset, type = 'general'), rename_columns(eval_dataset, type = 'general')
    if dataset_name == 'sms_spam':
        ds = load_dataset('ucirvine/sms_spam', split= f'train[:{val_size}]')
        split = ds.train_test_split(test_size=0.2, stratify_by_column='label', seed=42)
        val_dataset, eval_dataset = split['train'], split['test']
        return rename_columns(val_dataset, type = 'sms_spam'), rename_columns(eval_dataset, type = 'sms_spam')

def load_clustering(hf_name, split='test', is_dup=False,
                    max_per_class=400, max_classes=None, seed=0):                  # NEW: two arguments
    ds = load_dataset(hf_name, split=split)
    '''
    Load a clustering dataset.

    Uses 'load_dataset' with the given dataset name to retrieve the
    test split, which is later used to evaluate clustering performance.
    '''

    # 1. merge all rows (handles both nested and flat formats)
    if isinstance(ds[0]['sentences'], list):
        sentences = [s for row in ds for s in row['sentences']]
        labels    = [l for row in ds for l in row['labels']]
    else:
        sentences, labels = list(ds['sentences']), list(ds['labels'])

    sents, labs = sentences, labels
    # 2. remove duplicate texts (keep first occurrence)
    if is_dup == True:
        seen, sents, labs = set(), [], []
        for s, l in zip(sentences, labels):
            if s not in seen:
                seen.add(s); sents.append(s); labs.append(l)

    if max_classes is not None:
        counts = Counter(map(str, labs))
        top = {c for c, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:max_classes]}
        keep = [i for i, l in enumerate(labs) if str(l) in top]
        sents = [sents[i] for i in keep]
        labs  = [labs[i] for i in keep]

    # 3. relabel: any label type (str or int) → 0..K-1
    classes = sorted(set(map(str, labs)))
    label2id = {c: i for i, c in enumerate(classes)}
    labs = [label2id[str(l)] for l in labs]

    # 4. NEW: optionally keep at most max_per_class samples per cluster
    if max_per_class is not None:
        rng = np.random.default_rng(seed)
        labs_arr = np.asarray(labs)
        keep = []
        for c in np.unique(labs_arr):
            c_idx = np.where(labs_arr == c)[0]
            if len(c_idx) > max_per_class:
                c_idx = rng.choice(c_idx, size=max_per_class, replace=False)
            keep.extend(c_idx)
        keep = np.sort(keep)
        sents = [sents[i] for i in keep]
        labs  = [labs[i] for i in keep]

    return {'sentences': sents, 'labels': labs}

def get_CT_dataset(dataset_name, max_per_class = 500, seed = None):
    if dataset_name == 'news_cluster':
        return load_clustering('mteb/twentynewsgroups-clustering', split='test', is_dup=False,
                    max_per_class=max_per_class, seed=seed)
    if dataset_name == 'reddit':
        return load_clustering('mteb/reddit-clustering', split='test', is_dup=False,
                    max_per_class=max_per_class, seed=seed)
    if dataset_name == 'biorxiv':
        return load_clustering('mteb/biorxiv-clustering-s2s', split='test', is_dup=False,
                    max_per_class=max_per_class, seed=seed)
    if dataset_name == 'stack':
        return load_clustering('mteb/stackexchange-clustering', split='test', is_dup=False,
                    max_per_class=max_per_class, seed=seed)

def get_RT_dataset(dataset_name):
    '''
    Load a retrieval dataset.

    Uses 'load_dataset' with the given dataset name to retrieve the
    test split, which is later used to evaluate retrieval performance.
    '''
    if dataset_name == 'scifact':
        corpus  = load_dataset('BeIR/scifact', "corpus",  split="corpus")
        queries = load_dataset('BeIR/scifact', "queries", split="queries")
        qrels   = load_dataset('BeIR/scifact-qrels', split="test")
        rel = defaultdict(dict)
        for r in qrels:
            if r["score"] > 0:
                rel[str(r["query-id"])][str(r["corpus-id"])] = r["score"]

        doc_ids   = [str(d) for d in corpus["_id"]]
        doc_texts = [(t + " " + x).strip() for t, x in zip(corpus["title"], corpus["text"])]
        q_pairs   = [(str(q), t) for q, t in zip(queries["_id"], queries["text"]) if str(q) in rel]

        return {"doc_ids": doc_ids, "doc_texts": doc_texts,
                "q_ids": [q for q, _ in q_pairs], "q_texts": [t for _, t in q_pairs],
                "rel": rel}
    if dataset_name == 'nfcorpus':
        corpus  = load_dataset('BeIR/nfcorpus', "corpus",  split="corpus")
        queries = load_dataset('BeIR/nfcorpus', "queries", split="queries")
        qrels   = load_dataset('BeIR/nfcorpus-qrels', split="test")
        rel = defaultdict(dict)
        for r in qrels:
            if r["score"] > 0:
                rel[str(r["query-id"])][str(r["corpus-id"])] = r["score"]

        doc_ids   = [str(d) for d in corpus["_id"]]
        doc_texts = [(t + " " + x).strip() for t, x in zip(corpus["title"], corpus["text"])]
        q_pairs   = [(str(q), t) for q, t in zip(queries["_id"], queries["text"]) if str(q) in rel]

        return {"doc_ids": doc_ids, "doc_texts": doc_texts,
                "q_ids": [q for q, _ in q_pairs], "q_texts": [t for _, t in q_pairs],
                "rel": rel}
    if dataset_name == 'arguana':
        corpus  = load_dataset('BeIR/arguana', "corpus",  split="corpus")
        queries = load_dataset('BeIR/arguana', "queries", split="queries")
        qrels   = load_dataset('BeIR/arguana-qrels', split="test")
        rel = defaultdict(dict)
        for r in qrels:
            if r["score"] > 0:
                rel[str(r["query-id"])][str(r["corpus-id"])] = r["score"]

        doc_ids   = [str(d) for d in corpus["_id"]]
        doc_texts = [(t + " " + x).strip() for t, x in zip(corpus["title"], corpus["text"])]
        q_pairs   = [(str(q), t) for q, t in zip(queries["_id"], queries["text"]) if str(q) in rel]

        return {"doc_ids": doc_ids, "doc_texts": doc_texts,
                "q_ids": [q for q, _ in q_pairs], "q_texts": [t for _, t in q_pairs],
                "rel": rel}
# ---------------------------------------------------------------------------
# Train/test preparation
# ---------------------------------------------------------------------------

# use a list for user input, give user warning if the dataset name is invalid
NLI_KEYS = ("snli", "multi_nli", "nli")
STS_BENCHMARKS = ['STS-B', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'SICK-R']
STS_VALID_NAME = STS_BENCHMARKS + ['STS17', 'BIOSSES', 'SemRel']
SENTENCE_PAIR_VALID_NAME = ['QQP', 'MRPC', 'snli', 'multi_nli', 'RTE']
CLASSIFICATION_VALID_NAME = ['MR', 'CR', 'subj', 'sms_spam']
CLUSTERING_VALID_NAME = ['news_cluster', 'reddit', 'biorxiv', 'stack']
RETRIEVAL_VALID_NAME = ['scifact', 'nfcorpus', 'arguana']

def STS_train_test_split(dataset_name, split=0.3, seed=42):
    '''
    STS12-16 and SICK-R are test-only on the hub, so they are carved into train/test.
    A fixed seed guarantees both public functions see the same partition
    (otherwise the eval split could overlap the train split).
    '''
    dataset = get_sts_dataset(dataset_name, split='test')
    return dataset.train_test_split(test_size=split, seed=seed)


def prepare_train_dataset(dataset_name, split=0.3, seed=42):
    '''
    Returns (train_dataset, val_dataset) for the given dataset name.
    val_dataset is used only for best-checkpoint selection; [] means no validation set.
    - NLI / SNLI / MultiNLI: the NLI training data, no validation set.
    - STS:       STS-B train, STS-B validation.
    - SemRel:    SemRel train, SemRel dev.
    - STS-B:     STS-B train, no validation set.
    - STS-ablation: STS-B train, no validation set.
    - STS12-16, SICK-R: the carved train portion, no validation set.
    '''
    is_triplet = False
    if dataset_name =="triplet":
        is_triplet = True
    if dataset_name in NLI_KEYS:
        return get_sts_dataset(dataset_name, is_triplet=is_triplet), []
 
    if dataset_name == "STS":
        return (get_sts_dataset('STS-B', split='train'),
                get_sts_dataset('STS-B', split='validation'))
 
    if dataset_name == "SemRel":
        return (get_sts_dataset('SemRel', split='train'),
                get_sts_dataset('SemRel', split='dev'))
 
    if dataset_name in ('STS-B', 'STS-ablation'):
        return get_sts_dataset('STS-B', split='train'), []
 
    return STS_train_test_split(dataset_name, split=split, seed=seed)['train'], []


def prepare_eval_datasets(names=STS_BENCHMARKS):
    '''Return {name: test_dataset}. Defaults to the seven STS benchmarks.'''
    for n in names:
        if n not in STS_VALID_NAME:
            raise ValueError(f"Unknown STS eval dataset: {n}. Choose from {STS_VALID_NAME}")
    return {n: get_sts_dataset(n, split='test') for n in names}

def prepare_SP_eval_datasets(names = ['QQP']):
    for n in names:
        if n not in SENTENCE_PAIR_VALID_NAME:
            raise ValueError(f"Unknown Sentence pair eval dataset: {n}. Choose from {SENTENCE_PAIR_VALID_NAME}")
    return {n: get_SP_dataset(n) for n in names}

def prepare_CL_eval_datasets(names = ['MR']):
    for n in names:
        if n not in CLASSIFICATION_VALID_NAME:
            raise ValueError(f"Unknown Sentence pair eval dataset: {n}. Choose from {CLASSIFICATION_VALID_NAME}")
    return {n: get_CL_dataset(n) for n in names}

def prepare_CT_eval_datasets(names = ['news_cluster']):
    for n in names:
        if n not in CLUSTERING_VALID_NAME:
            raise ValueError(f"Unknown Sentence pair eval dataset: {n}. Choose from {CLUSTERING_VALID_NAME}")
    return {n: get_CT_dataset(n) for n in names}

def prepare_RT_eval_datasets(names = ['scifact']):
    for n in names:
        if n not in RETRIEVAL_VALID_NAME:
            raise ValueError(f"Unknown Sentence pair eval dataset: {n}. Choose from {CLUSTERING_VALID_NAME}")
    return {n: get_RT_dataset(n) for n in names}


class STSDataset(torch.utils.data.Dataset):
    '''
    A class for STSDataset.
    Has __len__ and __getitem__ for the training process.
    '''
    def __init__(self, sentence1, sentence2, label):
        self.label = label
        self.sentence1 = sentence1
        self.sentence2 = sentence2
 
    def __len__(self):
        return len(self.label)
 
    def __getitem__(self, idx):
        return self.sentence1[idx], self.sentence2[idx], self.label[idx]
 
 
class TripDataset(torch.utils.data.Dataset):
    '''
    A class for triplet Dataset.
    Has __len__ and __getitem__ for the training process.
    '''
    def __init__(self, anchor, positive, negative):
        self.anchor = anchor
        self.positive = positive
        self.negative = negative
 
    def __len__(self):
        return len(self.anchor)
 
    def __getitem__(self, idx):
        return self.anchor[idx], self.positive[idx], self.negative[idx]
    

class CLDataset(torch.utils.data.Dataset):
    '''
    A class for CLDataset.
    Has __len__ and __getitem__ for the validation process.
    '''
    def __init__(self, text, label):
        self.label = label
        self.text = text
 
    def __len__(self):
        return len(self.label)
 
    def __getitem__(self, idx):
        return self.text[idx], self.label[idx]

class RTDataset(torch.utils.data.Dataset):
    '''
    A class for retrieval texts (queries or documents).
    Returns only the text; relevance labels are handled separately via qrels.
    '''
    def __init__(self, text):
        self.text = text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx]

