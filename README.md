# Impact-of-loss-function-on-BERT-sentence-embedding
A sentence embedding evaluation framework that tests the impact of different loss functions (Cosine similarity mean square error, COSENT, IN-Batch Negative, Angle loss and combination of various losses) on the quality and performance of sentence embeddings on STS datasets (STS-B, STS 12 to STS 16 and STS-K).
The detail article is on overleaf project: https://www.overleaf.com/read/bzzbwqfxgrpn#1076e2


## Describtion on each python file

- Prepare_STS.py: prepare_STS.py file is the python file include build dataloader of STS datasets.
  
- Loss_Functions.py: Loss_Functions.py file is the python file include various of loss functions include default cosine similarity, COSENT, IBN and Angle losses and the combination of loss
  
- TrainSE.py: TrainSE.py file is the python file that has Training loop, evaluation function and base model.

## How to run the code
Follow the steps below to set up and run the project:

- clone the repo from github then directory to the repo
```bash
# Step 1: Clone the repository
git clone https://github.com/ShiqiZ1208/impact-of-loss-function-on-BERT-sentence-embedding.git
cd /impact-of-loss-function-on-BERT-sentence-embedding
```

- runing the Train.py code, -b refered to how many batch_size during fine-tune the model, -r refered to how many runs to average in order to get final result, -e refered to total eopchs takes during fine-tune the model.
```bash
# Step 2: run the code
python TrainSE.py -b BATCH_SIZE -r TOTAL_RUN -e TOTAL EPOCHS_DURING_FINETUNE
```
## Performance Graph

![Model Performance](result.png)

## Example of Datasets
-STS-B
```text
the detial infromation of STS-B datasets:
(Dataset({
    features: ['split', 'genre', 'dataset', 'year', 'sid', 'labels', 'sentence1', 'sentence2'],
    num_rows: 1379
})
```

-STS12
```text
the detial infromation of STS12 datasets:
(Dataset({
    features: ['split', 'sentence1', 'sentence2', 'labels'],
    num_rows: 3108
})
```

-STS13
```text
the detial infromation of STS13 datasets:
(Dataset({
    features: ['split', 'sentence1', 'sentence2', 'labels'],
    num_rows: 1500
})
```

-STS14
```text
the detial infromation of STS14 datasets:
(Dataset({
    features: ['split', 'sentence1', 'sentence2', 'labels'],
    num_rows: 3750
})
```

-STS15
```text
the detial infromation of STS15 datasets:
(Dataset({
    features: ['split', 'sentence1', 'sentence2', 'labels'],
    num_rows: 3000
})
```

-STS16
```text
the detial infromation of STS16 datasets:
(Dataset({
    features: ['split', 'sentence1', 'sentence2', 'labels'],
    num_rows: 1186
})
```

-SICK-R
```text
the detial infromation of SICK-R datasets:
(Dataset({
    features: ['sentence1', 'sentence2', 'labels'],
    num_rows: 9927
})
```

## Base model BERT-uncase-base
-structure in huggingface
```text
BertModel(
  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(30522, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): BertEncoder(
    (layer): ModuleList(
      (0-11): 12 x BertLayer(
        (attention): BertAttention(
          (self): BertSdpaSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
  (pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
)
```
