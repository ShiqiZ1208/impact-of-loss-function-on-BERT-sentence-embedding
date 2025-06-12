# Impact-of-loss-function-on-BERT-sentence-embedding
A sentence embedding evaluation framework that tests the impact of different loss functions (Cosine similarity mean square error, COSENT, IN-Batch Negative, Angle loss and combination of various losses) on the quality and performance of sentence embeddings on STS datasets (STS-B, STS 12 to STS 16 and STS-K).
The detail article is on overleaf project: https://www.overleaf.com/read/bzzbwqfxgrpn#1076e2

## 📈 Performance Graph

![Model Performance](result.png)

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
