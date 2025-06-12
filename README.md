# Impact-of-loss-function-on-BERT-sentence-embedding
A sentence embedding evaluation framework that tests the impact of different loss functions (Cosine similarity mean square error, COSENT, IN-Batch Negative, Angle loss and combination of various losses) on the quality and performance of sentence embeddings on STS datasets (STS-B, STS 12 to STS 16 and STS-K).
The detail article is on overleaf project: https://www.overleaf.com/read/bzzbwqfxgrpn#1076e2

## Describtion on each python file

- Prepare_STS.py: prepare_STS.py file is the python file include build dataloader of STS datasets.
  
- Loss_Functions.py: Loss_Functions.py file is the python file include various of loss functions include default cosine similarity, COSENT, IBN and Angle losses and the combination of loss
  
- TrainSE.py: TrainSE.py file is the python file that has Training loop, evaluation function and base model.
