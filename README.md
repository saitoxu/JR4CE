# JR4CE

## Abstract

The expanding online job market and the impact of recommender systems highlight the need for practical job recommendations.
While diversity is crucial, job recommendations uniquely require high-quality diversity to accelerate career exploration rather than just maximizing quantitative metrics.
Existing systems often fall short, as high diversity scores do not always align with user preferences, resulting in poor exploration and user dissatisfaction.
To address this, we propose Job Recommendation for Career Exploration (JR4CE), which uses user-job interaction data, explicit preferences, and current user information to achieve high-quality diversity.
JR4CE has three main modules: (1) Knowledge Graph Learning Module represents explicit preferences and current information as graphs and learns latent representations of users and jobs.
(2) Collaborative Filtering Module uses interaction data to refine the latent representations from (1) and predicts the likelihood of a target user’s job application.
(3) Diversity Data Augmentation Module performs data augmentation using the interaction data of role model users for a target user to enhance diversity, thereby accelerating career exploration.
Experimental results using our datasets from an actual job search website show that JR4CE outperforms several state-of-the-arts in both recommendation accuracy and diversity.
Specifically, JR4CE effectively works for users in the early stages of career exploration activities.

## Usage

### Requirements

- [pyenv](https://github.com/pyenv/pyenv)
- [Poetry](https://github.com/python-poetry/poetry)
- You need to install python (>=3.9 and <3.10) via pyenv in advance.

### Setup

```sh
$ poetry env use 3.9.6 # please specify your python version
$ poetry install
```

### Training

```sh
$ poetry run python -m JR4CE.train
```

You can see the usage by the following command.

```sh
$ poetry run python -m JR4CE.train -h
usage: train.py [-h] [--seed SEED] [--dataset [DATASET]] [--data_path [DATA_PATH]] [--dim DIM] [--epoch EPOCH] [--batch_size BATCH_SIZE] [--lr LR]
                [--patience PATIENCE] [--Ks [KS]] [--val_interval VAL_INTERVAL] [--save_path [SAVE_PATH]] [--model_path [MODEL_PATH]] [--neg_size NEG_SIZE]
                [--div_lambda DIV_LAMBDA] [--threshold_user THRESHOLD_USER] [--threshold_item THRESHOLD_ITEM] [--num_gcn_layer NUM_GCN_LAYER]
                [--kgl_module KGL_MODULE] [--cf_module CF_MODULE] [--use_edge_type USE_EDGE_TYPE]

Run JR4CE.

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           Random seed.
  --dataset [DATASET]   Choose a dataset from {glit2021, glit2022}.
  --data_path [DATA_PATH]
                        Input data path.
  --dim DIM             Number of dimension.
  --epoch EPOCH         Number of epoch.
  --batch_size BATCH_SIZE
                        Batch size.
  --lr LR               Learning rate.
  --patience PATIENCE   Number of epoch for early stopping.
  --Ks [KS]             Calculate metric@K when evaluating.
  --val_interval VAL_INTERVAL
                        Validation interval.
  --save_path [SAVE_PATH]
                        Model path for saving.
  --model_path [MODEL_PATH]
                        Model path for evaluation.
  --neg_size NEG_SIZE   Negative sampling size.
  --div_lambda DIV_LAMBDA
                        Lambda for diversity loss.
  --threshold_user THRESHOLD_USER
                        Similarity threshold for user.
  --threshold_item THRESHOLD_ITEM
                        Disimilarity threshold for item.
  --num_gcn_layer NUM_GCN_LAYER
                        Number of GCN layers.
  --kgl_module KGL_MODULE
                        Knowledge graph learning module.
  --cf_module CF_MODULE
                        Collaborative filtering module.
  --use_edge_type USE_EDGE_TYPE
                        Whether to use edge type information in GAT.
```

### Evaluation

```sh
$ poetry run python -m JR4CE.test --model_path trained_model/best.pth # please specify your model path
```

## Dataset

Due to privacy and business restrictions, we cannot release our dataset right now.
You can adapt our code for your own dataset with the following dataset format.
To use our code, the following five types of files are required.

### train.txt

Interaction data for training.

```
<user_id> <item_id> <item_id> ...
...
```

### val.txt

Interaction data for validation.

```
<user_id> <item_id>
...
```

### test.txt

Interaction data for testing.
The format is the same as `val.txt`.

### kg.txt

Knowledge graph data.

```
<head_entity_id> <relation_id> <tail_entity_id>
...
```

### info.txt

Information data for users, items, and knowledge graph entities.

```
<user_size> <item_size> <item_size>
<item_relation_id> <item_relation_id> ...
<user_preference_relation_id> <user_preference_relation_id> ...
<user_current_relation_id> <user_current_relation_id> ...
```

## Baseline Methods (MMR/DPP)

This repository also includes implementations of baseline reranking methods: MMR (Maximum Marginal Relevance) and DPP (Determinantal Point Process).

To evaluate using MMR or DPP, you first need to train a JR4CE model and extract embeddings, or prepare your own embeddings.

### Usage

```sh
# MMR
$ poetry run python -m reranker.test \
    --dataset <dataset_name> \
    --model mmr \
    --user_embeddings_path <path_to_user_embeddings> \
    --item_embeddings_path <path_to_item_embeddings> \
    --lambda_factor 0.7

# DPP
$ poetry run python -m reranker.test \
    --dataset <dataset_name> \
    --model dpp \
    --user_embeddings_path <path_to_user_embeddings> \
    --item_embeddings_path <path_to_item_embeddings> \
    --temperature 1.0
```

### Arguments

```sh
$ poetry run python -m reranker.test -h
usage: test.py [-h] [--dataset [DATASET]] [--data_path [DATA_PATH]] [--Ks [KS]] [--seed SEED]
               [--model MODEL] [--lambda_factor LAMBDA_FACTOR] [--temperature TEMPERATURE]
               [--user_embeddings_path USER_EMBEDDINGS_PATH] [--item_embeddings_path ITEM_EMBEDDINGS_PATH]
               [--use_parallel] [--n_workers N_WORKERS]

Run MMR/DPP reranking.

optional arguments:
  -h, --help            show this help message and exit
  --dataset [DATASET]   Choose a dataset from {glit2021, glit2022}.
  --data_path [DATA_PATH]
                        Input data path.
  --Ks [KS]             Calculate metric@K when evaluating.
  --seed SEED           Seed for ranking model.
  --model MODEL         Model name (mmr or dpp).
  --lambda_factor LAMBDA_FACTOR
                        Lambda factor for MMR (default: 0.7).
  --temperature TEMPERATURE
                        Temperature for DPP (default: 1.0).
  --user_embeddings_path USER_EMBEDDINGS_PATH
                        Path to user embeddings file (.pt).
  --item_embeddings_path ITEM_EMBEDDINGS_PATH
                        Path to item embeddings file (.pt).
  --use_parallel        Use parallel processing for reranking.
  --n_workers N_WORKERS
                        Number of workers for parallel processing.
```

### Embedding Format

User and item embeddings should be saved as PyTorch tensors:
- `user_embeddings.pt`: Tensor of shape `(user_size, dim)`
- `item_embeddings.pt`: Tensor of shape `(item_size, dim)`

## Citation

WIP
