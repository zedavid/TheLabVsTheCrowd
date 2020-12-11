# TheLabVsTheCrowd
Code and data for the paper "[The Lab vs The Crowd: An Investigation into DataQuality for Neural Dialogue Models](https://arxiv.org/abs/2012.03855)", accepted at the [NeurIPS 2020 workshop on Human-in-the-Loop Dialogue Systems](https://sites.google.com/view/hlds-2020).

## Install

The code is written in Python 3.6.

```shell script
# 1. Clone repo
git clone https://github.com/zedavid/TheLabVsTheCrowd.git && cd TheLabVsTheCrowd

# 2. Create virtual env
python3.6 -m venv .venv

# 3. Activate it
source .venv/bin/activate

# 4. Install packages
pip install -r requirements.txt

# 5. Set permissions
chmod +x src/python/*.sh
```

It was developed and run in Ubuntu 18.04 with Tensorflow-GPU 1.14 and CUDA 10.2 (driver version 440). Anaconda may be more suitable if using a different configuration.

## Data 

- in-lab: data colleced in lab
    - lab: partitioning for experiments with lab data
- mturk: data collected on mturk
    - mturk_full: partitioning for experiements with the full mturk dataset (147 dialogues)
    - mturk_partial: partitioning for experiments with the partial mturk dataset (63 dialogues)
    - mturk_tested_on_lab: partitioning for the experiments where mturk is used for training and lab for testing
- todbert_data: data formateed to be used with the [ToD-BERT](https://github.com/jasonwu0731/ToD-BERT) code

## Feature extraction

The script to execute is `src/python/extract_feat_batch.sh`:

```./extract_feat_batch.sh <train_dir:lab|mturk_full|mturk_partial|mturk_tested_on_lab> <path_to_embeddings> <test_dir:lab>```

Google News embeddigns can be downloaded from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit).

## Model training

### Cross-validation experiments

The script to execute is `src/python/train_x_val_batch.sh`:

```./train_x_val_batch.sh <train|eval> <lab|mturk_full|mturk_partial|>```

### Cross-data experiments

The script to execute is `src/python/train_batch.sh`:

```./train_batch.sh <train|eval> mturk_tested_on_lab <number_of_runs> ../../data/in-lab/```

- train: does not compute perplexity
- eval: uses an n-gram model for dialogue acts to compute the dialogue state sequence perplexity
