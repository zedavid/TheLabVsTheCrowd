# TheLabVsTheCrowd
Code and data for the paper "[The Lab vs The Crowd: An Investigation into DataQuality for Neural Dialogue Models](https://drive.google.com/file/d/1moVfRjmFD9JfBnjLz1P8xW3p9r9KwI_j/view?usp=sharing)", accepted at the [NeurIPS 2020 workshop on Human-in-the-Loop Dialogue Systems](https://sites.google.com/view/hlds-2020).

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


## Feature extraction

The script to execute is `src/python/extract_feat_batch.sh`:

```shell script
./extract_feat_batch.sh <lab/mturk_full/mturk_partial> <path_to_embeddings>
```

Google News embeddings can be downloaded from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit).
