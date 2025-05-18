# LLM4Rec-IGD

This repository contains the implementation code for the paper:

**IGD: Token Decisiveness Modeling via Information Gain in LLMs for Personalized Recommendation**

## Setup and Installation

### 1. Download Datasets

```bash
# Take the book dataset as an example 
# Download the dataset
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Books.json.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Books.json.gz

# Unzip
gunzip Books.json.gz
gunzip meta_Books.json.gz
```

### 2. Create Python Environment

```bash
conda create -n IGD python=3.10
conda activate IGD
pip install -r requirements.txt
```

### 3. Preprocess Dataset

```bash
# Preprocess and extract Item-frequency information
bash compute_item_freq.sh
```

### 4. IGD-Tuning

```bash
bash ig_monitor.sh
```

#### Parameter Settings for IGD-Tuning
- `beta` adjusts the weight of zero-IG tokens.
- To implement the baseline, set `beta=1.0`.
- For our method, `beta=0.1` works well in general. You can grid search over:
  `[0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]`

### 5. IGD-Decoding

```bash
bash evaluate.sh
```

#### Parameter Settings for IGD-Decoding
- Adjust the `alpha` parameter in the `evaluate.sh` script:
  - `alpha=0.0` is the baseline.
  - In the inference script, you can set: `(0.0 0.1 0.2 0.3 0.4)`
  - `alpha=0.2` generally yields the good results.
- For D3 method, set `length penalty` to `0.0`
- For BIGRec method, set the `length penalty` to `1.0` in the script.

## Comparison Methods

### CFT Method
- Uses `cft_monitor.py`. According to the original paper, search over:
  - `beta = 0.09, 0.16, 0.29, 0.38, 0.5, 0.66, 0.9, 0.96`
  - `alpha = 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.3`

### Pos Method
- Part of the CFT method. Set `alpha=0`, and only tune `beta`.

## Hardware Notes
In our experiments, we trained our methods on an H100 96G GPU and tested on an A5000 GPU. Different hardware configurations may cause minor differences in results.