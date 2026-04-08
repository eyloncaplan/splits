<div align="center">

<h1>
  <img src="other/cat_splits_.png" width="50" style="vertical-align: middle; margin-right: 0px;">
  Splits! Flexible Sociocultural Linguistic Investigation at Scale
</h1>

[![arxiv](https://img.shields.io/badge/arXiv-2504.04640-red)](https://arxiv.org/abs/2504.04640)

</div>

This repository provides the code and data necessary to quickly get started with the **Splits!** paper. It allows users to quickly explore the sociocultural linguistic differences across demographics and topics using our Lift and Triviality metrics.

---

## 📊 Dataset Versions (Summary)

| Version | Size | # Posts / Instances | Demographic Labels | Topic Labels |
|--------|------|----------------------|---------------------|--------------|
| 1. All Seed Users | ~115 GB | 350M posts | ✅ | ❌ |
| 2. High-Groupness Users | ~34 GB | 90M posts | ✅ (less noisy) | ❌ |
| 3. Splits! | ~2.6 GB | 3.6M posts | ✅ (less noisy) | ✅ |

See [DETAILS.md](DETAILS.md) for group-ness thresholds, sampling procedure, and detailed schema.

---

## 📦 Dataset Versions

### 1. Posts from All Seed Users (by Demographic)

Includes **all posts** made by any user who has posted in a labeled seed subreddit.  
High-recall, low-precision demographic labels.

---

### 2. Posts from High-Groupness Seed Users (by Demographic)

Refines (1) using a **group-ness** metric to select users more likely to belong to a demographic group.  
Higher precision, decent recall. Group-ness thresholds in [DETAILS.md](DETAILS.md).

---

### 3. Splits!

Builds on (2) by organizing posts into **topics** using ColBERT retrieval. Useful for studying **topic-based differences** across groups.

---

## 🗂️ Metadata

The full demographic subreddit seed sets, self-ID phrases, and anti-self-ID phrases can be found in `metadata/demographics.json`. All topics and topic keywords can be found in `metadata/topics.json`.

---

## ⚙️ Environment Setup

We use `conda` to provision the base Python and Java (OpenJDK) requirements, and `uv` for fast Python dependency installation.

1. **Create the Conda environment**:
   ```bash
   conda create -y -n splits_demo python=3.10 openjdk uv
   ```

2. **Activate the environment**:
   ```bash
   conda activate splits_demo
   ```

3. **Install dependencies using `uv pip`**:
   ```bash
   uv pip install -r requirements.txt
   ```

---

## 📥 Data Download

Run the downloaded data script to get the demo data into `lexica/`:

```bash
./download_data.sh
```

To access the original full variants of the data remotely directly via HF, you can run the following:

```python
from datasets import load_dataset

v1 = load_dataset("ecaplan/splits", "all_seed_user_posts")['train']
v2 = load_dataset("ecaplan/splits", "high_groupness_user_posts")['train']
v3 = load_dataset("ecaplan/splits", "high_groupness_by_topic")['train']
```

---

## 🚀 Running the Demo

To run the Lift and Triviality metrics, you just need to run the `splits_metrics_demo.ipynb` notebook! 

---

## 📄 Citation

If you use this dataset, please cite the paper:

```
@misc{caplan2025splitsflexibledatasetevaluating,
      title={Splits! A Flexible Dataset for Evaluating a Model's Demographic Social Inference}, 
      author={Eylon Caplan and Tania Chakraborty and Dan Goldwasser},
      year={2025},
      eprint={2504.04640},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.04640}, 
}
```
