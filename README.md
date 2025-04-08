# Splits!  
## A Flexible Dataset for Evaluating a Model’s Demographic Social Inference

This repository contains the resources described in the **Splits!** paper, including:

- Demographic-split corpora  
- Demographic-topic-split corpora  
- The evaluation method proposed in the paper

---

## 📊 Dataset Versions (Summary)

| Version | Size | # Posts / Instances | Demographic Labels | Topic Labels | Task-Ready Format |
|--------|------|----------------------|---------------------|--------------|-------------------|
| 1. All Seed Users | ~115 GB | 350M posts | ✅ | ❌ | ❌ |
| 2. High-Groupness Users | ~34 GB | 90M posts | ✅ (refined) | ❌ | ❌ |
| 3. High-Groupness by Topic | ~2.6 GB | 3.6M posts | ✅ (refined) | ✅ | ❌ |
| 4. Splits! | ~3.9 GB | 177K instances | ✅ (refined) | ✅ | ✅ |

See [DETAILS.md](DETAILS.md) for group-ness thresholds, sampling procedure, and detailed schema.

---

## 📦 Dataset Versions

### 1. Posts from All Seed Users (by Demographic)

Includes **all posts** made by any user who has posted in a labeled seed subreddit.  
High-recall, low-precision demographic labels.

---

### 2. Posts from High-Groupness Seed Users (by Demographic)

Refines (1) using a **group-ness** metric to select users more likely to belong to a demographic group.  
Higher precision, decent recall. Group-ness thresholds in [DETAILS.md](docs/DETAILS.md).

---

### 3. Posts from High-Groupness Seed Users (by Demographic and Topic)

Builds on (2) by organizing posts into **neutral topics** using BM25 retrieval.  
Useful for studying **topic-based differences** across groups.

---

### 4. Splits! Dataset for Group Theorization Task

Final version for the **Group Theorization** task.  
Includes matched sets of posts, topic descriptions, and a balanced, anonymized calibration set.

---

## ⚙️ Setup

Install the `datasets` package (via HuggingFace):
```bash
pip install datasets
```

---

## 📥 Data Download

All versions are hosted at [`ecaplan/splits`](https://huggingface.co/datasets/ecaplan/splits):

```python
from datasets import load_dataset

v1 = load_dataset("ecaplan/splits", "all_seed_user_posts")['train']
v2 = load_dataset("ecaplan/splits", "high_groupness_user_posts")['train']
v3 = load_dataset("ecaplan/splits", "high_groupness_by_topic")['train']
v4 = load_dataset("ecaplan/splits", "splits")['train']
```

⚠️ Versions (1) and (2) are large—be sure to set your `HF_HOME` to a drive with enough space!

---

## 🚀 Quickstart

1. Install dependencies  
2. Load your dataset version with `load_dataset(...)`  
3. Run the demo notebook in `tutorial.ipynb` to evaluate your model  

---

## 📄 Citation

If you use this dataset, please cite the paper:

```
[Coming soon!]
```
