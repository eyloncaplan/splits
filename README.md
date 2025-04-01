# Splits!  
## A Flexible Dataset for Evaluating a Model’s Demographic Social Inference

This repository contains the resources described in the **Splits!** paper, including:

- Demographic-split corpora  
- Demographic-topic-split corpora  
- The evaluation method proposed in the paper  

---

## 📦 Dataset Versions

There are **four versions** of the dataset. The best version for you depends on your use case. The versions are listed in order from **least** to **most processed**.

---

### 1. Posts from All Seed Users (by Demographic)

This version uses a **very loose** definition of group membership. If a user has ever posted in one of our labeled seed subreddits, they are considered a *seed user*. This dataset contains **all posts** made by all seed users across **all subreddits** (not just seed subreddits!), labeled by demographic group.

Think of this as a **high-recall, low-precision** dataset of posts labeled by demographic.

`[specifics]`

---

### 2. Posts from High-Groupness Seed Users (by Demographic)

This version refines (1) by using our **group-ness** metric to select users more likely to belong to a given demographic. We determine thresholds by analyzing self-identification vs. anti-self-identification rates and selecting the point where they diverge significantly.

**Group-ness thresholds (percentile cutoff):**
- Teacher: 75  
- Catholic: 75  
- Black: 75  
- Construction Worker: 90  
- Jewish: 90  
- Hindu/Jain/Sikh: 80  

This results in **decent recall and higher precision** compared to version (1).

`[specifics]`

---

### 3. Posts from High-Groupness Seed Users (by Demographic **and** Topic)

This version takes the data from (2) and organizes it into **neutral topics**. Each demographic group’s posts are categorized into:

- **10 neutral topic categories**
- **20 specific topics per category** (200 total topics)
- Topics selected via **BM25 retrieval**

Useful for investigating **topic-based differences** in discourse across demographics.

Topic categories and keywords are listed in `[specifics]`.

---

### 4. Splits! Dataset for Group Theorization Task

This is the final version, formatted specifically for the **Group Theorization task**.

Each instance includes:
- Inputs to the user-implemented function `τ`:  
  `(d_A, d_B, t, u)`  
- Inputs to the evaluator module:  
  `(S₁, S₂)`

Samples are drawn from version (3), matched across demographics with **no duplicates**.

Full sampling and formatting details are available in the paper.

`[specifics]`

---

## ⚙️ Setup

## 📥 Data Download

## 📊 Evaluate Your Model/Framework’s Group Theorization!

---

Let me know if you want to add badges, links, or examples in this README too!
