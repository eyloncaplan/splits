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

### 1. Posts from All Seed Users (by Demographic) ~ 115 GB, 350 million posts

This version uses a **very loose** definition of group membership. If a user has ever posted in one of our labeled seed subreddits, they are considered a *seed user*. This dataset contains **all posts** made by all seed users across **all subreddits** (not just seed subreddits!), labeled by demographic group. Think of this as a **high-recall, low-precision** dataset of posts labeled by demographic.

**Features:**
- `id`: Unique identifier for the post  
- `user`: Reddit username of the author  
- `timestamp`: Unix timestamp (in milliseconds) of when the post was created  
- `text`: The body of the post  
- `demographic`: The demographic label assigned to the post  
- `subreddit`: The subreddit where the post appeared

---

### 2. Posts from High-Groupness Seed Users (by Demographic) ~ 34 GB, 90 million posts

This version refines (1) by using our **group-ness** metric to select users more likely to belong to a given demographic. We determine thresholds by analyzing self-identification vs. anti-self-identification rates and selecting the point where they diverge significantly. Think of this as a **decent recall** and **higher precision** version of (1).

**Group-ness thresholds (percentile cutoff):**
- Teacher: 75  
- Catholic: 75  
- Black: 75  
- Construction Worker: 90  
- Jewish: 90  
- Hindu/Jain/Sikh: 80  

**Features:**

Identical to (1), but adds
- `metric_percentile`: Percentile for group-ness of the user within their demographic.

---

### 3. Posts from High-Groupness Seed Users (by Demographic **and** Topic) ~ 2.6 GB, 3.6 million posts

This version builds on (2) by organizing each demographic group's posts into **neutral topics**. It's designed to help analyze how different groups discuss a shared set of themes.

**Structure:**
- **10 neutral topic categories**
- **20 specific topics per category** (200 total topics)
- Topics selected using **BM25 retrieval**

This version is especially useful for exploring **topic-based differences in discourse across demographics**. Topic categories, specific topics, and retrieval keywords are available in `[specifics]`.

**Features:**
- `id`: Unique identifier for the post
- `description`: The specific neutral topic the post was retrieved for
- `demographic`: The demographic group assigned to the post
- `content`: The text of the post
- `metadata`: A dictionary containing post metadata:
  - `timestamp`: When the post was created
  - `score`: Reddit score (upvotes - downvotes)
  - `subreddit`: The subreddit where the post appeared
  - `user`: The username of the one who posted
- `score`: BM25 similarity score between the post and the topic keywords

---

### 4. Splits! Dataset for Group Theorization Task ~ 3.9 GB, 177 million dataset instances

This is the final version, formatted specifically for the **Group Theorization task**.

Each instance includes:
- Inputs to the user-implemented function `τ`:  
  `(d_A, d_B, t, u)`  
- Inputs to the evaluator module:  
  `(S₁, S₂)`

Samples are drawn from version (3), matched across demographics with **no duplicates** on the same split. Full sampling details are available in the paper.

**Features:**
- `description`: A short natural-language description of the neutral topic `t`
- `demographic_A`: The name of demographic group A (`d_A`)
- `demographic_B`: The name of demographic group B (`d_B`)
- `sample_A`: A list of 3 posts written by users from **either** group A or B about the topic (`S₁`)
- `sample_B`: A list of 3 posts written by users from the **opposite** group (`S₂`)
- `switched`: A boolean flag indicating whether the group identities have been swapped:  
  - `False`: `sample_A` corresponds to `demographic_A`, `sample_B` to `demographic_B`  
  - `True`: `sample_A` corresponds to `demographic_B`, `sample_B` to `demographic_A`
- `example_posts`: A calibration set `u` containing **42 posts** about the topic:  
  - 21 from group A  
  - 21 from group B  
  Posts are **shuffled and anonymized**.

---

## ⚙️ Setup

## 📥 Data Download

## 📊 Evaluate Your Model/Framework’s Group Theorization!

---
