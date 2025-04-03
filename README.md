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

This version uses a **very loose** definition of group membership. If a user has ever posted in one of our labeled seed subreddits, they are considered a *seed user*. This dataset contains **all posts** made by all seed users across **all subreddits** (not just seed subreddits!), labeled by demographic group. Think of this as a **high-recall, low-precision** dataset of posts labeled by demographic.

**Features:**
- `id`: Unique identifier for the post  
- `user`: Reddit username of the author  
- `timestamp`: Unix timestamp (in milliseconds) of when the post was created  
- `text`: The body of the post  
- `demographic`: The demographic label assigned to the post  
- `subreddit`: The subreddit where the post appeared

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

Think of this as a **decent recall** and **higher precision** version of (1).

**Features:**
Identical to (1), but adds
- `metric_percentile`: Percentile for 'groupness' of the user within their demographic.

---

### 3. Posts from High-Groupness Seed Users (by Demographic **and** Topic)

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
  - `user`: The username of the poster
- `score`: BM25 similarity score between the post and the topic keywords

---

### 4. Splits! Dataset for Group Theorization Task

This is the final version, formatted specifically for the **Group Theorization task**.

Each instance includes:
- Inputs to the user-implemented function `τ`:  
  `(d_A, d_B, t, u)`  
- Inputs to the evaluator module:  
  `(S₁, S₂)`

Samples are drawn from version (3), matched across demographics with **no duplicates** on the same split.

Full sampling and formatting details are available in the paper.

`[specifics]`

---

## ⚙️ Setup

## 📥 Data Download

## 📊 Evaluate Your Model/Framework’s Group Theorization!

---
