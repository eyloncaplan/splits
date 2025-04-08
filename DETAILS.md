# DETAILS.md

This document contains additional information about the **Splits!** dataset, including:

- Group-ness thresholds
- Topic structure
- Sampling details for the Group Theorization task
- Field definitions

---

## 🧠 Group-ness Thresholds

We use a group-ness metric to identify users more likely to authentically belong to a demographic group.  
Thresholds are determined by analyzing self-identification vs. anti-self-identification rates and selecting the point where they diverge significantly.

**Percentile Cutoffs:**

| Demographic Group        | Threshold (Percentile) |
|--------------------------|------------------------|
| Teacher                  | 75                     |
| Catholic                 | 75                     |
| Black                    | 75                     |
| Construction Worker      | 90                     |
| Jewish                   | 90                     |
| Hindu/Jain/Sikh          | 80                     |

---

## 📚 Topic Structure (Versions 3 & 4)

- **10 neutral topic categories**
- **20 specific topics per category** (200 total)
- Topics selected using **BM25 retrieval** based on manually defined keyword prompts

Each post in versions (3) and (4) is retrieved for a specific neutral topic. Topic categories, specific topics, and retrieval keywords are listed in the main paper and included in this repo under `[specifics]`.

---

## 🧪 Sampling Procedure for Group Theorization (Version 4)

- Samples are drawn from Version (3)
- Matched across demographics with **no duplicates** on the same split
- Inputs for task and evaluation formatted specifically for the Group Theorization task
- Group identities are optionally **switched** to test whether models make assumptions based on labels
- Includes a **calibration set** of 42 posts per topic

---

## 🧾 Field Descriptions

### Version 1 & 2

| Field              | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `id`               | Unique identifier for the post                                              |
| `user`             | Reddit username of the author                                               |
| `timestamp`        | Unix timestamp (in milliseconds) of when the post was created              |
| `text`             | The body of the post                                                        |
| `demographic`      | The demographic label assigned to the post                                 |
| `subreddit`        | The subreddit where the post appeared                                       |
| `metric_percentile` (v2 only) | Percentile for group-ness of the user within their demographic |

---

### Version 3

| Field        | Description                                                                                      |
|--------------|--------------------------------------------------------------------------------------------------|
| `id`         | Unique identifier for the post                                                                   |
| `description`| The specific neutral topic the post was retrieved for                                            |
| `demographic`| The demographic group assigned to the post                                                       |
| `content`    | The text of the post                                                                             |
| `metadata`   | A dictionary containing post metadata:<br> - `timestamp`: When the post was created <br> - `score`: Reddit score (upvotes - downvotes) <br> - `subreddit`: The subreddit where the post appeared <br> - `user`: The username of the one who posted |
| `score`      | BM25 similarity score between the post and the topic keywords                                   |

---

### Version 4 (Splits! for Group Theorization)

| Field            | Description                                                                                                                           |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `description`    | A short natural-language description of the neutral topic `t`                                                                         |
| `demographic_A`  | The name of demographic group A (`d_A`)                                                                                               |
| `demographic_B`  | The name of demographic group B (`d_B`)                                                                                               |
| `sample_A`       | A list of 3 posts written by users from **either** group A or B about the topic (`S₁`)                                               |
| `sample_B`       | A list of 3 posts written by users from the **opposite** group (`S₂`)                                                                 |
| `switched`       | A boolean flag indicating whether the group identities have been swapped:<br> - `False`: `sample_A` corresponds to `demographic_A`, `sample_B` to `demographic_B`<br> - `True`: `sample_A` corresponds to `demographic_B`, `sample_B` to `demographic_A` |
| `example_posts`  | A calibration set `u` containing **42 posts** (**shuffled and anonymized**) about the topic:<br> - 21 from group A<br> - 21 from group B |

---

## 🧭 Suggested Use Cases

- **Version 1**: Broad demographic analysis, exploratory studies  
- **Version 2**: Precision-sensitive analysis of group language  
- **Version 3**: Topic-based and group-based discourse comparison  
- **Version 4**: Evaluation on group theorization task (Splits!)

---

## 📬 Questions?

Feel free to open an issue or reach out via [GitHub](https://github.com/eyloncaplan).
