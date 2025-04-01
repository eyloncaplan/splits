# Splits! 
## A Flexible Dataset for Evaluating a Model’s Demographic Social Inference

This repo includes the resources described in the Splits! paper, including demographic-split corpora, demographic-topic-split corpora, and the evaluation method proposed in the paper.

# Dataset Versions
There are four versions of the dataset. Which you pick depends entirely on your use case. We explain them in an order going from less processed -> more processed.

## Posts from All Seed Users (by Demographic)
This version uses a very loose definition of someone belonging to a demographic group. If a user has ever posted in one of our labeled seed subreddits, then they are considered a 'seed user'. This version of the dataset contains all posts made by all seed users across all subreddits (not just seed subreddits!), labeled by demographic group. You can think of this like a large collection of posts labeled by demographic with very high recall and low precision.

[specifics]

## Posts from High-Groupness Seed Users (by Demographic)
This version is similar to (1), but the definition of belonging to the demographic group is much tighter. In this version, we take (1), and use our 'group-ness' metric to select users who are more likely to be in the target demographic. As described in the paper, we pick our threshold by looking at the self-identification and anti-self-identification rates and choosing a point at which the two diverge significantly. The chosen values for the demographics are as follows (percentile):

Teacher: 75
Catholic: 75
Black: 75
Construction Worker: 90
Jewish: 90
Hindu/Jain/Sikh: 80

Filtering above the cutoff point gives this version of the dataset. You can think of this like a large collection of posts labeled by demographic with decent recall and pretty high precision.

[specifics]

## Posts from High-Groupness Seed Users (by Demographic AND Topic)
This version takes each demographic group's posts in (2) and further breaks it down into 10 categories of neutral topics, each with 20 individual topics (200 total neutral topics), using BM25 as a retrieval engine. The neutral categories, topics, and their retrieval keywords can be found in [specifics]. This version of the dataset is useful for investigating how different demographic groups discuss various topics differently.

[specifics]

## Splits! Dataset for Group Theorization Task
The final version is in the format ready for doing the Group Theorization task. Each row of this version is one instance of the Splits! dataset, including all inputs to the user-implemented function \tau, (d_A, d_B, t, u), and the inputs for our evaluator module (S_1, S_2). This version of the dataset is directly sampled from (3), matching together posts from two demographics and ensuring no duplicates. The details for this process are in the paper.

[specifics]

# Setup

# Data Download

# Evaluate your Model/Framework's Group Theorization!
