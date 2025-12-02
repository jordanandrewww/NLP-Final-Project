# A Hybrid Approach to Mental Health Classification

### Integrating Topic Modeling and BERT for Sentiment Analysis

**Authors:** Jordan Andrew, Aesha Gandhi, Ammy Lin

**Date:** December 2025

## Project Overview

This project investigates topic-aware text classification for mental health monitoring. Using a dataset labeled across seven mental states (Normal, Depression, Suicidal, Anxiety, Stress, Bipolar, Personality Disorder), we evaluate whether integrating traditional unsupervised structure discovery (LDA, K-Means) can enhance the predictive performance and interpretability of deep learning models (BERT).

The repository contains the code for benchmarking unsupervised, supervised, and hybrid architectures to determine the most effective approach for classifying short, informal mental health text.

## Abstract

We developed classifiers using Latent Dirichlet Allocation (LDA), K-Means clustering, and BERT embeddings, as well as combined hybrid architectures. Our results demonstrate that **deep contextual embeddings are the primary driver of performance**.

* The **BERT baseline achieved 70% accuracy**, vastly outperforming LDA (34%) and K-Means (36%).
* While unsupervised methods suffered from "class collapse" on lower-frequency labels, the addition of BERT embeddings corrected these errors.
* Hybrid models failed to significantly outperform the BERT-only baseline, suggesting that for short, sparse social media text, complex hybrid architectures may be over-engineering.

## Dataset

The project utilizes the **Sentiment Analysis for Mental Health** dataset from Kaggle, which aggregates user-generated content from various sources (e.g., social media posts).

* **Categories:** Normal, Depression, Suicidal, Anxiety, Stress, Bipolar, Personality Disorder.
* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

*Note: The dataset is not included in this repo due to size/licensing. Please download `combined_dataset.csv` from the source and place it in the `data/` directory.*

## Methodology

We implemented and compared five distinct modeling approaches:

### Baselines

1. **Baseline 1 (LDA Only):** Uses topic distribution vectors from Latent Dirichlet Allocation as features for Logistic Regression.
2. **Baseline 2 (BERT Only):** Uses `all-MiniLM-L6-v2` Sentence-BERT embeddings as features for Logistic Regression.
3. **Baseline 3 (K-Means Only):** Uses K-Means cluster assignments (derived from TF-IDF vectors) as features.

### Hybrid Models

4. **Hybrid 1 (LDA + BERT):** Concatenates BERT embeddings with LDA topic probability vectors.
5. **Hybrid 2 (Clustering + BERT):** Concatenates BERT embeddings with K-Means cluster indicator features.

## Key Results

| Model | Accuracy | Macro F1 | Key Observation |
| :--- | :--- | :--- | :--- |
| **LDA Only** | 34% | 0.28 | Struggled with nuanced classes; high confusion between Depression/Suicidal. |
| **K-Means Only** | 36% | 0.16 | Severe "class collapse" (F1 score of 0.00 for Bipolar, Depression, Stress). |
| **BERT Only** | **70%** | **0.62** | **Best performance.** Captured semantic nuance effectively. |
| **Hybrid 1 (LDA+BERT)** | 70% | 0.62 | No significant gain over BERT alone. |
| **Hybrid 2 (Cluster+BERT)** | 70% | 0.62 | No significant gain over BERT alone. |

**Conclusion:** A well-tuned transformer (BERT) is sufficient and more efficient than trying to force-fit topic modeling features for this specific domain.

## Installation & Usage

### Prerequisites

* Python 3.8+
* Jupyter Notebook

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/jordanandrewww/NLP-Final-Project.git](https://github.com/jordanandrewww/NLP-Final-Project.git)
   cd NLP-Final-Project