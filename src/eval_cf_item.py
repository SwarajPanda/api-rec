import pandas as pd
import os
import numpy as np
from collections import defaultdict

FINAL = "data_final"

# ---- Load Train & Test Data ----
train_edges = pd.read_csv(os.path.join(FINAL, "train_edges.csv"))
test_edges = pd.read_csv(os.path.join(FINAL, "test_edges.csv"))

# ---- Build Co-occurrence Counts ----
mashup_groups = train_edges.groupby("mashup_id")["api_id"].apply(list)

co_counts = defaultdict(int)
api_counts = defaultdict(int)

for apis in mashup_groups:
    unique_apis = list(set(apis))
    for i in range(len(unique_apis)):
        api_counts[unique_apis[i]] += 1
        for j in range(i+1, len(unique_apis)):
            a, b = sorted([unique_apis[i], unique_apis[j]])
            co_counts[(a, b)] += 1

# ---- Compute Similarity ----
similarity = defaultdict(dict)

for (a, b), c in co_counts.items():
    sim = c / np.sqrt(api_counts[a] * api_counts[b])
    similarity[a][b] = sim
    similarity[b][a] = sim

# ---- Recommender ----
def recommend_cf(mashup_apis, topN=10):
    scores = defaultdict(float)
    for api in mashup_apis:
        for neigh, sim in similarity.get(api, {}).items():
            if neigh not in mashup_apis:
                scores[neigh] += sim
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [api for api, _ in ranked[:topN]]

# ---- Metrics ----
def recall_at_k(true_items, pred_items, k):
    pred_topk = set(pred_items[:k])
    return len(set(true_items) & pred_topk) / len(set(true_items)) if true_items else 0

def precision_at_k(true_items, pred_items, k):
    pred_topk = set(pred_items[:k])
    return len(set(true_items) & pred_topk) / len(pred_topk) if pred_topk else 0

# ---- Evaluate on Test ----
def evaluate_cf(K=10):
    recalls, precisions = [], []
    grouped = test_edges.groupby("mashup_id")["api_id"].apply(list)

    for mashup_id, true_apis in grouped.items():
        preds = recommend_cf(true_apis, K)
        if preds:  # only evaluate if we have predictions
            recalls.append(recall_at_k(true_apis, preds, K))
            precisions.append(precision_at_k(true_apis, preds, K))

    print(f"Evaluation CF @K={K}")
    print("Avg Recall:", np.mean(recalls))
    print("Avg Precision:", np.mean(precisions))

# Run evaluation
evaluate_cf(K=10)
