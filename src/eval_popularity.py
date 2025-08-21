import pandas as pd
import os
import numpy as np

FINAL = "data_final"

# ---- Load Data ----
train_edges = pd.read_csv(os.path.join(FINAL, "train_edges.csv"))
test_edges = pd.read_csv(os.path.join(FINAL, "test_edges.csv"))

# Build popularity model
popularity = train_edges.groupby("api_id")["mashup_id"].count().sort_values(ascending=False)
top_apis = popularity.index.tolist()

def recommend_topN(N=10):
    return top_apis[:N]

# ---- Evaluation ----
def recall_at_k(true_items, pred_items, k):
    pred_topk = set(pred_items[:k])
    return len(set(true_items) & pred_topk) / len(set(true_items)) if true_items else 0

def precision_at_k(true_items, pred_items, k):
    pred_topk = set(pred_items[:k])
    return len(set(true_items) & pred_topk) / len(pred_topk) if pred_topk else 0

def evaluate(K=10):
    recalls, precisions = [], []
    grouped = test_edges.groupby("mashup_id")["api_id"].apply(list)

    for mashup_id, true_apis in grouped.items():
        preds = recommend_topN(K)
        recalls.append(recall_at_k(true_apis, preds, K))
        precisions.append(precision_at_k(true_apis, preds, K))

    print(f"Evaluation @K={K}")
    print("Avg Recall:", np.mean(recalls))
    print("Avg Precision:", np.mean(precisions))

# Run evaluation
evaluate(K=10)
