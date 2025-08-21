import pandas as pd
import os
import numpy as np
from collections import defaultdict

FINAL = "data_final"

# ---- Load Train Data ----
train_edges = pd.read_csv(os.path.join(FINAL, "train_edges.csv"))
test_edges = pd.read_csv(os.path.join(FINAL, "test_edges.csv"))

# ---- Build co-occurrence counts ----
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

# ---- Compute similarity ----
similarity = defaultdict(dict)

for (a, b), c in co_counts.items():
    sim = c / np.sqrt(api_counts[a] * api_counts[b])
    similarity[a][b] = sim
    similarity[b][a] = sim

# ---- Recommendation function ----
def recommend_cf(mashup_apis, topN=10):
    scores = defaultdict(float)
    for api in mashup_apis:
        for neigh, sim in similarity.get(api, {}).items():
            if neigh not in mashup_apis:  # don’t recommend already used APIs
                scores[neigh] += sim
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [api for api, _ in ranked[:topN]]

# Quick test on one mashup
example_mashup = test_edges.groupby("mashup_id")["api_id"].first().index[0]
true_apis = test_edges[test_edges["mashup_id"] == example_mashup]["api_id"].tolist()
preds = recommend_cf(true_apis, topN=5)
print("Example true APIs:", true_apis)
print("Predicted:", preds)
