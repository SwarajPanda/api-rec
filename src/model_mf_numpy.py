import pandas as pd
import os
import numpy as np
from scipy.sparse import coo_matrix

FINAL = "data_final"

# ---- Load Train & Test ----
train_edges = pd.read_csv(os.path.join(FINAL, "train_edges.csv"))
test_edges = pd.read_csv(os.path.join(FINAL, "test_edges.csv"))

# Build mappings
mashup_ids = {m: i for i, m in enumerate(train_edges["mashup_id"].unique())}
api_ids = {a: i for i, a in enumerate(train_edges["api_id"].unique())}
rev_api_ids = {i: a for a, i in api_ids.items()}

# Sparse matrix (mashup × api)
rows = train_edges["mashup_id"].map(mashup_ids)
cols = train_edges["api_id"].map(api_ids)
data = [1] * len(train_edges)

R = coo_matrix((data, (rows, cols)), shape=(len(mashup_ids), len(api_ids))).toarray()

# ---- SVD Decomposition ----
U, s, Vt = np.linalg.svd(R, full_matrices=False)
k = 50
U_k = U[:, :k]
S_k = np.diag(s[:k])
Vt_k = Vt[:k, :]
R_hat = np.dot(np.dot(U_k, S_k), Vt_k)

# ---- Popularity (for fallback) ----
popularity = train_edges.groupby("api_id")["mashup_id"].count().sort_values(ascending=False)
popular_apis = popularity.index.tolist()

# ---- Recommend function ----
def recommend_svd(mashup_id, topN=10):
    if mashup_id not in mashup_ids:
        return popular_apis[:topN]   # completely new mashup → fallback
    idx = mashup_ids[mashup_id]
    scores = R_hat[idx]
    used = set(train_edges[train_edges["mashup_id"] == mashup_id]["api_id"])
    ranked = np.argsort(-scores)
    recs = [rev_api_ids[i] for i in ranked if rev_api_ids[i] not in used][:topN]
    if not recs:   # fallback if empty
        recs = popular_apis[:topN]
    return recs

# Example
example_mashup = test_edges.iloc[0]["mashup_id"]
print("Example mashup:", example_mashup)
print("Top-5 SVD Recommendations:", recommend_svd(example_mashup, 5))
