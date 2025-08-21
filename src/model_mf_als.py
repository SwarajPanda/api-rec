import pandas as pd
import os
import implicit
import scipy.sparse as sp
import numpy as np

FINAL = "data_final"

# ---- Load Data ----
train_edges = pd.read_csv(os.path.join(FINAL, "train_edges.csv"))
test_edges = pd.read_csv(os.path.join(FINAL, "test_edges.csv"))

# Create mapping (for implicit we need continuous indices starting from 0)
mashup_ids = {m: i for i, m in enumerate(train_edges["mashup_id"].unique())}
api_ids = {a: i for i, a in enumerate(train_edges["api_id"].unique())}

# Reverse mapping (for decoding back to original IDs)
rev_api_ids = {i: a for a, i in api_ids.items()}

# Build sparse matrix (mashup × api)
rows = train_edges["mashup_id"].map(mashup_ids)
cols = train_edges["api_id"].map(api_ids)
data = [1] * len(train_edges)

user_item_matrix = sp.coo_matrix((data, (rows, cols)),
                                 shape=(len(mashup_ids), len(api_ids))).tocsr()

# ---- Train ALS ----
model = implicit.als.AlternatingLeastSquares(
    factors=50,
    regularization=0.1,
    iterations=20,
    random_state=42
)

# Fit model
model.fit(user_item_matrix)

# ---- Recommend function ----
def recommend_als(mashup_id, topN=10):
    if mashup_id not in mashup_ids:
        return []  # unknown mashup
    idx = mashup_ids[mashup_id]
    recs = model.recommend(idx, user_item_matrix[idx], N=topN)
    return [rev_api_ids[i] for i, _ in recs]

# Example
example_mashup = test_edges.iloc[0]["mashup_id"]
print("Example mashup:", example_mashup)
print("Top-5 ALS Recommendations:", recommend_als(example_mashup, 5))
