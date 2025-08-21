import pandas as pd
import os
import numpy as np
from lightfm import LightFM
from scipy.sparse import coo_matrix

FINAL = "data_final"

# ---- Load Data ----
train_edges = pd.read_csv(os.path.join(FINAL, "train_edges.csv"))
test_edges = pd.read_csv(os.path.join(FINAL, "test_edges.csv"))

# Build ID mappings
mashup_ids = {m: i for i, m in enumerate(train_edges["mashup_id"].unique())}
api_ids = {a: i for i, a in enumerate(train_edges["api_id"].unique())}
rev_api_ids = {i: a for a, i in api_ids.items()}

# Build interaction matrix (mashup × api)
rows = train_edges["mashup_id"].map(mashup_ids)
cols = train_edges["api_id"].map(api_ids)
data = [1] * len(train_edges)

interaction_matrix = coo_matrix((data, (rows, cols)),
                                shape=(len(mashup_ids), len(api_ids)))

# ---- Train LightFM MF model ----
model = LightFM(loss="warp", no_components=50, random_state=42)
model.fit(interaction_matrix, epochs=20, num_threads=4)

# ---- Recommend function ----
def recommend_lightfm(mashup_id, topN=10):
    if mashup_id not in mashup_ids:
        return []
    idx = mashup_ids[mashup_id]
    scores = model.predict(idx, np.arange(len(api_ids)))
    top_items = np.argsort(-scores)[:topN]
    return [rev_api_ids[i] for i in top_items]

# Example
example_mashup = test_edges.iloc[0]["mashup_id"]
print("Example mashup:", example_mashup)
print("Top-5 LightFM Recommendations:", recommend_lightfm(example_mashup, 5))
