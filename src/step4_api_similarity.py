import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

READY = "data_ready"

# Load APIs
apis = pd.read_parquet(os.path.join(READY, "apis_ready.parquet"))

# Fill NA categories
apis["api_category"] = apis["api_category"].fillna("unknown")

# TF-IDF on category
vectorizer = TfidfVectorizer()
cat_vectors = vectorizer.fit_transform(apis["api_category"])

# Compute similarity in chunks
sim_matrix = cosine_similarity(cat_vectors)

# For each API, keep top-20 similar
pairs = []
top_k = 20
api_ids = apis["api_id"].values

for i in range(sim_matrix.shape[0]):
    sims = sim_matrix[i]
    top_idx = np.argsort(sims)[-top_k-1:-1]  # top k excluding self
    for j in top_idx:
        pairs.append((api_ids[i], api_ids[j], sims[j]))

sim_df = pd.DataFrame(pairs, columns=["api_id1","api_id2","sim_score"])
sim_df.to_parquet(os.path.join(READY, "api_similarity.parquet"), index=False)

print("Similarity pairs saved:", sim_df.shape)
