import pandas as pd
import os

READY = "data_ready"
FINAL = "data_final"
os.makedirs(FINAL, exist_ok=True)

# Load data
apis = pd.read_parquet(os.path.join(READY, "apis_ready.parquet"))
mashups = pd.read_parquet(os.path.join(READY, "mashups_splits.parquet"))
edges = pd.read_parquet(os.path.join(READY, "edges_splits.parquet"))
pop = pd.read_parquet(os.path.join(READY, "api_popularity.parquet"))
sim = pd.read_parquet(os.path.join(READY, "api_similarity.parquet"))

# ---- Save final splits ----
train_edges = edges[edges["split"]=="train"]
val_edges = edges[edges["split"]=="val"]
test_edges = edges[edges["split"]=="test"]

train_edges.to_csv(os.path.join(FINAL, "train_edges.csv"), index=False)
val_edges.to_csv(os.path.join(FINAL, "val_edges.csv"), index=False)
test_edges.to_csv(os.path.join(FINAL, "test_edges.csv"), index=False)

# ---- Save metadata ----
apis.to_csv(os.path.join(FINAL, "apis.csv"), index=False)
mashups.to_csv(os.path.join(FINAL, "mashups.csv"), index=False)
pop.to_csv(os.path.join(FINAL, "api_popularity.csv"), index=False)
sim.to_csv(os.path.join(FINAL, "api_similarity.csv"), index=False)

# ---- Print Summary ----
print("=== FINAL DATASET SUMMARY ===")
print("APIs:", apis.shape)
print("Mashups:", mashups.shape)
print("Edges:", edges.shape)
print("Train edges:", train_edges.shape)
print("Val edges:", val_edges.shape)
print("Test edges:", test_edges.shape)
print("Popularity entries:", pop.shape)
print("Similarity pairs:", sim.shape)
