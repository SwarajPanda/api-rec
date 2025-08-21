import pandas as pd
import os

FINAL = "data_final"

# ---- Load Data ----
train_edges = pd.read_csv(os.path.join(FINAL, "train_edges.csv"))
val_edges = pd.read_csv(os.path.join(FINAL, "val_edges.csv"))
test_edges = pd.read_csv(os.path.join(FINAL, "test_edges.csv"))

# ---- Train Popularity Model ----
# Popularity = number of mashups using each API
popularity = train_edges.groupby("api_id")["mashup_id"].count().sort_values(ascending=False)
top_apis = popularity.index.tolist()

def recommend_topN(N=10):
    """Always recommend top-N most popular APIs"""
    return top_apis[:N]

# Quick check
print("Top-5 Popular APIs:", recommend_topN(5))
