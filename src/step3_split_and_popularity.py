import pandas as pd
import numpy as np
import os

READY = "data_ready"

# Load cleaned data
apis = pd.read_parquet(os.path.join(READY, "apis_ready.parquet"))
mashups = pd.read_parquet(os.path.join(READY, "mashups_ready.parquet"))
edges = pd.read_parquet(os.path.join(READY, "edges_ready.parquet"))

# ---- Temporal Split ----
mashups = mashups.sort_values("mashup_submit_date").reset_index(drop=True)

n = len(mashups)
train_cut = int(0.8 * n)
val_cut   = int(0.9 * n)

mashups["split"] = "train"
mashups.loc[train_cut:val_cut, "split"] = "val"
mashups.loc[val_cut:, "split"] = "test"

mashups.to_parquet(os.path.join(READY, "mashups_splits.parquet"), index=False)
print("Mashup splits:", mashups["split"].value_counts().to_dict())

# Merge edges with split info
edges = edges.merge(mashups[["mashup_id","split"]], on="mashup_id", how="left")
edges.to_parquet(os.path.join(READY, "edges_splits.parquet"), index=False)

# ---- Popularity Stats (Train only) ----
train_edges = edges[edges["split"]=="train"]

pop = train_edges.groupby("api_id")["mashup_id"].nunique().rename("pop_raw").reset_index()
pop = apis[["api_id","api_name","api_url","api_category"]].merge(pop, on="api_id", how="left").fillna({"pop_raw":0})
pop["pop_log"] = np.log1p(pop["pop_raw"])

# Bins
q50 = pop["pop_raw"].quantile(0.5)
q80 = pop["pop_raw"].quantile(0.8)

def bin_pop(x):
    if x > q80: return "head"
    if x > q50: return "torso"
    return "tail"

pop["pop_bin"] = pop["pop_raw"].apply(bin_pop)

pop.to_parquet(os.path.join(READY, "api_popularity.parquet"), index=False)

print("Popularity bins:", pop["pop_bin"].value_counts().to_dict())
