import pandas as pd
import os

WORK = "data_work"
READY = "data_ready"
os.makedirs(READY, exist_ok=True)

# ---- Load APIs ----
apis = pd.read_parquet(os.path.join(WORK, "apis_clean.parquet"))
# parse dates
apis["api_submit_date"] = pd.to_datetime(apis["api_submit_date"], errors="coerce")
apis["api_dead_date"] = pd.to_datetime(apis["api_dead_date"], errors="coerce")
# assign ID
apis = apis.reset_index(drop=True)
apis["api_id"] = apis.index
apis.to_parquet(os.path.join(READY, "apis_ready.parquet"), index=False)

print("APIs ready:", apis.shape)

# ---- Load Mashups ----
mashups = pd.read_parquet(os.path.join(WORK, "mashups_clean.parquet"))
mashups["mashup_submit_date"] = pd.to_datetime(mashups["mashup_submit_date"], errors="coerce")
mashups = mashups.reset_index(drop=True)
mashups["mashup_id"] = mashups.index
mashups.to_parquet(os.path.join(READY, "mashups_ready.parquet"), index=False)

print("Mashups ready:", mashups.shape)

# ---- Load Edges ----
edges_raw = pd.read_parquet(os.path.join(WORK, "edges_clean.parquet"))

# Split the single column on tab
edges_split = edges_raw.iloc[:,0].str.split("\t", expand=True)
edges_split.columns = ["mashup_name", "api_url"]

# Clean names (strip spaces)
edges_split["mashup_name"] = edges_split["mashup_name"].str.strip()
edges_split["api_url"] = edges_split["api_url"].str.strip()

# Merge with IDs
edges = edges_split.merge(mashups[["mashup_id","mashup_name"]], on="mashup_name", how="inner")
edges = edges.merge(apis[["api_id","api_url"]], on="api_url", how="inner")

edges = edges[["mashup_id","api_id"]].drop_duplicates()
edges.to_parquet(os.path.join(READY, "edges_ready.parquet"), index=False)

print("Edges ready:", edges.shape)
