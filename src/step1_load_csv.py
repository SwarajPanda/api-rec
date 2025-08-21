import pandas as pd
import os

RAW = "data_raw"
WORK = "data_work"
os.makedirs(WORK, exist_ok=True)

def load_csv_safely(path):
    """Try different delimiters until it works"""
    try:
        return pd.read_csv(path)  # default comma
    except:
        try:
            return pd.read_csv(path, sep=";")
        except:
            return pd.read_csv(path, sep="\t")  # tab

# ---- APIs ----
api_file = os.path.join(RAW, "api_nodes_estimator.csv")
apis = load_csv_safely(api_file)
apis = apis.rename(columns={
    "url": "api_url",
    "name": "api_name",
    "st": "api_submit_date",
    "et": "api_dead_date",
    "c": "api_category"
})
print("APIs loaded:", apis.shape)
apis.to_parquet(os.path.join(WORK, "apis_clean.parquet"), index=False)

# ---- Mashups ----
mashup_file = os.path.join(RAW, "mashup_nodes_estimator.csv")
mashups = load_csv_safely(mashup_file)
mashups = mashups.rename(columns={
    "url": "mashup_url",
    "name": "mashup_name",
    "st": "mashup_submit_date"
})
print("Mashups loaded:", mashups.shape)
mashups.to_parquet(os.path.join(WORK, "mashups_clean.parquet"), index=False)

# ---- Edges ----
edges_file = os.path.join(RAW, "m-a_edges.csv")
edges = load_csv_safely(edges_file)
edges = edges.rename(columns={
    "s": "mashup_id",
    "t": "api_id"
})
print("Edges loaded:", edges.shape)
edges.to_parquet(os.path.join(WORK, "edges_clean.parquet"), index=False)
