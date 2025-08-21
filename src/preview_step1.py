import pandas as pd

# Load cleaned files
apis = pd.read_parquet("data_work/apis_clean.parquet")
mashups = pd.read_parquet("data_work/mashups_clean.parquet")
edges = pd.read_parquet("data_work/edges_clean.parquet")

print("\n--- APIs ---")
print(apis.head(5))   # first 5 rows
print("Shape:", apis.shape)

print("\n--- Mashups ---")
print(mashups.head(5))
print("Shape:", mashups.shape)

print("\n--- Edges ---")
print(edges.head(5))
print("Shape:", edges.shape)
