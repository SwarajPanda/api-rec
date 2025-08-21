import json, re, pandas as pd, os

RAW = "data_raw"
WORK = "data_work"
os.makedirs(WORK, exist_ok=True)

def normalize_text(s):
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    s = re.sub(r'/+$','', s)       # trailing slash hatao
    s = re.sub(r'\s+',' ', s)      # multiple spaces -> single space
    return s

# ---- APIs ----
apis = []
if os.path.exists(f"{RAW}/apis.json"):
    with open(f"{RAW}/apis.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    for d in data:
        ap = {
            "api_name": d.get("name"),
            "api_url": d.get("url"),
            "api_description": d.get("description"),
            "api_tags": d.get("tags"),
            "api_category": d.get("category"),
        }
        ap["api_url_norm"] = normalize_text(ap["api_url"])
        ap["api_name_norm"] = normalize_text(ap["api_name"])
        apis.append(ap)
    df_api = pd.DataFrame(apis)
    df_api.to_parquet(f"{WORK}/apis_raw_clean.parquet", index=False)
    print("APIs loaded:", len(df_api))
else:
    print("⚠️ apis.json file data_raw folder me nahi mila")

# ---- Mashups ----
mashups = []
if os.path.exists(f"{RAW}/mashups.json"):
    with open(f"{RAW}/mashups.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    for d in data:
        ms = {
            "mashup_name": d.get("name"),
            "mashup_url": d.get("url"),
            "mashup_description": d.get("description"),
            "apis_used": d.get("apis") or d.get("related_apis") or [],
        }
        ms["mashup_url_norm"] = normalize_text(ms["mashup_url"])
        ms["mashup_name_norm"] = normalize_text(ms["mashup_name"])
        mashups.append(ms)
    df_mash = pd.DataFrame(mashups)
    df_mash.to_parquet(f"{WORK}/mashups_raw_clean.parquet", index=False)
    print("Mashups loaded:", len(df_mash))
else:
    print("⚠️ mashups.json file data_raw folder me nahi mila")
