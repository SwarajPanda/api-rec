import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 1. Load Data
train_edges = pd.read_csv('data_final/train_edges.csv')
val_edges = pd.read_csv('data_final/val_edges.csv')
test_edges = pd.read_csv('data_final/test_edges.csv')
apis = pd.read_csv('data_final/apis.csv')
mashups = pd.read_csv('data_final/mashups.csv')
api_popularity = pd.read_csv('data_final/api_popularity.csv')
api_similarity = pd.read_csv('data_final/api_similarity.csv')

# 2. Build ID Mappings
all_mashup_ids = pd.concat([train_edges['mashup_id'], val_edges['mashup_id'], test_edges['mashup_id']]).unique()
all_api_ids = pd.concat([train_edges['api_id'], val_edges['api_id'], test_edges['api_id']]).unique()
mashup2idx = {id: idx for idx, id in enumerate(all_mashup_ids)}
api2idx = {id: idx for idx, id in enumerate(all_api_ids)}
num_mashups = len(mashup2idx)
num_apis = len(api2idx)

# 3. Convert Edges to Indices
def edge_to_idx(df):
    df['mashup_idx'] = df['mashup_id'].map(mashup2idx)
    df['api_idx'] = df['api_id'].map(api2idx)
    return df[['mashup_idx', 'api_idx']].values

train_edge_index = edge_to_idx(train_edges)
val_edge_index = edge_to_idx(val_edges)
test_edge_index = edge_to_idx(test_edges)

# 4. LightGCN Model
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.num_layers = num_layers
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, edge_index):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings_list = [all_embeddings]
        num_nodes = self.user_embedding.num_embeddings + self.item_embedding.num_embeddings

        # Build adjacency matrix
        edge_tensor = torch.tensor(edge_index)
        adj_indices = torch.cat([
            torch.stack([edge_tensor[:,0], edge_tensor[:,1]+num_mashups]),
            torch.stack([edge_tensor[:,1]+num_mashups, edge_tensor[:,0]])
        ], dim=1)
        adj = torch.sparse_coo_tensor(
            adj_indices,
            torch.ones(adj_indices.shape[1]),
            (num_nodes, num_nodes)
        )

        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        final_embedding = sum(embeddings_list) / (self.num_layers + 1)
        user_emb_final = final_embedding[:num_mashups]
        item_emb_final = final_embedding[num_mashups:]
        return user_emb_final, item_emb_final

    def get_score(self, user_emb, item_emb):
        return torch.matmul(user_emb, item_emb.t())

# 5. Dataset for BPR Loss
class MashupApiDataset(Dataset):
    def __init__(self, edge_index, num_users, num_items):
        self.edge_index = edge_index
        self.num_users = num_users
        self.num_items = num_items
        self.user_pos_items = {}
        for u, i in edge_index:
            self.user_pos_items.setdefault(u, set()).add(i)

    def __len__(self):
        return len(self.edge_index)

    def __getitem__(self, idx):
        u, i = self.edge_index[idx]
        while True:
            j = np.random.randint(0, self.num_items)
            if j not in self.user_pos_items[u]:
                break
        return u, i, j

def bpr_loss(user_emb, pos_emb, neg_emb):
    pos_score = torch.sum(user_emb * pos_emb, dim=1)
    neg_score = torch.sum(user_emb * neg_emb, dim=1)
    return -torch.mean(F.logsigmoid(pos_score - neg_score))

# 6. Training Loop
dataset = MashupApiDataset(train_edge_index, num_mashups, num_apis)
loader = DataLoader(dataset, batch_size=1024, shuffle=True)
model = LightGCN(num_mashups, num_apis, embedding_dim=64, num_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    user_emb_final, item_emb_final = model(train_edge_index)
    for batch in loader:
        u, i, j = batch
        user_emb_final, item_emb_final = model(train_edge_index)
        user_emb = user_emb_final[u]
        pos_emb = item_emb_final[i]
        neg_emb = item_emb_final[j]
        loss = bpr_loss(user_emb, pos_emb, neg_emb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss {loss.item()}")

# 7. Validation/Evaluation (example for val set)
model.eval()
user_emb_final, item_emb_final = model(train_edge_index)
val_scores = model.get_score(user_emb_final, item_emb_final)
# For each mashup, get top-N recommended APIs
top_N = 10
recommendations = torch.topk(val_scores, top_N, dim=1).indices

def get_ground_truth(edge_index, num_mashups):
    """Returns a dict: mashup_idx -> set of true api_idx"""
    mashup_to_apis = {i: set() for i in range(num_mashups)}
    for mashup_idx, api_idx in edge_index:
        mashup_to_apis[mashup_idx].add(api_idx)
    return mashup_to_apis

def recall_at_k(recommendations, ground_truth, k=10):
    hits = 0
    total = 0
    for mashup_idx, true_apis in ground_truth.items():
        recs = set(recommendations[mashup_idx][:k])
        hits += len(recs & true_apis)
        total += len(true_apis)
    return hits / total if total > 0 else 0

def precision_at_k(recommendations, ground_truth, k=10):
    hits = 0
    total = 0
    for mashup_idx, true_apis in ground_truth.items():
        recs = set(recommendations[mashup_idx][:k])
        hits += len(recs & true_apis)
        total += k
    return hits / total if total > 0 else 0

def ndcg_at_k(recommendations, ground_truth, k=10):
    ndcg_total = 0
    num_users = len(ground_truth)
    for mashup_idx, true_apis in ground_truth.items():
        recs = recommendations[mashup_idx][:k]
        dcg = 0
        idcg = 0
        true_apis_list = list(true_apis)
        for rank, api_idx in enumerate(recs):
            if api_idx in true_apis:
                dcg += 1 / np.log2(rank + 2)
        for rank in range(min(len(true_apis), k)):
            idcg += 1 / np.log2(rank + 2)
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_total += ndcg
    return ndcg_total / num_users if num_users > 0 else 0

# --- After training, for evaluation on test set ---
# Convert test edges to indices
test_edges = pd.read_csv('data_final/test_edges.csv')
test_edges['mashup_idx'] = test_edges['mashup_id'].map(mashup2idx)
test_edges['api_idx'] = test_edges['api_id'].map(api2idx)
test_edge_index = test_edges[['mashup_idx', 'api_idx']].values

# Generate recommendations for each mashup
model.eval()
user_emb_final, item_emb_final = model(train_edge_index)
scores = model.get_score(user_emb_final, item_emb_final)  # [num_mashups, num_apis]
top_k = 10
recommendations = torch.topk(scores, top_k, dim=1).indices.cpu().numpy()  # [num_mashups, top_k]

# Prepare ground truth
ground_truth = get_ground_truth(test_edge_index, num_mashups)

# Compute metrics
recall = recall_at_k(recommendations, ground_truth, k=top_k)
precision = precision_at_k(recommendations, ground_truth, k=top_k)
ndcg = ndcg_at_k(recommendations, ground_truth, k=top_k)

print(f"Recall@{top_k}: {recall:.4f}")
print(f"Precision@{top_k}: {precision:.4f}")
print(f"NDCG@{top_k}: {ndcg:.4f}")