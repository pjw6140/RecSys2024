import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from artifact import Artifact
from dataset import RecSysDataset

with open("test.pickle", "rb") as f:
    test_feature_data = pickle.load(f)

model = torch.load("checkpoints/latest.pt").to("cpu")
model.eval()

artifact = Artifact("data/FacebookAI_xlm_roberta_base/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet")
train_data = RecSysDataset("small_train.pickle", artifact)

user_id_mapping = {}
item_id_mapping = {}

unknown_users = set()
unknown_items = set()

behaviors = pd.read_parquet("data/ebnerd_testset/ebnerd_testset/test/behaviors.parquet")
for i in tqdm(range(len(behaviors)), desc="read behaviors"):
    row = behaviors.iloc[i]
    user_id = row['user_id']

    if user_id in train_data.user_mapping:
        user_short_id = train_data.user_mapping[user_id]
        user_id_mapping[user_id] = user_short_id
    else:
        unknown_users.add(user_id)
    
    article_ids_inview = row['article_ids_inview']
    for article_id in article_ids_inview:
        if article_id in train_data.item_mapping:
            item_short_id = train_data.item_mapping[article_id]
            item_id_mapping[article_id] = item_short_id
        else:
            unknown_items.add(article_id)

unknown_users = list(unknown_users)
unknown_items = list(unknown_items)

print("unknown users: ", len(unknown_users))
print("unknown items: ", len(unknown_items))

print("users: ", len(user_id_mapping))
print("items: ", len(item_id_mapping))

batch_size = 128

# nn search user
num_batch = len(unknown_users) // batch_size
if len(unknown_users) % num_batch != 0:
    num_batch += 1
for bi in tqdm(range(num_batch), desc="nn search user"):
    batch_unknown_users = unknown_users[bi * batch_size : (bi + 1) * batch_size]
    features = []
    for user_id in batch_unknown_users:
        features.append(test_feature_data[user_id])
    features = np.vstack(features)
    short_ids = train_data.nn_search_user(features)
    for i in range(len(batch_unknown_users)):
        user_id = batch_unknown_users[i]
        short_id = short_ids[i]
        user_id_mapping[user_id] = short_id

# nn search item
num_batch = len(unknown_items) // batch_size
if len(unknown_items) % num_batch != 0:
    num_batch += 1
for bi in tqdm(range(num_batch), desc="nn search item"):
    batch_unknown_items = unknown_items[bi * batch_size : (bi + 1) * batch_size]
    features = []
    for article_id in batch_unknown_items:
        features.append(artifact[article_id])
    features = np.vstack(features)
    short_ids = train_data.nn_search_item(features)
    for i in range(len(batch_unknown_items)):
        user_id = batch_unknown_items[i]
        short_id = short_ids[i]
        item_id_mapping[user_id] = short_id

print("users: ", len(user_id_mapping))
print("items: ", len(item_id_mapping))

with open("user_id_mapping.pickle", "wb") as f:
    pickle.dump(user_id_mapping, f)
with open("item_id_mapping.pickle", "wb") as f:
    pickle.dump(item_id_mapping, f)