import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from artifact import Artifact
from dataset import RecSysDataset

def get_ranking(float_array):
    sorted_array = sorted(float_array, reverse=True)
    int_array = [sorted_array.index(num) + 1 for num in float_array]
    return int_array

output = open("data/predictions.txt", 'w')

with open("test.pickle", "rb") as f:
    test_feature_data = pickle.load(f)

with open("user_id_mapping.pickle", "rb") as f:
    user_id_mapping = pickle.load(f)

with open("item_id_mapping.pickle", "rb") as f:
    item_id_mapping = pickle.load(f)

#model = torch.load("checkpoints/latest.pt").to("cpu")
model = torch.load("checkpoints/best.pt").to("cpu")
model.eval()

artifact = Artifact("data/FacebookAI_xlm_roberta_base/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet")

behaviors = pd.read_parquet("data/ebnerd_testset/ebnerd_testset/test/behaviors.parquet")
for i in tqdm(range(len(behaviors))):
    row = behaviors.iloc[i]
    impression_id = row['impression_id']
    user_id = row['user_id']

    user_short_id = user_id_mapping[user_id]
        
    input_user_ids = []
    input_item_ids = []
    input_bert_embs = []

    article_ids_inview = row['article_ids_inview']
    for article_id in article_ids_inview:
        input_user_ids.append(user_short_id)
        bert_emb = artifact[article_id]
        input_bert_embs.append(bert_emb)

        item_short_id = item_id_mapping[article_id]
        input_item_ids.append(item_short_id)

    input_user_ids = torch.Tensor(input_user_ids).int()
    input_item_ids = torch.Tensor(input_item_ids).int()
    input_bert_embs = torch.from_numpy(np.vstack(input_bert_embs)).float()

    with torch.no_grad():
        predict_score = model(input_user_ids, input_item_ids, input_bert_embs).numpy()

    rank = get_ranking(predict_score)
    
    line = f"{impression_id} {str(rank).replace(' ','')}"
    output.write(line + "\n")

output.close()