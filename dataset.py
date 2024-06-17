from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
from id_mapping import IDMapping

class RecSysDataset(Dataset):
    def __init__(self, interact_json_path, mapping_path, artifacts_path):
        super().__init__()
        with open(interact_json_path, "r") as f:
            interacts = json.load(f)
        self.mapping = IDMapping.load(mapping_path)
        artifacts = pd.read_parquet(artifacts_path)

        self.article_bert_embeddings = {}
        for i in range(len(artifacts)):
            row = artifacts.iloc[i]
            article_id = row.iloc[0]
            embedding = row.iloc[1]
            self.article_bert_embeddings[article_id] = embedding

        self.data = []
        for user in interacts.keys():
            for item, score in interacts[user].items():
                
                user_short_id = self.mapping.try_get(int(user), "user")
                item_short_id = self.mapping.try_get(int(item), "item")

                if user_short_id != None and item_short_id != None:
                    self.data.append((int(user), int(item), float(score)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user_id, item_id, score = self.data[index]
        user_short_id = self.mapping.try_get(user_id, "user")
        item_short_id = self.mapping.try_get(item_id, "item")
        
        if user_short_id == None or item_short_id == None:
            raise

        bert_embedding = self.article_bert_embeddings[item_id]
        return (user_short_id, item_short_id, score, bert_embedding)

if __name__ == '__main__':
    train_data = RecSysDataset("small_train.json", "small_id_mapping.pickle", "data/FacebookAI_xlm_roberta_base/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet")
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    user_short_id, item_short_id, score, bert_embedding = next(iter(train_dataloader))
    print(user_short_id, item_short_id, score, bert_embedding)