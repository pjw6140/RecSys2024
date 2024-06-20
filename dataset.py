from torch.utils.data import Dataset, DataLoader
import faiss
import pickle
from artifact import Artifact
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class RecSysDataset(Dataset):
    def __init__(self, pickle_path:str, artifact:Artifact, origin_pickle_path:str=None):
        super().__init__()
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        
        self.users = data["users"]
        self.user_mapping = data["user_mapping"]
        self.items = data["items"]
        self.item_mapping = data["item_mapping"]
        self.collaboration = data["collaboration"]
        self.artifact = artifact

        if origin_pickle_path != None:
            with open(origin_pickle_path, "rb") as f:
                origin_data = pickle.load(f)
                #self.users = origin_data["users"]
                self.user_mapping = origin_data["user_mapping"]
                #self.items = origin_data["items"]
                self.item_mapping = origin_data["item_mapping"]

        # build training data
        self.data = []
        for user in self.collaboration.keys():
            for item, score in self.collaboration[user]["interactions"].items():
                if int(user) in self.user_mapping and int(item) in self.item_mapping:
                    self.data.append((int(user), int(item), float(score)))

        # build user faiss index
        user_features = []
        for user_id in self.users:
            feature = self.collaboration[user_id]["features"]
            user_features.append(feature)
        self.user_features = np.vstack(user_features)
        self.user_faiss_index = faiss.IndexFlatL2(len(user_features[0]))
        self.user_faiss_index.add(self.user_features)

        # build item faiss index
        item_features = []
        for item_id in self.items:
            feature = self.artifact[item_id]
            item_features.append(feature)
        self.item_features = np.vstack(item_features)
        self.item_faiss_index = faiss.IndexFlatL2(len(item_features[0]))
        self.item_faiss_index.add(self.item_features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user_id, item_id, score = self.data[index]
        
        user_short_id = self.user_mapping[user_id]
        item_short_id = self.item_mapping[item_id]

        bert_embedding = self.artifact[item_id]
        return (user_short_id, item_short_id, score, bert_embedding)

    def nn_search_user(self, user_feature:np.ndarray):
        if len(user_feature.shape) == 1:
            user_feature = user_feature[np.newaxis, :]
        _, I = self.user_faiss_index.search(user_feature, 1)
        user_short_id = I[:,0]
        return user_short_id
    
    def nn_search_item(self, item_feature:np.ndarray):
        if len(item_feature.shape) == 1:
            item_feature = item_feature[np.newaxis, :]
        _, I = self.item_faiss_index.search(item_feature, 1)
        item_short_id = I[:,0]
        return item_short_id

if __name__ == '__main__':
    artifact = Artifact("data/FacebookAI_xlm_roberta_base/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet")
    train_data = RecSysDataset("small_train.pickle", artifact)
    #train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    #user_short_id, item_short_id, score, bert_embedding = next(iter(train_dataloader))
    #print(user_short_id, item_short_id, score, bert_embedding)

    print("load finish")
    user_feature = np.random.rand(32, 768+6).astype(np.float32)
    nearest_user_short_id = train_data.nn_search_user(user_feature)
    print(nearest_user_short_id)

    item_feature = np.random.rand(32, 768).astype(np.float32)
    nearest_item_short_id = train_data.nn_search_item(item_feature)
    print(nearest_item_short_id)