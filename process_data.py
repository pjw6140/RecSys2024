import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import pickle

def gen_collaborative_data(behaviors_path, history_path, artifacts_path, output_path, is_test_data=False):
    '''
    把原始数据处理成协同信息（类天池数据集）
    数据格式：dict: user_id -> article_id
    '''
    behaviors = pd.read_parquet(behaviors_path)
    history = pd.read_parquet(history_path)
    artifacts = pd.read_parquet(artifacts_path)

    # 构建文章的bert embedding字典
    article_bert_embeddings = {}
    for i in range(len(artifacts)):
        row = artifacts.iloc[i]
        article_id = row.iloc[0]
        embedding = row.iloc[1]
        article_bert_embeddings[article_id] = embedding
    
    collaboration = {}
    for i in tqdm(range(len(behaviors))):
        row = behaviors.iloc[i]
        user_id = row['user_id']

        if user_id not in collaboration:
            collaboration[user_id] = {}

            device_type = float(row["device_type"] * 10) # scale for nn search
            is_sso_user = float(1 if row["is_sso_user"] else 0)
            gender = float(row["gender"] * 10) # scale for nn search
            postcode = float(row["postcode"] * 10) # scale for nn search
            age = float(row["age"])
            is_subscriber = float(1 if row["is_subscriber"] else 0)

            if math.isnan(device_type):
                device_type = 0
            if math.isnan(gender):
                gender = 0
            if math.isnan(postcode):
                postcode = 0
            if math.isnan(age):
                age = 0

            collaboration[user_id]["features"] = []
            collaboration[user_id]["features"] += [device_type, is_sso_user, gender, postcode, age, is_subscriber]

            collaboration[user_id]["interactions"] = {}
        
        if not is_test_data:
            article_ids_clicked = row['article_ids_clicked']
            for article_id in article_ids_clicked:
                if int(article_id) in article_bert_embeddings:
                    collaboration[user_id]["interactions"][article_id] = 1
                else:
                    print(f"warning: {article_id} is not in artifacts!")
            
            article_ids_inview = row['article_ids_inview']
            for article_id in article_ids_inview:
                if int(article_id) in article_bert_embeddings:
                    if article_id not in collaboration[user_id]["interactions"]:
                        collaboration[user_id]["interactions"][article_id] = 0
                else:
                    print(f"warning: {article_id} is not in artifacts!")
    
    for i in tqdm(range(len(history))):
        row = history.iloc[i]
        user_id = row['user_id']
        article_id_fixed = row['article_id_fixed']
        scroll_percentage_fixed = row['scroll_percentage_fixed']
        read_time_fixed = row['read_time_fixed']

        if user_id not in collaboration:
            raise

        for j in range(len(article_id_fixed)):
            article_id = article_id_fixed[j]
            read_time = read_time_fixed[j]
            # 确保文章的bert embedding存在
            if int(article_id) not in article_bert_embeddings:
                print(f"warning: {article_id} is not in artifacts!")
                continue
            scroll_percentage = scroll_percentage_fixed[j]

            # 只有浏览进度超过10%，或者阅读时间超过2s的文章，才被看作正向交互
            if (not math.isnan(scroll_percentage) and scroll_percentage > 0.1) or (not math.isnan(read_time) and read_time >= 2.0):
                collaboration[user_id]["interactions"][article_id] = 1
            #elif not math.isnan(scroll_percentage):
            #    print(f"user {user_id} article {article_id} scroll precent less then 0.1 ({scroll_percentage}), so mark as unlike (read time: {read_time})")

    # 计算用户的平均文章
    for user_id in collaboration:
        liked_article_embs = []
        interactions = collaboration[user_id]["interactions"]
        for article_id in interactions:
            if interactions[article_id] == 1:
                liked_article_embs.append(article_bert_embeddings[int(article_id)])
        
        if len(liked_article_embs) == 0:
            avg_liked_emb = [0] * 768
        else:
            liked_article_embs = np.vstack(liked_article_embs)
            avg_liked_emb = liked_article_embs.mean(axis=0)
        for v in list(avg_liked_emb):
            collaboration[user_id]["features"].append(float(v))
        collaboration[user_id]["features"] = np.array(collaboration[user_id]["features"]).astype(np.float32)
    
    users = []
    user_mapping = {}
    items = []
    item_mapping = {}
    user_current_index = 0
    item_current_index = 0
    for user_id in collaboration.keys():
        if user_id not in user_mapping:
            users.append(user_id)
            user_mapping[user_id] = user_current_index
            user_current_index += 1
        interacts = collaboration[user_id]["interactions"]
        for item_id in interacts.keys():
            if item_id not in item_mapping:
                items.append(item_id)
                item_mapping[item_id] = item_current_index
                item_current_index += 1

    print("user count: ", len(users))
    print("item count: ", len(items))

    if is_test_data:
        data = collaboration
        for user_id in data:
            data[user_id] = data[user_id]["features"]
    else:
        data = {
            "users" : users,
            "user_mapping" : user_mapping,
            "items" : items,
            "item_mapping" : item_mapping,
            "collaboration" : collaboration
        }


    with open(output_path, "wb") as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    # gen_collaborative_data(
    #     "data/ebnerd_small/validation/behaviors.parquet", 
    #     "data/ebnerd_small/validation/history.parquet",
    #     "data/FacebookAI_xlm_roberta_base/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet",
    #     "small_validation.pickle")

    # gen_collaborative_data(
    #     "data/ebnerd_small/train/behaviors.parquet", 
    #     "data/ebnerd_small/train/history.parquet",
    #     "data/FacebookAI_xlm_roberta_base/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet",
    #     "small_train.pickle")

    gen_collaborative_data(
        "data/ebnerd_testset/ebnerd_testset/test/behaviors.parquet", 
        "data/ebnerd_testset/ebnerd_testset/test/history.parquet",
        "data/FacebookAI_xlm_roberta_base/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet",
        "test.pickle",is_test_data=True)

    # gen_collaborative_data(
    #     "data/ebnerd_demo/train/behaviors.parquet", 
    #     "data/ebnerd_demo/train/history.parquet",
    #     "data/FacebookAI_xlm_roberta_base/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet",
    #     "test_demo.pickle",is_test_data=True)
