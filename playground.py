import pandas as pd
import numpy as np
import math
import json
from tqdm import tqdm

def get_embbedding_count(behaviors_path, history_path):
    '''
    了解下数据集里面用户和文章的id最大值，方便设置embedding大小
    '''
    behaviors = pd.read_parquet(behaviors_path)
    history = pd.read_parquet(history_path)
    
    max_user_id = 0
    max_article_id = 0

    for i in tqdm(range(len(behaviors))):
        row = behaviors.iloc[i]
        user_id = int(row['user_id'])
        max_user_id = max(user_id, max_user_id)

        article_ids_inview = row['article_ids_inview']
        for article_id in article_ids_inview:
            article_id = int(article_id)
            max_article_id = max(article_id, max_article_id)

    for i in tqdm(range(len(history))):
        row = history.iloc[i]
        user_id = int(row['user_id'])
        max_user_id = max(user_id, max_user_id)

        article_id_fixed = row['article_id_fixed']
        for j in range(len(article_id_fixed)):
            article_id = article_id_fixed[j]
            article_id = int(article_id)
            max_article_id = max(article_id, max_article_id)
    print("max user id: ", max_user_id)
    print("max article id: ", max_article_id)
    return max_user_id, max_article_id


if __name__ == '__main__':
    get_embbedding_count(
        "data/ebnerd_demo/validation/behaviors.parquet", 
        "data/ebnerd_demo/validation/history.parquet")

    get_embbedding_count(
        "data/ebnerd_demo/train/behaviors.parquet", 
        "data/ebnerd_demo/train/history.parquet")

    get_embbedding_count(
        "data/ebnerd_large/train/behaviors.parquet", 
        "data/ebnerd_large/train/history.parquet")

    get_embbedding_count(
        "data/ebnerd_large/train/behaviors.parquet", 
        "data/ebnerd_large/train/history.parquet")

    get_embbedding_count(
        "data/ebnerd_testset/test/behaviors.parquet", 
        "data/ebnerd_testset/test/history.parquet")
