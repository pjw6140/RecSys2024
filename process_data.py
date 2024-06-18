import pandas as pd
import numpy as np
import math
import json
from tqdm import tqdm

def gen_collaborative_data(behaviors_path, history_path, artifacts_path, output_path):
    '''
    把原始数据处理成协同信息（类天池数据集）
    数据格式：dict: user_id -> article_id
    '''
    behaviors = pd.read_parquet(behaviors_path)
    history = pd.read_parquet(history_path)
    artifacts = pd.read_parquet(artifacts_path)
    artifacts = set(artifacts['article_id'])
    
    collaboration = {}
    for i in tqdm(range(len(behaviors))):
        row = behaviors.iloc[i]
        user_id = str(row['user_id'])

        article_ids_clicked = row['article_ids_clicked']
        if user_id not in collaboration:
            collaboration[user_id] = {}
        for article_id in article_ids_clicked:
            article_id = str(article_id)
            if int(article_id) in artifacts:
                collaboration[user_id][article_id] = 1
            else:
                print(f"warning: {article_id} is not in artifacts!")

        article_ids_inview = row['article_ids_inview']
        if user_id not in collaboration:
            collaboration[user_id] = {}
        for article_id in article_ids_inview:
            article_id = str(article_id)
            if int(article_id) in artifacts:
                if article_id not in collaboration[user_id]:
                    collaboration[user_id][article_id] = 0
            else:
                print(f"warning: {article_id} is not in artifacts!")
    
    for i in tqdm(range(len(history))):
        row = history.iloc[i]
        user_id = str(row['user_id'])
        article_id_fixed = row['article_id_fixed']
        scroll_percentage_fixed = row['scroll_percentage_fixed']
        read_time_fixed = row['read_time_fixed']

        for j in range(len(article_id_fixed)):
            article_id = article_id_fixed[j]
            article_id = str(article_id)
            read_time = read_time_fixed[j]
            # 确保文章的bert embedding存在
            if int(article_id) not in artifacts:
                print(f"warning: {article_id} is not in artifacts!")
                continue
            scroll_percentage = scroll_percentage_fixed[j]

            # 只有浏览进度超过10%，或者阅读时间超过2s的文章，才被看作正向交互
            if (not math.isnan(scroll_percentage) and scroll_percentage > 0.1) or (not math.isnan(read_time) and read_time >= 2.0):
                collaboration[user_id][article_id] = 1
            #elif not math.isnan(scroll_percentage):
            #    print(f"user {user_id} article {article_id} scroll precent less then 0.1 ({scroll_percentage}), so mark as unlike (read time: {read_time})")

    with open(output_path, "w") as f:
        json.dump(collaboration, f, indent=4)

if __name__ == '__main__':
    # gen_collaborative_data(
    #     "data/ebnerd_demo/validation/behaviors.parquet", 
    #     "data/ebnerd_demo/validation/history.parquet",
    #     "data/FacebookAI_xlm_roberta_base/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet",
    #     "demo_validation.json")

    # gen_collaborative_data(
    #     "data/ebnerd_demo/train/behaviors.parquet", 
    #     "data/ebnerd_demo/train/history.parquet",
    #     "data/FacebookAI_xlm_roberta_base/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet",
    #     "demo_train.json")

    # gen_collaborative_data(
    #     "data/ebnerd_large/validation/behaviors.parquet", 
    #     "data/ebnerd_large/validation/history.parquet",
    #     "data/FacebookAI_xlm_roberta_base/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet",
    #     "large_validation.json")

    # gen_collaborative_data(
    #     "data/ebnerd_large/train/behaviors.parquet", 
    #     "data/ebnerd_large/train/history.parquet",
    #     "data/FacebookAI_xlm_roberta_base/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet",
    #     "large_train.json")

    gen_collaborative_data(
        "data/ebnerd_small/validation/behaviors.parquet", 
        "data/ebnerd_small/validation/history.parquet",
        "data/FacebookAI_xlm_roberta_base/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet",
        "small_validation.json")

    gen_collaborative_data(
        "data/ebnerd_small/train/behaviors.parquet", 
        "data/ebnerd_small/train/history.parquet",
        "data/FacebookAI_xlm_roberta_base/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet",
        "small_train.json")
