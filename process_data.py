import pandas as pd
import numpy as np
import math
import json

def gen_collaborative_data(behaviors_path, history_path, artifacts_path, output_path):
    '''
    把原始数据处理成协同信息（类天池数据集）
    '''
    behaviors = pd.read_parquet(behaviors_path)
    history = pd.read_parquet(history_path)
    artifacts = pd.read_parquet(artifacts_path)
    artifacts = set(artifacts['article_id'])
    
    collaboration = {}
    for i in range(len(behaviors)):
        row = behaviors.iloc[i]
        user_id = str(row['user_id'])
        article_ids_inview = row['article_ids_inview']
        if user_id not in collaboration:
            collaboration[user_id] = {}
        for article_id in article_ids_inview:
            article_id = str(article_id)
            if int(article_id) in artifacts:
                collaboration[user_id][article_id] = 0
            else:
                print(f"warning: {article_id} is not in artifacts!")
    
    for i in range(len(history)):
        row = history.iloc[i]
        user_id = str(row['user_id'])
        article_id_fixed = row['article_id_fixed']
        scroll_percentage_fixed = row['scroll_percentage_fixed']
        read_time_fixed = row['read_time_fixed']

        for j in range(len(article_id_fixed)):
            article_id = article_id_fixed[j]
            article_id = str(article_id)
            read_time = read_time_fixed[j]
            if int(article_id) not in artifacts:
                print(f"warning: {article_id} is not in artifacts!")
                continue
            scroll_percentage = scroll_percentage_fixed[j]

            if (not math.isnan(scroll_percentage) and scroll_percentage > 0.1) or (not math.isnan(read_time) and read_time >= 2.0):
                collaboration[user_id][article_id] = 1
            elif not math.isnan(scroll_percentage):
                print(f"{user_id} {article_id} scroll precent less then 0.1 ({scroll_percentage}), so mark as unlike (read time: {read_time})")

    with open(output_path, "w") as f:
        json.dump(collaboration, f, indent=4)

if __name__ == '__main__':
    gen_collaborative_data(
        "data/ebnerd_demo/validation/behaviors.parquet", 
        "data/ebnerd_demo/validation/history.parquet",
        "data/FacebookAI_xlm_roberta_base/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet",
        "demo_validation.json")
