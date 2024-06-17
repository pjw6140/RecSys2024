- ebnerd_demo (20MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip  
(*5,000 users)

—————— ——————

- ebnerd_small (80MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip
(*50,000 users)

—————— ——————

- ebnerd_large (3.0GB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip 

- Articles (140MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/articles_large_only.zip
(Only download the articles from Large)

—————— ——————

- ebnerd_testset (1.5GB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip 

- Example of full submission file (220MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/predictions_large_random.zip
(It’s all random predictions but this file will successfully upload to the leaderboard)

—————— ——————
Artifacts:
- Ekstra-Bladet-word2vec (133MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_word2vec.zip 

- Ekstra_Bladet_image_embeddings (372MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_image_embeddings.zip 

- Ekstra-Bladet-contrastive_vector (341MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_contrastive_vector.zip

- google-bert-base-multilingual-cased (344MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/google_bert_base_multilingual_cased.zip 

- FacebookAI-xlm-roberta-base (341MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/FacebookAI_xlm_roberta_base.zip 

预测的是候选新闻的点击可能性排序
每个impression一行
e.g. 
6451339 [8,1,6,7,4,2,9,5,3]
6451363 [5,4,3,7,6,8,1,2]
6451382 [1,5,3,4,2]
6451383 [7,6,8,9,3,4,2,11,10,1,5]
6451385 [5,3,2,4,7,6,1]
6451411 [9,1,8,3,4,6,2,7,5]

train behaviors
>>> list(data.head())
['impression_id', 'article_id', 'impression_time', 'read_time', 'scroll_percentage', 'device_type', 'article_ids_inview', 'article_ids_clicked', 'user_id', 'is_sso_user', 'gender', 'postcode', 'age', 'is_subscriber', 'session_id', 'next_read_time', 'next_scroll_percentage']

test behaviors
>>> list(data.head())
['impression_id', 'impression_time', 'read_time', 'scroll_percentage', 'device_type', 'article_ids_inview', 'user_id', 'is_sso_user', 'gender', 'postcode', 'age', 'is_subscriber', 'session_id', 'is_beyond_accuracy']

test history (train一样)
>>> list(history.head())
['user_id', 'impression_time_fixed', 'scroll_percentage_fixed', 'article_id_fixed', 'read_time_fixed']

需要解决冷启动问题：训练集里面的用户、文章，在验证集、测试集可能没有
--TODO: 增加近邻匹配，给未训练的人或文章找最近的训练过的对应，用最近的embedding代替预测
-- 用户的近邻匹配用人工特征
-- 文章的近邻匹配用bert embedding