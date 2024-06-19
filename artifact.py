import pandas as pd

class Artifact:
    def __init__(self, artifacts_path):
        artifacts = pd.read_parquet(artifacts_path)

        self.article_bert_embeddings = {}
        for i in range(len(artifacts)):
            row = artifacts.iloc[i]
            article_id = row.iloc[0]
            embedding = row.iloc[1]
            self.article_bert_embeddings[article_id] = embedding

    def __getitem__(self, article_id):
        return self.article_bert_embeddings[article_id]

if __name__ == '__main__':
    artifact = Artifact("data/FacebookAI_xlm_roberta_base/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet")
    print(artifact[9775402])
    print(type(artifact[9775402]))
    print(artifact[9775402].shape)
    print(artifact[9775402].dtype)