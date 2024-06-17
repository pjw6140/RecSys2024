import torch
import pandas as pd
from tqdm import tqdm

model = torch.load("checkpointS/cp38.pt").to("cpu")
model.eval()

output = open("data/predict.csv", 'w')
#output.write('user_id,click_article_id,ratings\n')
output.write('ID,ratings\n')

test_data = pd.read_csv("data/Tianchi_Test_uid_iid.csv", header=0)
num_rows = len(test_data)
for index in tqdm(range(num_rows)):
    row = test_data.iloc[index]
    user_id = row["user_id"]
    item_id = row["click_article_id"]

    user = torch.Tensor([user_id]).int()
    item = torch.Tensor([item_id]).int()

    with torch.no_grad():
        score = model(user, item).item()
    
    if score >= 0.5:
        score = 1
    else:
        score = 0

    #output.write(f"{int(user_id)},{int(item_id)},{score}\n")
    output.write(f"{index},{score}\n")

output.close()