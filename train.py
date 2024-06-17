from NeMF import NeMF
from dataset import RecSysDataset
from id_mapping import IDMapping
from torch.utils.data import DataLoader
import torch
from validation import validate
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, # 日志的级别
                    filename='training.log', 
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s - %(message)s'), # 时间

import os
if not os.path.isdir("checkpoints"):
    os.mkdir("checkpoints")

#user count:  15143
#item count:  10075
num_users = 15143
num_items = 10075
gmf_emb_dim = 64
mlp_user_emb_dim = 64
mlp_item_emb_dim = 768
mlp_layer_dims = [512,512,64]

batch_size = 32
learning_rate = 1e-4

dataset = RecSysDataset("small_train.json", "small_id_mapping.pickle", "data/FacebookAI_xlm_roberta_base/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

validate_dataset = RecSysDataset("small_validation.json", "small_id_mapping.pickle", "data/FacebookAI_xlm_roberta_base/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet")
validate_dataloader = DataLoader(validate_dataset, batch_size=128)

model = NeMF(num_users, num_items, gmf_emb_dim, mlp_user_emb_dim, mlp_item_emb_dim, mlp_layer_dims)
#model = torch.load("checkpoints/cp322.pt")
model = model.to("cuda")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

print("start training", flush=True)

max_validation_accuracy = -1
epoch = 5000
for ei in range(epoch):
    avg_loss = 0.0
    for batch, (users, items, labels, item_emb) in enumerate(dataloader):
        users = users.to("cuda")
        items = items.to("cuda")
        labels = labels.float().to("cuda")
        item_emb = item_emb.to("cuda")
        
        predicts = model(users, items, item_emb)
        loss = torch.nn.functional.binary_cross_entropy(predicts, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        avg_loss = (avg_loss * batch + loss.item()) / (batch + 1)

    accuracy = validate(validate_dataloader, model, device="cuda")
    training_accuracy = validate(dataloader, model, device="cuda")
    print(f"[{datetime.now()}] epoch {ei} Loss: {avg_loss} training accuracy: {training_accuracy} validate accuracy: {accuracy}", flush=True)
    logging.info(f"epoch {ei} Loss: {avg_loss} training accuracy: {training_accuracy} validate accuracy: {accuracy}")

    if accuracy < max_validation_accuracy:
        max_validation_accuracy = accuracy
        torch.save(model, f'checkpoints/best.pt')

    if ei % 50 == 0:
        torch.save(model, f'checkpoints/cp{ei}.pt')

    torch.save(model, f'checkpoints/latest.pt')
