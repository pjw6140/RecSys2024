import torch

def validate(dataloader, model, device):
    with torch.no_grad():
        correct_count = 0
        total_count = 0
        for batch, (users, items, labels, item_emb) in enumerate(dataloader):
            users = users.to(device)
            items = items.to(device)
            labels = labels.float().to(device)
            item_emb = item_emb.to(device)
            
            predicts = model(users, items, item_emb)
            predicts[predicts >= 0.5] = 1.
            predicts[predicts < 0.5] = 0.

            batch_correct_count = (predicts == labels).sum().item()

            correct_count += batch_correct_count
            total_count += len(labels)
    #print(correct_count, total_count)
    return correct_count / total_count

if __name__ == '__main__':
    from dataset import RecSysDataset
    from NeMF import NeMF
    from id_mapping import IDMapping
    validate_dataset = RecSysDataset("small_validation.json", "small_id_mapping.pickle", "data/FacebookAI_xlm_roberta_base/FacebookAI_xlm_roberta_base/xlm_roberta_base.parquet")
    from torch.utils.data import DataLoader
    dataloader = DataLoader(validate_dataset, batch_size=128)

    num_users = 15143
    num_items = 10075
    gmf_emb_dim = 64
    mlp_user_emb_dim = 64
    mlp_item_emb_dim = 768
    mlp_layer_dims = [512,512,64]

    model = NeMF(num_users, num_items, gmf_emb_dim, mlp_user_emb_dim, mlp_item_emb_dim, mlp_layer_dims)
    model = model.to('cuda')
    accuracy = validate(dataloader, model, device='cuda')
    print(accuracy)