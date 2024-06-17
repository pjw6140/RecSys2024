from turtle import forward
import torch
from torch import nn

class GMFLayer(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, user, item):
        '''
        user: 1D int tensor
        item: 1D int tensor
        '''
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        gmf = user_emb * item_emb
        return gmf

class MLPLayer(nn.Module):
    def __init__(self, num_users, num_items, user_embedding_dim, item_embeddinbg_dim, hidden_dims):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)
        #self.item_embedding = nn.Embedding(num_items, embedding_dim)

        mlp_dims = [user_embedding_dim + item_embeddinbg_dim] + hidden_dims
        self.mlp = nn.Sequential()
        for i in range(len(mlp_dims) - 1):
            self.mlp.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            self.mlp.append(nn.ReLU())

    def forward(self, user, item):
        '''
        user: 1D int tensor
        item: 1D int tensor
        '''
        user_emb = self.user_embedding(user)
        #item_emb = self.item_embedding(item)
        item_emb = item
        mlp_input = torch.concatenate((user_emb, item_emb), dim=1)
        mlp_output = self.mlp(mlp_input)
        return mlp_output

class NeMF(nn.Module):
    def __init__(self, num_users, num_items, gmf_embedding_dim, mlp_user_embedding_dim, mlp_item_embedding_dim, mlp_hidden_dims):
        super().__init__()
        self.gmf_layer = GMFLayer(num_users, num_items, gmf_embedding_dim)
        self.mlp_layer = MLPLayer(num_users, num_items, mlp_user_embedding_dim, mlp_item_embedding_dim, mlp_hidden_dims)
        self.nemf_layer = nn.Linear(gmf_embedding_dim + mlp_hidden_dims[-1], 1, bias=False)

    def forward(self, user, item, item_mlp_emb):
        '''
        user: 1D int tensor
        item: 1D int tensor
        '''
        gmf_output = self.gmf_layer(user, item)
        mlp_output = self.mlp_layer(user, item_mlp_emb)
        nemf_input = torch.concat((gmf_output, mlp_output), dim=1)
        score = self.nemf_layer(nemf_input).squeeze(dim=1)

        return torch.nn.functional.sigmoid(score)