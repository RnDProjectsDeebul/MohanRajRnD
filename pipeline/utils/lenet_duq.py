import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, dropout=True):
        super().__init__()
        self.use_dropout = dropout
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(20000, 500)
        self.fc2 = nn.Linear(500, 256)

    def compute_features(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 1))
        x = F.relu(F.max_pool2d(self.conv2(x), 1))
        x = x.reshape(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNN_DUQ(Model):
    def __init__(self, num_classes, embedding_size, learnable_length_scale, length_scale, gamma,):
        super().__init__()
        self.gamma = gamma
        self.W = nn.Parameter(torch.normal(torch.zeros(embedding_size, num_classes, 256), 0.05))
        self.register_buffer("N", torch.ones(num_classes) * 12)
        self.register_buffer("m", torch.normal(torch.zeros(embedding_size, num_classes), 1))
        self.m = self.m * self.N.unsqueeze(0)
        if learnable_length_scale:
            self.sigma = nn.Parameter(torch.zeros(num_classes) + length_scale)
        else:
            self.sigma = length_scale

            
    def update_embeddings(self, x, y):
        z = self.last_layer(self.compute_features(x))
        # normalizing value per class, assumes y is one_hot encoded
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)
        # compute sum of embeddings on class by class basis
        features_sum = torch.einsum("ijk,ik->jk", z, y)
        self.m = self.gamma * self.m + (1 - self.gamma) * features_sum

        
    def last_layer(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)
        return z

    
#     def output_layer(self, z):
#         embeddings = self.m / self.N.unsqueeze(0)
#         diff = z - embeddings.unsqueeze(0)
#         distances = (-(diff**2)).mean(1).div(2 * self.sigma**2).exp()
#         return distances

    def output_layer(self, z):
        embeddings = self.m / self.N.unsqueeze(0)
        diff = z - embeddings.unsqueeze(0)
        
        a1 = self.sigma**2
        
        a2 = -(diff**2)
        a3 = (a2).mean(1)
        a4 = a3.div(2 * a1)
        a5 = a4.exp()

        return a5, embeddings, z, diff, a1, a2, a3, a4
    
    
    def forward(self, x):
        z = self.last_layer(self.compute_features(x))
        y_pred, embeddings, z, diff, a1, a2, a3, a4 = self.output_layer(z)
        return y_pred, embeddings, z, diff, a1, a2, a3, a4
