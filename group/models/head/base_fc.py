import torch
import torch.nn as nn

class BaseFC(nn.Module):
    def __init__(self, config):
        super(BaseFC, self).__init__()
        self.input_features = config.input_features
        self.embed_features = config.embed_features
        dropout_porb = config.dropout_porb
        self.actions_num_classes = config.actions_num_classes
        self.activities_num_classes = config.activities_num_classes

        self.embed_fc = nn.Linear(self.input_features, self.embed_features)
        self.dropout = nn.Dropout(p=dropout_porb)

        self.activities_fc = nn.Linear(self.embed_features, self.activities_num_classes)
        self.actions_fc = nn.Linear(self.embed_features, self.actions_num_classes)

    def forward(self, x):
        B, T, N, F = x.shape
        #print(self.input_features)
        #print(x.shape)
        assert self.input_features == F
        x = x.reshape(-1, self.input_features)
        x = self.embed_fc(x)
        x = self.dropout(x)

        actions_scores = self.actions_fc(x)
        actions_scores = actions_scores.reshape(B, T, N, -1)
        actions_scores = actions_scores.mean(dim=1).reshape(B*N, -1)

        x = x.reshape(B,T,N,-1)
        x,_ = torch.max(x, dim=2)
        x = x.reshape(B*T, -1)
        activities_scores = self.activities_fc(x)
        activities_scores = activities_scores.reshape(B,T,-1).mean(dim=1)

        return actions_scores, activities_scores







