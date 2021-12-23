import torch
import torch.nn as nn

class BaseTR(nn.Module):
    def __init__(self, config):
        super(BaseTR, self).__init__()
        self.input_features = config.input_features
        self.embed_features = config.embed_features

        self.actions_num_classes = config.actions_num_classes
        self.activities_num_classes = config.activities_num_classes

        self.embed_fc = nn.Linear(self.input_features, self.embed_features)

        encoder_layers = nn.TransformerEncoderLayer(self.embed_features, config.Nhead, dropout=config.dropout_porb)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=config.num_layers)

        self.token = nn.Parameter(torch.randn(self.embed_features), requires_grad=True)
        self.norm_pre = nn.LayerNorm(self.embed_features)

        self.activities_fc = nn.Linear(self.embed_features, self.activities_num_classes)
        self.actions_fc = nn.Linear(self.embed_features, self.actions_num_classes)

    def forward(self, x):
        B, T, N, F = x.shape
        #print(self.input_features)
        #print(x.shape)
        assert self.input_features == F
        x = x.reshape(-1, self.input_features)
        x = self.embed_fc(x)
        
        x = x.reshape(B*T,N,-1)
        #token = x.mean(dim=1,keepdim=True)
        token = self.token.repeat(B*T,1).reshape(B*T,1,-1)
        tr_in = torch.cat([x, token],dim=1).permute(1,0,2) #(N+1, B*T, e_F)
        
        tr_in = self.norm_pre(tr_in)
        tr_out = self.transformer_encoder(tr_in)
        x = x.reshape(-1, self.embed_features)
        x += tr_out[:N].permute(1,0,2).reshape(-1,self.embed_features)  #(B*T*N,e_F)
        token = tr_out[N].reshape(-1,self.embed_features) #(B*T,e_F)
                

        actions_scores = self.actions_fc(x)
        actions_scores = actions_scores.reshape(B, T, N, -1)
        actions_scores = actions_scores.mean(dim=1).reshape(B*N, -1)

        activities_scores = self.activities_fc(token)
        activities_scores = activities_scores.reshape(B,T,-1).mean(dim=1)

        return actions_scores, activities_scores
