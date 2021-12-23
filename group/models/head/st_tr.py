import torch
import torch.nn as nn

from group.models.transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer2, TransformerDecoder

class STTR(nn.Module):
    def __init__(self, config):
        super(STTR, self).__init__()
        self.input_features = config.input_features
        self.embed_features = config.embed_features

        self.actions_num_classes = config.actions_num_classes
        self.activities_num_classes = config.activities_num_classes

        self.embed_fc = nn.Linear(self.input_features, self.embed_features)

        #actor tr encoder
        encoder_layer_actor = TransformerEncoderLayer(self.embed_features, config.Nhead, dropout=config.dropout_porb, normalize_before=True)
        encoder_norm_actor = nn.LayerNorm(self.embed_features)
        self.encoder_actor = TransformerEncoder(encoder_layer_actor, num_layers=config.num_encoder_layers, norm=encoder_norm_actor)

        #temporal tr encoder
        encoder_layer_temp = TransformerEncoderLayer(self.embed_features, config.Nhead, dropout=config.dropout_porb,
                                                      normalize_before=True)
        encoder_norm_temp = nn.LayerNorm(self.embed_features)
        self.encoder_temp = TransformerEncoder(encoder_layer_temp, num_layers=config.num_encoder_layers,
                                                norm=encoder_norm_temp)


        #self.token = nn.Parameter(torch.randn(self.embed_features), requires_grad=True)
        #self.token_actor = nn.Parameter(torch.Tensor(12,self.embed_features))
        #self.token_temp=
        self.activities_fc_1 = nn.Linear(self.embed_features, self.activities_num_classes)
        self.activities_fc_2 = nn.Linear(self.embed_features, self.activities_num_classes)
        self.actions_fc = nn.Linear(self.embed_features, self.actions_num_classes)

    def forward(self, x):
        B, T, N, F = x.shape
        #print(self.input_features)
        #print(x.shape)
        assert self.input_features == F
        x = x.reshape(-1, self.input_features)
        #B,T,N,C_e
        out = self.embed_fc(x).reshape(B,T,N,-1)
        #1,B*T,-1
        token=out.mean(dim=2,keepdim=True)
        token=token.reshape(B*T,1,-1).permute(1,0,2).contiguous()
        actor_o = out.reshape(B*T,N,-1).permute(1,0,2) #(N,B*T,-1)
        actor_o=torch.cat([token,actor_o])
        #N+1,B*T,-1
        memory_actor = self.encoder_actor(actor_o)
        #B*T,-1
        token=memory_actor[0,:,:].reshape(B*T,-1)
        #B*T*N,-1
        memory_actor=memory_actor[1:,:,:].permute(1,0,2).contiguous().reshape(B*T*N,-1)



        #T,B*N,-1
        temp_o=out.permute(1,0,2,3).contiguous().reshape(T,B*N,-1)
        token_temp=out.mean(dim=1,keepdim=True).reshape(1,B*N,-1)
        temp_o=torch.cat([token_temp,temp_o])
        #T+1,B*N,-1
        memory_temp=self.encoder_temp(temp_o)
        #B*N,-1
        token_temp=memory_temp[0,:,:].reshape(B*N,-1)
        memory_temp=memory_temp[1:,:,:].reshape(T,B,N,-1).permute(1,0,2,3).contiguous().reshape(B*T*N,-1)


        memory=(memory_temp+memory_actor)/2


        actions_scores = self.actions_fc(memory)
        actions_scores = actions_scores.reshape(B, T, N, -1)
        actions_scores = actions_scores.mean(dim=1).reshape(B*N, -1)


        activities_scores_1 = self.activities_fc_1(token)
        activities_scores_1 = activities_scores_1.reshape(B,T,-1).mean(dim=1)

        activities_scores_2=self.activities_fc_2(token_temp)
        activities_scores_2=activities_scores_2.reshape(B,N,-1).mean(dim=1)

        activities_scores=(activities_scores_2+activities_scores_1)/2

        return actions_scores, activities_scores
