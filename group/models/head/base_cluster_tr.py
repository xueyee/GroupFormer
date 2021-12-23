import torch
import torch.nn as nn
#from group.models.routing_transformer import RoutingTransformerLM,RoutingTransformer
from group.models.transformer_cluster import TransformerEncoderLayer_cluster,TransformerEncoder_cluster,TransformerDecoderLayer2,TransformerDecoder

class Base_cluster_TR(nn.Module):
    def __init__(self, config):
        super(Base_cluster_TR, self).__init__()
        self.input_features = config.input_features
        self.embed_features = config.embed_features

        self.actions_num_classes = config.actions_num_classes
        self.activities_num_classes = config.activities_num_classes

        self.embed_fc = nn.Linear(self.input_features, self.embed_features)



        self.activities_fc = nn.Linear(self.embed_features, self.activities_num_classes)
        self.actions_fc = nn.Linear(self.embed_features, self.actions_num_classes)

        #self.TR=RoutingTransformerLM(num_tokens=1024,dim=1024,heads=8,depth=2,max_seq_len=12,causal=True,emb_dim=128,weight_tie=False,dim_head=128,attn_dropout=0.1,attn_layer_dropout=0.,ff_dropout=0.1,layer_dropout=0.,window_size=4,n_local_attn_heads=0,reversible=True,ff_chunks=1,ff_glu=False)
        encoder_layer_actor = TransformerEncoderLayer_cluster(self.embed_features, config.Nhead,total_size=config.total_size, window_size=config.window_size, dropout=config.dropout_porb,
                                                      normalize_before=True)
        encoder_norm_actor = nn.LayerNorm(self.embed_features)
        self.TR = TransformerEncoder_cluster(encoder_layer_actor, num_layers=config.num_encoder_layers,
                                                norm=encoder_norm_actor)

    def forward(self, x):
        B, T, N, F = x.shape
        #print(self.input_features)
        #print(x.shape)
        assert self.input_features == F
        x = x.reshape(-1, self.input_features)
        x = self.embed_fc(x)
        x=x.reshape(B*T,N,-1)

        x,aux_x=self.TR(x)
        x= x.reshape(-1,self.embed_features)
        #token = x.mean(dim=1,keepdim=True)
        actions_scores = self.actions_fc(x)
        actions_scores = actions_scores.reshape(B, T, N, -1)
        actions_scores = actions_scores.mean(dim=1).reshape(B*N, -1)



        activities_scores = self.activities_fc(x)
        activities_scores = activities_scores.reshape(B,T,N,-1).mean(dim=1)
        activities_scores=activities_scores.mean(1)
        return actions_scores, activities_scores,aux_x
