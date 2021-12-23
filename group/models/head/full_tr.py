import torch
import torch.nn as nn

from group.models.transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer2, TransformerDecoder

class FullTR(nn.Module):
    def __init__(self, config):
        super(FullTR, self).__init__()
        self.input_features = config.input_features
        self.embed_features = config.embed_features

        self.actions_num_classes = config.actions_num_classes
        self.activities_num_classes = config.activities_num_classes

        self.embed_fc = nn.Linear(self.input_features, self.embed_features)

        encoder_layer = TransformerEncoderLayer(self.embed_features, config.Nhead, dropout=config.dropout_porb, normalize_before=True)
        encoder_norm = nn.LayerNorm(self.embed_features)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers, norm=encoder_norm)
        
        decoder_layer = TransformerDecoderLayer2(self.embed_features, config.Nhead, dropout=config.dropout_porb, normalize_before=True)
        decoder_norm = nn.LayerNorm(self.embed_features)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers, norm=decoder_norm)

        self.query = nn.Parameter(torch.randn(self.embed_features), requires_grad=True)

        self.activities_fc = nn.Linear(self.embed_features, self.activities_num_classes)
        self.actions_fc = nn.Linear(self.embed_features, self.actions_num_classes)

    def forward(self, x):
        B, T, N, F = x.shape
        #print(self.input_features)
        #print(x.shape)
        assert self.input_features == F
        x = x.reshape(-1, self.input_features)
        x = self.embed_fc(x)
        
        x = x.reshape(B*T,N,-1).permute(1,0,2) #(N,B*T,-1)
        memory = self.encoder(x)

        query = self.query.repeat(B*T,1).reshape(1, B*T, -1) 
        tgt = self.decoder(query, memory)

        
        memory = memory.permute(1,0,2).reshape(-1,self.embed_features)  #(B*T*N,e_F)
        tgt = tgt.reshape(-1,self.embed_features) #(B*T,e_F)  
 
                

        actions_scores = self.actions_fc(memory)
        actions_scores = actions_scores.reshape(B, T, N, -1)
        actions_scores = actions_scores.mean(dim=1).reshape(B*N, -1)

        activities_scores = self.activities_fc(tgt)
        activities_scores = activities_scores.reshape(B,T,-1).mean(dim=1)

        return actions_scores, activities_scores
