import torch
import torch.nn as nn

from group.models.transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer2, TransformerDecoder,TransformerDecoderLayer
from torch.nn.init import normal_

#input: config,input_feat,embed_feat
#output: memeory--->size(N,B*T,C_e)
#       tgt-------->size(1,B*T,C_e)
class ST_plus_TR_block(nn.Module):
    def __init__(self,config,embed_feat):
        super(ST_plus_TR_block, self).__init__()
        #self.input_features = input_feat
        self.embed_features = embed_feat

        #self.embed_fc = nn.Linear(self.input_features, self.embed_features)

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
        #actor decoder--first
        decoder_actor=TransformerDecoderLayer2(self.embed_features, config.Nhead, dropout=config.dropout_porb, normalize_before=True)
        decoder_norm_actor= nn.LayerNorm(self.embed_features)
        self.decoder_actor = TransformerDecoder(decoder_actor, num_layers=config.num_decoder_layers, norm=decoder_norm_actor)

        #temporal decoder
        decoder_temp=TransformerDecoderLayer2(self.embed_features,config.Nhead,dropout=config.dropout_porb, normalize_before=True)
        decoder_norm_temp=nn.LayerNorm(self.embed_features)
        self.decoder_temp=TransformerDecoder(decoder_temp,num_layers=config.num_decoder_layers, norm=decoder_norm_temp)

    #x--->N,B*T,C
    #query=1,B*T,C
    def forward(self,x,query_actor,query_temp):
        return self.forward_para(x,query_actor,query_temp)
    def forward_ori(self, x,query):
        N,B,T,C=x.shape
        tgt_len,bsz,dim=query.shape
        actor_o = x.reshape(N,B*T,-1) #(N,B*T,-1)
        #N,B*T,-1
        memory_actor = self.encoder_actor(actor_o)

        #T,B*N,-1
        temp_o=x.permute(2,1,0,3).contiguous().reshape(T,B*N,-1)
        #T,B*N,-1
        memory_temp=self.encoder_temp(temp_o)
        #N,B*T,-1
        memory_temp=memory_temp.reshape(T,B,N,-1).permute(2,1,0,3).contiguous().reshape(N,B*T,-1)
        #N,B*T,-1
        memory=self.decoder_actor(memory_actor,memory_temp)
        memory=memory.reshape(N,B,T,-1)
        #1,B*T,C
        tgt=self.decoder_temp(query,memory)
        tgt=tgt.reshape(tgt_len,bsz,dim)

        return memory,tgt

    def forward_para(self,x,query_actor,query_temp):
        N, B, T, C = x.shape

        #tgt_a, bsz_a, dim_a = query_actor.shape
        actor_o = x.reshape(N, B * T, -1)  # (N,B*T,-1)
        #N,B*T,-1
        memory_actor = self.encoder_actor(actor_o)
        #1,B*T,-1
        tgt_actor=self.decoder_actor(query_actor,memory_actor)
        #1,B*T,-1
        tgt_actor=tgt_actor.reshape(1,B*T,-1)

        # T,B*N,-1
        temp_o = x.permute(2, 1, 0, 3).contiguous().reshape(T, B * N, -1)
        # T,B*N,-1
        memory_temp = self.encoder_temp(temp_o)
        #1,B*N,-1
        tgt_temp=self.decoder_temp(query_temp,memory_temp)
        #1,B,N,-1
        tgt_temp=tgt_temp.reshape(1,B*N,-1)

        #temp,actor fusion
        #N,B,T,C
        out_actor=memory_actor.reshape(N,B,T,-1)
        out_temp=memory_temp.reshape(T,B,N,-1).permute(2,1,0,3).contiguous()
        #res
        out=out_temp+out_actor+x

        return out,tgt_actor,tgt_temp




class ST_plus_TR_para(nn.Module):
    def __init__(self,config):
        super(ST_plus_TR_para, self).__init__()
        self.num_STTR_layers=config.num_STTR_layers
        self.input_features=config.input_features
        embed_feat=config.embed_features
        self.actions_num_classes = config.actions_num_classes
        self.activities_num_classes = config.activities_num_classes
        self.embed_features=embed_feat
        self.STTR_module=nn.ModuleList()
        self.embed_fc = nn.Linear(self.input_features, embed_feat)
        for i in range(self.num_STTR_layers):
            self.STTR_module.append(ST_plus_TR_block(config,embed_feat))
        #self.token_actor=torch.tensor(embed_feat,dtype=torch.float32,requires_grad=True)
        #self.token_temp = torch.tensor(embed_feat, dtype=torch.float32, requires_grad=True)
        #normal_(self.token_actor)
        #normal_(self.token_temp)
        self.activities_fc_actor = nn.Linear(self.embed_features, self.activities_num_classes)
        self.activities_fc_temp = nn.Linear(self.embed_features, self.activities_num_classes)
        self.actions_fc = nn.Linear(self.embed_features, self.actions_num_classes)
    def forward(self,x):
        B, T, N, F = x.shape
        # print(self.input_features)
        # print(x.shape)
        assert self.input_features == F
        x = x.reshape(-1, self.input_features)
        # B,T,N,C_e
        x = self.embed_fc(x).reshape(B, T, N, -1)
        #N,B,T,C_e
        x = x.permute(2,0,1,3)
        #1,B*T,C_e
        #self.token.data=self.token.data.repeat(B*T,1).reshape(1,B*T,-1)
        token_actor=x.mean(dim=0,keepdim=True)
        token_actor=token_actor.reshape(1,B*T,-1)
        #N,B,C_e
        token_temp=x.mean(dim=2)
        #1,B*N,C_e
        token_temp=token_temp.permute(1,0,2).contiguous().reshape(1,B*N,-1)
        #B=N,B*T,C_e
        memory=x
        #1,B*T,C_e
        tgt_actor=token_actor
        tgt_temp=token_temp
        #print(tgt.shape)
        for i in range(self.num_STTR_layers):
            memory,tgt_actor,tgt_temp=self.STTR_module[i](memory,tgt_actor,tgt_temp)
            #print(tgt.shape)
        #B*T*N,-1
        #N,B,T,-1
        memory=memory.permute(1,2,0,3).reshape(-1,self.embed_features)
        #B*T,-1
        tgt_actor=tgt_actor.reshape(-1,self.embed_features)
        #B*N,-1
        tgt_temp=tgt_temp.reshape(-1,self.embed_features)


        #actions scores
        actions_scores=self.actions_fc(memory)
        actions_scores = actions_scores.reshape(B, T, N, -1)
        actions_scores = actions_scores.mean(dim=1).reshape(B * N, -1)

        #B*T,num_cls
        activities_scores_actor=self.activities_fc_actor(tgt_actor)
        #B*N,num_cls
        activities_scores_temp=self.activities_fc_temp(tgt_temp)

        #mean fusion
        activities_scores_actor=activities_scores_actor.reshape(B,T,-1).mean(dim=1)
        activities_scores_temp=activities_scores_temp.reshape(B,N,-1).mean(dim=1)
        activities_scores=(activities_scores_temp+activities_scores_actor)/2

        return actions_scores,activities_scores