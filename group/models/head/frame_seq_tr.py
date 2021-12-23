import torch
import torch.nn as nn

from group.models.transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer2, TransformerDecoder,TransformerDecoderLayer
from torch.nn.init import normal_

class frame_wise_tr(nn.Module):
    def __init__(self,config):
        super(frame_wise_tr, self).__init__()
        # actor tr encoder
        self.embed_features=config.embed_features
        encoder_layer_actor = TransformerEncoderLayer(self.embed_features, config.Nhead, dropout=config.dropout_porb,
                                                      normalize_before=True)
        encoder_norm_actor = nn.LayerNorm(self.embed_features)
        self.encoder_actor = TransformerEncoder(encoder_layer_actor, num_layers=config.num_encoder_layers,
                                                norm=encoder_norm_actor)

        #actor decoder
        decoder_actor = TransformerDecoderLayer2(self.embed_features, config.Nhead, dropout=config.dropout_porb,
                                                 normalize_before=True)
        decoder_norm_actor = nn.LayerNorm(self.embed_features)
        self.decoder_actor = TransformerDecoder(decoder_actor, num_layers=config.num_decoder_layers,
                                                norm=decoder_norm_actor)

    def forward(self,q,cross_k=None):
        N,B,C=q.shape
        if cross_k is None:
            cross_k=q
        #k=cross_k.permute(1,0,2).contiguous()
        #N
        #q_s=q.permute(1,0,2).contiguous()
        memory=self.encoder_actor(q)

        tgt=self.decoder_actor(cross_k,memory)
        tgt=tgt.reshape(N,B,self.embed_features)
        return tgt,memory



class Frame_TR(nn.Module):
    def __init__(self,config):
        super(Frame_TR, self).__init__()
        self.input_features=config.input_features
        self.embed_features=config.embed_features
        self.num_frames=config.num_frames
        self.actions_num_classes = config.actions_num_classes
        self.activities_num_classes = config.activities_num_classes
        self.embed_fc=nn.Linear(self.input_features,self.embed_features)
        self.frame_tr=nn.ModuleList()
        for i in range(config.num_frames):
            self.frame_tr.append(frame_wise_tr(config))
        decoder_g = TransformerDecoderLayer2(self.embed_features, config.Nhead, dropout=config.dropout_porb,
                                                 normalize_before=True)
        decoder_norm_g = nn.LayerNorm(self.embed_features)
        self.decoder_g = TransformerDecoder(decoder_g, num_layers=config.num_decoder_layers,
                                                norm=decoder_norm_g)
        self.activities_fc = nn.Linear(self.embed_features, self.activities_num_classes)
        self.actions_fc = nn.Linear(self.embed_features, self.actions_num_classes)

    def forward(self,x,global_token):
        B,T,N,F=x.shape
        assert F==self.input_features
        assert T==self.num_frames
        token=global_token.mean(dim=0,keepdim=True)

        x=x.reshape(-1,self.input_features)
        #B,T,N,C_e--->N,T,B,C_e
        x=self.embed_fc(x).reshape(B,T,N,-1).permute(2,1,0,3).contiguous()
        tgt=None
        memory_list=[]
        for i in range(T):
            q=x[:,i,:,:].reshape(N,B,-1)
            #N,B,C
            tgt,memory=self.frame_tr[i](q,tgt)
            tgt=tgt.reshape(N,B,-1)
            memory_list.append(memory)
        #print(tgt.shape)
        tgt=tgt.permute(1,0,2).contiguous().reshape(B*N,-1)

        #activity decoder:
        #N,B*T,-1
        memory=torch.stack(memory_list,dim=0).reshape(T,N,B,-1).permute(1,2,0,3).reshape(N,B*T,-1).contiguous()
        #print(memory.shape)
        #1,B*T,-1
        activities_token=self.decoder_g(token,memory)


        action_scores=self.actions_fc(tgt)
        activities_scores=self.activities_fc(activities_token)
        activities_scores=activities_scores.reshape(B,T,-1).mean(dim=1)
        #print(action_scores.shape,activities_scores.shape)
        return action_scores,activities_scores


