import torch
import torch.nn as nn

from group.models.transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer2, TransformerDecoder,TransformerDecoderLayer
from group.models.transformer_pos_local import TransformerEncoderLayer_pos_local
from group.models.transformer_cluster import TransformerEncoderLayer_cluster,TransformerEncoder_cluster
from torch.nn.init import normal_
from group.utils.pos_encoding import postion_attn_mask


def knn_attn_mask(x, y=None,num_head=8,k=5):
    """
    :param x: Bx head xNxC
    :param y: B x head x M x C
    :param k: scalar
    :return: BxMxk
    """
    # B,h,C,N
    tgt_len, bsz, num_heads_head_dim=x.shape
    head_dim=num_heads_head_dim//num_head
    device = x.device
    assert head_dim*num_head==num_heads_head_dim
    #bsz*head,N,C
    x=x.reshape(tgt_len,bsz*num_head,head_dim).transpose(0,1)
    if y is not None:
        y = y.reshape(tgt_len,bsz*num_head,head_dim).transpose(0,1)
    #bsz*head,M,C
    if y is None:
        y = x
    # B,h,C,M
    _,M,_ = y.shape
    #B*head,M,N
    inner = -2 * torch.matmul(y, x.permute(0,2,1).contiguous())
    # B,h,1,N
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    # B,h,1,M
    yy = torch.sum(y ** 2, dim=2, keepdim=True)
    pairwise_distance = - inner - yy-xx.permute(0,2,1).contiguous()
    # import pdb
    # pdb.set_trace()
    _, idx = pairwise_distance.topk(k=k, dim=-1)  # (batch_size*head, M, k)

    idx_knn = idx.reshape(bsz*num_head, M, k)
    # idx_base=(torch.arange(0,B*head,device=device).view(-1,1,1))*M
    # print(idx_base)
    # print(idx_knn)
    idx = idx_knn  # +idx_base
    # print(idx.shape)
    idx = idx.reshape(bsz * num_head * M, -1)
    # print(idx)
    # print(idx.shape)
    attn_mask = torch.zeros_like(pairwise_distance,device=device).view(bsz * num_head * M, -1)

    for i in range(bsz * num_head * M):
        attn_mask[i, idx[i]] = 1

    attn_mask = attn_mask.reshape(bsz * num_head, M, tgt_len)
    # print(attn_mask)
    # print(attn_mask[:,:,...])
    return attn_mask

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
    def forward(self, x,query):
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

class STTR_block(nn.Module):
    def __init__(self,config,embed_feat):
        super(STTR_block, self).__init__()
        self.embed_features=embed_feat
        #self.k=config.k
        encoder_layer_st = TransformerEncoderLayer_cluster(self.embed_features, config.Nhead,total_size=config.total_size,window_size=config.window_size ,dropout=config.dropout_porb,
                                                      normalize_before=True)
        encoder_norm_st = nn.LayerNorm(self.embed_features)
        self.encoder_st = TransformerEncoder_cluster(encoder_layer_st, num_layers=config.num_encoder_layers,
                                                norm=encoder_norm_st)

        decoder_st= TransformerDecoderLayer2(self.embed_features, config.Nhead, dropout=config.dropout_porb,
                                                normalize_before=True)
        decoder_norm_st = nn.LayerNorm(self.embed_features)
        self.decoder_st = TransformerDecoder(decoder_st, num_layers=config.num_decoder_layers,
                                               norm=decoder_norm_st)



    #input:token--->T,B,C
    #input:x------->N,B,T,C
    def forward(self,x,token):
        #TN,B,C
        #attn_mask=postion_attn_mask(box,k=self.k)
        #TN,B,C
        x,aux_loss=self.encoder_st(x)
        #T,B,C
        tgt = self.decoder_st(token, x)

        return x,tgt,aux_loss




class cluster_nttr(nn.Module):
    def __init__(self,config):
        super(cluster_nttr, self).__init__()
        self.num_STTR_layers=config.num_STTR_layers
        self.input_features=config.input_features
        embed_feat=config.embed_features
        self.actions_num_classes = config.actions_num_classes
        self.activities_num_classes = config.activities_num_classes
        self.embed_features=embed_feat
        self.STTR_module=nn.ModuleList()
        self.embed_fc = nn.Linear(self.input_features, embed_feat)
        #self.token=nn.Parameter(torch.randn(self.embed_features), requires_grad=True)
        #normal_(self.token)
        for i in range(self.num_STTR_layers):
            self.STTR_module.append(STTR_block(config,embed_feat))
        self.activities_fc = nn.Linear(self.embed_features, self.activities_num_classes)
        self.actions_fc = nn.Linear(self.embed_features, self.actions_num_classes)
    def forward(self,x,global_token):
        B, T, N, F = x.shape
        # print(self.input_features)
        # print(x.shape)
        assert self.input_features == F
        x = x.reshape(-1, self.input_features)
        # B,T,N,C_e
        x = self.embed_fc(x).reshape(B, T, N, -1)
        # N,B,T,C_e
        x = x.permute(2,0,1,3)
        #1,B*T,C_e
        #token=self.token.repeat(B*T,1).reshape(1,B*T,-1)
        token=global_token.mean(dim=0,keepdim=True)
        tgt=token.reshape(1,B*T,-1)
        memory=x.reshape(N,B,T,-1).permute(2,0,1,3).reshape(T*N,B,-1)
        #T,B,C
        tgt=tgt.reshape(B,T,-1).permute(1,0,2).contiguous()
        #me:TN,B,C
        #tgt:T,B,C
        #total_loss = torch.tensor(0., requires_grad=True,dtype=x.dtype)
        total_loss = torch.zeros(1,device=x.device ,dtype=x.dtype)
        for i in range(self.num_STTR_layers):
            memory,tgt,loss=self.STTR_module[i](memory,tgt)
            total_loss+=loss
            tgt=tgt.reshape(T,B,-1)

        #total_loss=total_loss.sum()
        #B*T*N,-1
        memory = memory.reshape(T,N,B,-1).permute(2,0,1,3).contiguous().reshape(-1,self.embed_features)
        #import pdb
        #pdb.set_trace()
        tgt=tgt.reshape(T,B,-1).permute(1,0,2).contiguous().reshape(-1,self.embed_features)


        actions_scores=self.actions_fc(memory)
        actions_scores = actions_scores.reshape(B,T,N, -1)
        actions_scores = actions_scores.mean(dim=1).reshape(B * N, -1)


        activities_scores=self.activities_fc(tgt)
        activities_scores=activities_scores.reshape(B,T,-1).mean(dim=1)

        return actions_scores,activities_scores,total_loss