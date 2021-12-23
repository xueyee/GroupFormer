import torch
import torch.nn as nn

from group.models.transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer2, TransformerDecoder,TransformerDecoderLayer
from torch.nn.init import normal_


class tokenizer(nn.Module):
    def __init__(self,config):
        super(tokenizer, self).__init__()
        self.token_num=config.token_num
        self.token_featdim=config.token_feature_dim
        self.in_channel=config.token_inchannel
        self.embed_channel=config.embed_channel
        #C_t,L
        self.spatial_conv=nn.Sequential(
            nn.Conv2d(self.in_channel,self.token_num,kernel_size=1,padding=0,bias=False),
            nn.ReLU(inplace=True)
        )
        #
        self.conv=nn.Sequential(
            nn.Conv2d(self.in_channel,self.embed_channel,kernel_size=1,padding=0),
            nn.ReLU(inplace=True)
        )
        #
        self.softmax=nn.Softmax(dim=2)


    def forward(self,x):
        B,T,C,H,W=x.shape
        L=self.token_num
        C_o=self.embed_channel
        x=x.reshape(B*T,C,H,W)
        #B*T,L,H,W
        adj=self.spatial_conv(x)
        #B*T,C,H,W
        embed_feat=self.conv(x)
        embed_feat=embed_feat.reshape(B*T,C_o,H*W)
        spatial_adj=adj.reshape(B*T,L,H*W)
        spatial_adj=self.softmax(spatial_adj)
        spatial_adj=spatial_adj.permute(0,2,1)
        #B*T,C,L
        embed_feat=torch.matmul(embed_feat,spatial_adj)

        return embed_feat

class global_sttr_block(nn.Module):
    def __init__(self):
        super(global_sttr_block, self).__init__()


class global_sttr(nn.Module):
    def __init__(self,config):
        super(global_sttr, self).__init__()
        self.embed_channel = config.embed_channel
        self.tokenizer=tokenizer(config)
        encoder_layer_actor = TransformerEncoderLayer(self.embed_channel, config.Nhead, dropout=config.dropout_porb,
                                                      normalize_before=True)
        encoder_norm_actor = nn.LayerNorm(self.embed_channel)
        self.encoder = TransformerEncoder(encoder_layer_actor, num_layers=config.num_encoder_layers,
                                                norm=encoder_norm_actor)
    #input=B,T,C,H,W
    def forward(self,x):
        B,T,C,H,W=x.shape
        #B*T,C,L
        token=self.tokenizer(x)
        #L,B*T,C
        token_s=token.permute(2,0,1)
        #L,B*T,C
        token_s=self.encoder(token_s)

        return token_s


