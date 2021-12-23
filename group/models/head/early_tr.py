import torch
import torch.nn as nn
from group.models.Relative_2d_Pos import relative_logits
from group.models.transformer_pos import TransformerEncoderLayer,TransformerEncoder,TransformerDecoderLayer2,TransformerDecoder

class early_tr_gl(nn.Module):
    def __init__(self,config):
        super(early_tr_gl, self).__init__()
        self.embed_feat=config.embed_feat
        self.input_feat=config.input_feat
        self.H=config.H
        self.conv2d1=nn.Conv2d(self.input_feat,self.embed_feat,(1,1),1)
        self.conv2d2=nn.Conv2d(self.embed_feat,self.input_feat,(1,1),1)
        encoder_layer_actor = TransformerEncoderLayer(self.embed_feat, config.Nhead,H=config.H,relative_pos=config.relative_pos,
                                                      dropout=config.dropout_porb,
                                                      normalize_before=True)
        encoder_norm_actor = nn.LayerNorm(self.embed_feat)
        self.encoder = TransformerEncoder(encoder_layer_actor, num_layers=config.num_encoder_layers,
                                                norm=encoder_norm_actor)
    #x-->B,T,C,87,157
    def forward(self,x):
        B_T,C,OH,OW=x.shape
        x=x.reshape(B_T,C,OH,OW)
        #B*T,1024,OH,OW
        x=self.conv2d1(x)
        #tgt,B*T,C_o
        x=x.reshape(B_T,self.embed_feat,-1).permute(2,0,1).contiguous()
        out=self.encoder(x)
        #B*T,1024,OH,OW
        out=out.reshape(OH,OW,B_T,self.embed_feat).permute(2,3,0,1)
        out=self.conv2d2(out)
        return out


