import math
import torch


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe



def bboxencoding2d(d_model, positions):
    """
    :param d_model: dimension of the model
    :param positions: the center positions of bboxes, shape as (N, 2) :
    Now wavelengths from 1* 2pi to 10000* 2pi
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    num_pos = positions.shape[0]
    pe = positions.new_zeros((num_pos, d_model))
    # Each dimension use half of d_model
    assert d_model % 2 == 0
    d_model = d_model // 2
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float, device=positions.device) *
                         -(math.log(10000.0) / d_model)))
    pos_x = ((positions[:, 0] + positions[:, 2]) / 2).unsqueeze(1)
    pos_y = ((positions[:, 1] + positions[:, 3]) / 2).unsqueeze(1)
    pe[:, 0:d_model:2] = torch.sin(pos_x.float() * div_term)
    pe[:, 1:d_model:2] = torch.cos(pos_x.float() * div_term)
    pe[:, d_model::2] = torch.sin(pos_y.float() * div_term)
    pe[:, d_model+1::2] = torch.cos(pos_y.float() * div_term)

    return pe

def spatialencoding2d(d_model, height, width, device):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width, device=device)
    # Each dimension use half of d_model
    d_model = d_model // 2
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe



def postion_attn_mask(x,xyxy=4,k=8):
    """
    :param x: Bx(TxN)x4
    :param y: BxTxNx4
    :param k: scalar
    :return: BxMxk
    """
    # B,h,C,N
    B,TN,_=x.shape
    device = x.device
    x=x.reshape(-1,xyxy)
    #import pdb
    #pdb.set_trace()
    x[:,0]=(x[:,0]+x[:,2])/2
    x[:,1]=(x[:,1]+x[:,3])/2
    x=x[:,:2]
    x=x.reshape(B,TN,-1)
    y=x
    # B,M,2
    _,M,_ = y.shape
    #B,M,TN
    inner = -2 * torch.matmul(y, x.permute(0,2,1).contiguous())
    # B,TN,1
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    # B,M,1
    yy = torch.sum(y ** 2, dim=2, keepdim=True)
    pairwise_distance = - inner - yy-xx.permute(0,2,1).contiguous()
    _, idx = pairwise_distance.topk(k=k, dim=-1)  # (batch_size*head, M, k)

    idx_knn = idx.reshape(B, M, k)
    idx = idx_knn  # +idx_base
    # print(idx.shape)
    idx = idx.reshape(B * M, -1)
    # print(idx)
    # print(idx.shape)
    attn_mask = torch.zeros_like(pairwise_distance,device=device).view(B * M, -1)

    for i in range(B * M):
        attn_mask[i, idx[i]] = 1

    attn_mask = attn_mask.reshape(B, M, TN)
    # print(attn_mask)
    # print(attn_mask[:,:,...])
    #pdb.set_trace()
    return attn_mask