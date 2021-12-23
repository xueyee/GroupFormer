import torch
import torch.nn as nn
def relative_logits(q,H,W,num_head,dkh):
	#compute relative in width dimension
	rel_embedding_w=torch.normal(mean=dkh**0.5,std=1,size=(2*W-1,dkh),device=q.device)
	rel_embedding_w=nn.Parameter(rel_embedding_w,requires_grad=True)

	rel_logits_w=relative_logits_1d(q,rel_embedding_w,H,W,num_head,[0,1,2,4,3,5])


	rel_embedding_h=torch.normal(mean=dkh**0.5,std=1,size=(2*H-1,dkh),device=q.device)
	rel_logits_h=relative_logits_1d(q.permute(0,1,3,2,4).contiguous(),
									rel_embedding_h,W,H,num_head,[0,1,4,2,5,3])

	return rel_logits_h,rel_logits_w

def rel_to_abs(x):
	B,Nh,L,_=x.shape
	col_pad=torch.zeros((B,Nh,L,1),device=x.device)
	#B,Nh,L,2*L
	x=torch.cat([x,col_pad],dim=3)
	flat_x=torch.reshape(x,[B,Nh,L*2*L])
	flat_pad=torch.zeros((B,Nh,L-1),device=x.device)
	flat_x_padded=torch.cat([flat_x,flat_pad],dim=2)
	final_x=torch.reshape(flat_x_padded,[B,Nh,L+1,2*L-1])
	final_x=final_x[:,:,:L,L-1:]
	return final_x
#q-->B,num_head,H,W,dkh or dvh
#rel_k--->2*W-1,dkh
#output---->B,num_head,H*W,H*W
def relative_logits_1d(q,rel_k,H,W,num_head,transpose_mask):
	#B,num_head,H,W,2*W-1
	rel_logits=torch.einsum('bhxyd,md->bhxym',q,rel_k)
	rel_logits=rel_logits.reshape(-1,num_head*H,W,2*W-1)
	#B,num_head*H,W,W

	rel_logits=rel_to_abs(rel_logits)
	#B,num_head,H,H,W,W
	rel_logits=rel_logits.reshape(-1,num_head,H,W,W).unsqueeze(3)
	rel_logits=rel_logits.repeat(1,1,1,H,1,1)
	#0,1,2,4,3,5
	rel_logits=rel_logits.permute(transpose_mask).contiguous()
	#B*num_head,HW,HW
	rel_logits=rel_logits.reshape(-1,H*W,H*W)
	return rel_logits

