import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from pathlib import Path
from .backbone import BackboneBuilder
from .head import HeadBuilder
from roi_align.roi_align import RoIAlign

import os.path as osp


from group.utils.utils import load_model, load_DDPModel
from group.utils.log_helper import init_log

from group.utils.pos_encoding import spatialencoding2d

init_log('group')
logger = logging.getLogger('group')

class ModelBuilder(nn.Module):
    def __init__(self, config):
        super(ModelBuilder, self).__init__()
        self.backbone = BackboneBuilder(config.structure.backbone)
        self.backbone2=BackboneBuilder(config.structure.flow_backbone)
        self.roi_align = RoIAlign(config.structure.crop_h, config.structure.crop_w)
        #self.flow_roi_align = RoIAlign(14, 14)
        self.global_head=HeadBuilder(config.structure.acy_head)
        self.flow_global_head = HeadBuilder(config.structure.flow_head)
        #self.early_head=HeadBuilder(config.structure.early_head)
        self.head = HeadBuilder(config.structure.head)
        self.pose_head = HeadBuilder(config.structure.pose_head)
        #self.flow_head = HeadBuilder(config.structure.head)

        #self.bbox_fc = nn.Linear(12544,256)
        self.bbox_fc = nn.Sequential(nn.Linear(12544,1024),nn.Linear(1024,256))
        self.pose_fc = nn.Sequential(nn.Linear(34,1024),nn.Linear(1024,256))
        self.flow_fc = nn.Sequential(nn.Linear(12544,1024),nn.Linear(1024,256))
        #self.flow_fc = nn.Linear(12544,256)
 
        #self.embed_fc_1 = nn.Linear(512,256)
        #self.embed_fc_2 = nn.Linear(512,256)

        self.spatial_encoding = config.structure.spatial_encoding
        assert  (self.spatial_encoding in [True,False])
        self.load_checkpoint(config)

    def load_checkpoint(self, config):
        if config.checkpoint is not None:
            assert osp.exists(config.checkpoint), 'checkpoint file does not exist'
            logger.info("load check point: " + str(config.checkpoint))
            load_DDPModel(self, str(config.checkpoint))

        # import pdb
        # pdb.set_trace()
        # if config.test.evaluate or config.test_img:
        #     if config.test.load_epoch:
        #         checkPointName = 'checkpoint_' + str(config.test.load_epoch) + '.pth.tar'
        #         checkPointPath = checkPointSavePath.joinpath(checkPointName)
        #         logger.info("load Check point: " + str(checkPointPath))
        #         # load_model(model, str(checkPointPath))
        #         load_DDPModel(self, str(checkPointPath))
        #     elif config.test.load_path:
        #         checkPointPath = config.test.load_path
        #         logger.info("load Check point: " + str(checkPointPath))
        #         load_model(self, str(checkPointPath))
        # else:
        #     if config.train.resume:
        #         if not Path(config.train.resume).exists():
        #             logger.error(config.train.resume + 'is not exists!')
        #         load_model(self, config.train.resume)
        #     elif config.train.pretrain:
        #         if not Path(config.train.pretrain).exists():
        #             logger.error(config.train.pretrain + 'is not exists!')
        #         load_model(self, config.train.pretrain)

    def forward(self, imgs, gt_boxes, poses, flows):
        #imgs, gt_boxes = x
        B, T, C, H ,W =imgs.shape
        FH,FW = flows.shape[-2:]
        N = gt_boxes.shape[2]

        imgs = imgs.reshape(B*T,C,H,W)
        flows = flows.reshape(B,T,2,FH,FW).permute(0,2,1,3,4).contiguous()
        #global-feat=B*T,768,43,78;imgs_feat:B*T,256,87,157
        global_feat,imgs_features = self.backbone(imgs)
        #flows:14,256,12,20
        #global_flow:2,1,1024,6,10
        flows,global_flow=self.backbone2(flows)
        #flows=flows.permute(0,2,1,3,4).contiguous().reshape(B*T,256,12,20)
        global_flow=global_flow.reshape(B,T,256,6,10)

        #global_feat=self.early_head(global_feat)


        OC, OH, OW = imgs_features.shape[-3:]
        device = imgs_features.device
        if self.spatial_encoding is True:
            sp = spatialencoding2d(OC, OH, OW, device).detach()
            imgs_features+=sp
            #imgs_features=torch.cat((imgs_features,sp),dim=1)
        flows = F.interpolate(flows, size=[OH,OW], mode='bilinear', align_corners=True)
        #roi align

        boxes_in_flat = torch.reshape(gt_boxes, (B*T*N, 4))

        boxes_scale = torch.zeros_like(boxes_in_flat)
        boxes_scale[:, 0], boxes_scale[:, 1], boxes_scale[:, 2], boxes_scale[:, 3] = OW, OH, OW, OH
        boxes_in_flat *= boxes_scale


        boxes=boxes_in_flat.reshape(B,T*N,4)
        boxes.requires_grad=False


        boxes_idx = torch.arange(B*T, device=boxes_in_flat.device, dtype=torch.int).reshape(-1,1).repeat(1,N) # B*T, N
        boxes_idx_flat=torch.reshape(boxes_idx, (B*T*N,))

        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        boxes_features = self.roi_align(imgs_features,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N,D,K,K
        flows_features = self.roi_align(flows,
                                        boxes_in_flat,
                                        boxes_idx_flat)
        #boxes_features = boxes_features.reshape(B*T*N,D,K,K)
        #boxes_features = self.early_head(boxes_features)
        boxes_features=boxes_features.reshape(B*T*N,-1)
        boxes_features=self.bbox_fc(boxes_features).reshape(B,T,N,-1)
        
        poses=poses.reshape(B*T*N,-1)
        poses_features=self.pose_fc(poses).reshape(B,T,N,-1)

        flows_features=flows_features.reshape(B*T*N,-1) 
        flows_features=self.flow_fc(flows_features).reshape(B,T,N,-1)

        #flows_features=flows_features.reshape(B,T,N,-1)
        #L,B*T,C
        #poses = poses.reshape(B,T*N,-1)
        poses_token=poses_features.permute(2,0,1,3).contiguous().reshape(N,B*T,-1).mean(0,keepdim=True)
        global_feat=global_feat.reshape(B,T,768,43,78)
        global_token=self.global_head(global_feat)

        global_flow=self.flow_global_head(global_flow)
        global_token=torch.cat([global_token,global_flow],dim=2)
        boxes_features=torch.cat([boxes_features,flows_features],dim=3)

        #boxes_features=self.embed_fc_1(boxes_features)
        #global_token=self.embed_fc_2(global_token)
        actions_scores1, activities_scores1,aux_loss1 = self.head(boxes_features,global_token)
        actions_scores2, activities_scores2,aux_loss2 = self.pose_head(poses_features,poses_token)
        #actions_scores3, activities_scores3,aux_loss3 = self.flow_head(flows_features,global_token)
        
        #return actions_scores1,activities_scores1,aux_loss1
        return [actions_scores1,actions_scores2], [activities_scores1,activities_scores2], aux_loss1+aux_loss2




