import torch
import torch.nn as nn

import logging
from pathlib import Path
from .backbone import BackboneBuilder
from .head import HeadBuilder
from roi_align.roi_align import RoIAlign


from group.utils.utils import load_model, load_DDPModel
from group.utils.log_helper import init_log

from group.utils.pos_encoding import spatialencoding2d

init_log('group')
logger = logging.getLogger('group')

class ModelBuilder(nn.Module):
    def __init__(self, config, checkPointSavePath):
        super(ModelBuilder, self).__init__()
        self.backbone = BackboneBuilder(config.structure.backbone)
        self.roi_align = RoIAlign(config.structure.crop_h, config.structure.crop_w)
        #self.global_head=HeadBuilder(config.structure.acy_head)
        #self.early_head=HeadBuilder(config.structure.early_head)
        self.head = HeadBuilder(config.structure.head)
        self.spatial_encoding = config.structure.spatial_encoding
        assert  (self.spatial_encoding in [True,False])
        self.load_checkpoint(config, checkPointSavePath)

    def load_checkpoint(self, config, checkPointSavePath):
        if config.evaluate or config.test_img:
            if config.test.load_epoch:
                checkPointName = 'checkpoint_' + str(config.test.load_epoch) + '.pth.tar'
                checkPointPath = checkPointSavePath.joinpath(checkPointName)
                logger.info("load Check point: " + str(checkPointPath))
                # load_model(model, str(checkPointPath))
                load_DDPModel(self, str(checkPointPath))
            elif config.test.load_path:
                checkPointPath = config.test.load_path
                logger.info("load Check point: " + str(checkPointPath))
                load_model(self, str(checkPointPath))
        else:
            if config.train.resume:
                if not Path(config.train.resume).exists():
                    logger.error(config.train.resume + 'is not exists!')
                load_model(self, config.train.resume)
            elif config.train.pretrain:
                if not Path(config.train.pretrain).exists():
                    logger.error(config.train.pretrain + 'is not exists!')
                load_model(self, config.train.pretrain)

    def forward(self, imgs, gt_boxes):
        #imgs, gt_boxes = x
        B, T, C, H ,W =imgs.shape
        N = gt_boxes.shape[2]

        imgs = imgs.reshape(B*T,C,H,W)
        #global-feat=B*T,768,43,78;imgs_feat:B*T,256,87,157
        imgs_features = self.backbone(imgs)
        #global_feat=self.early_head(global_feat)


        OC, OH, OW = imgs_features.shape[-3:]
        device = imgs_features.device
        if self.spatial_encoding is True:
            sp = spatialencoding2d(OC, OH, OW, device).detach()
            imgs_features+=sp
            #imgs_features=torch.cat((imgs_features,sp),dim=1)
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

        #boxes_features = boxes_features.reshape(B*T*N,D,K,K)
        #boxes_features = self.early_head(boxes_features)
        boxes_features=boxes_features.reshape(B,T,N,-1)
        #L,B*T,C
        #global_feat=global_feat.reshape(B,T,768,43,78)
        #global_token=self.global_head(global_feat)

        actions_scores, activities_scores,aux_loss = self.head(boxes_features)

        return actions_scores, activities_scores,aux_loss




