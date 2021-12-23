import torch
import torch.nn as nn
from torch.autograd import Variable

class edge_loss(nn.Module):
    def __init__(self):
        super(edge_loss, self).__init__()
        self.criterion = nn.BCELoss(reduction='sum')

    def forward(self, all_scores, labels, masks, num_classes):
        total_loss = 0
        masks = masks.view(-1, 1).repeat(1, num_classes).float()
        masks = masks.detach()
        labels = labels.view((-1, 1))
        one_hot_labels = Variable(torch.zeros((labels.size(0), num_classes)).float()).cuda()
        one_hot_labels.scatter_(1, labels, 1.0)
        # loss_num = len(all_scores)

        for scores in all_scores[-1:]:
            scores = scores.view(-1, num_classes)
            raw_losses = self.criterion(scores * masks, one_hot_labels * masks)
            losses = raw_losses / torch.sum(masks)
            total_loss += losses

        return total_loss