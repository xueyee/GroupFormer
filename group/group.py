import time
import copy
import torch
import random
import logging
import os.path as osp

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from pathlib import Path
from functools import reduce
from easydict import EasyDict
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

from group.models import ModelBuilder
from group.datasets import dataset_entry
from group.loss.loss import LabelSmoothCELoss
from group.utils.distributed_utils import average_gradients, DistModule
from group.utils.log_helper import init_log
from group.utils.evaluation import eval_group
from group.utils.utils import save_checkpoint, AverageMeter, print_speed, load_model, load_DDPModel

init_log('group')
logger = logging.getLogger('group')


class Group():
    def __init__(self, config, work_dir):
        self.config = EasyDict(config)
        self.save_path = Path(work_dir)
        self.start_epoch = 0
        self.total_epochs = self.config.train.scheduler.epochs
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # check default attribute from args
        # args_attr = {
        #     'evaluate': False,
        #     'test_img': False,
        #     'resume': False,
        #     'load_path': None
        # }
        # for k, v in args_attr.items():
        #     if not hasattr(self.config, k):
        #         setattr(self.config, k, v)

        self._build()

    def _build(self):
        self._build_seed()
        self._build_dir()
        self._build_model()
        self._build_optimizer()
        self._build_datasetLoader()
        self._build_scheduler()
        self._build_criterion()
        self._build_tb_logger()

    def _build_seed(self):
        seed = self.config.common.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _build_dir(self):
        self.checkPointPath = self.save_path.joinpath('checkpoints')
        self.saveLogPath = self.save_path.joinpath('log')
        self.saveResultsPath = self.save_path.joinpath('results')
        self.tbSavePath = Path(self.save_path).joinpath('tb_logger')
        self.nartSavePath = self.save_path.joinpath('nart')
        if self.rank == 0:
            if not self.checkPointPath.exists():
                self.checkPointPath.mkdir(parents=True)
            if not self.saveLogPath.exists():
                self.saveLogPath.mkdir(parents=True)
            if not self.saveResultsPath.exists():
                self.saveResultsPath.mkdir(parents=True)
            if not self.tbSavePath.exists():
                self.tbSavePath.mkdir(parents=True)
            if not self.nartSavePath.exists():
                self.nartSavePath.mkdir(parents=True)

    def _build_model(self):
        config = self.config
        model = ModelBuilder(config)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.model = DistributedDataParallel(model.cuda(), device_ids=[self.rank % torch.cuda.device_count()],
                                             find_unused_parameters=True)
        # self.model = DistModule(model.cuda())

    def _build_optimizer(self):
        config = self.config
        model = self.model
        try:
            optim = getattr(torch.optim, config.train.optimizer.type)
        except Exception:
            raise NotImplementedError('not implemented optim method ' +
                                      config.train.optimizer.type)
        optimizer = optim(filter(lambda p: p.requires_grad, model.parameters()), **config.train.optimizer.kwargs)
        self.optimizer = optimizer

    def _build_datasetLoader(self):
        train_loader, val_loader = dataset_entry(self.config.dataset, self.world_size, self.rank,
                                                 evaluate=False, test_img=False)
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _build_scheduler(self):
        config = self.config
        self.lr_scheduler = MultiStepLR(
            self.optimizer,
            milestones=config.train.scheduler.milestones,
            gamma=config.train.scheduler.gamma,
            last_epoch=self.start_epoch - 1)

    def _build_criterion(self):
        config = self.config
        if config.train.criterion == 'ce_loss':
            actions_weight = torch.tensor(config.train.actions_weight).cuda().detach()
            activities_weight = torch.tensor(config.train.activities_weight).cuda().detach()
            self.actions_criterion = nn.CrossEntropyLoss(weight=actions_weight)
            self.activities_criterion = nn.CrossEntropyLoss(weight=activities_weight)
        else:
            raise NotImplementedError('not implemented criterion ' + config.criterion)

    def get_dump_dict(self):
        return {
            'config': copy.deepcopy(self.config),
            'epoch': self.epoch + 1,
            'optimizer': self.lr_scheduler.optimizer.state_dict(),
            'state_dict': self.model.state_dict(),
        }

    def _build_tb_logger(self):
        self.tb_logger = SummaryWriter(self.tbSavePath)


    @staticmethod
    def to_device(input, device="cuda"):
        """Transfer data between devidces"""

        def transfer(x):
            if torch.is_tensor(x):
                return x.to(device=device)
            elif isinstance(x, list) and torch.is_tensor(x[0]):
                return [_.to(device=device) for _ in x]
            return x

        if isinstance(input, dict):
            return {k: transfer(v) for k, v in input.items()}
        return [transfer(k) for k in input]

    def get_batch(self, batch_type='train'):
        """
        Return the batch of the given batch_type.
        The valid batch_type is set in config
        The returned batch will be used to call `forward` function of SpringCommonInterface.
        The first item will be used to forward model like: model(get_batch('train')[0])
        self:
            batch_type: str. default: 'train'. It can also be 'val', 'test' or other custom type.

        Returns:
            a tuple of batch (input, label)
        """
        iter_name = batch_type + '_iterator'
        loader_name = batch_type + '_loader'

        def get_iterator():
            loader = getattr(self, loader_name)
            iterator = iter(loader)
            return iterator

        if not hasattr(self, iter_name):
            iterator = get_iterator()
            setattr(self, iter_name, iterator)
        else:
            iterator = getattr(self, iter_name)

        try:
            batch = next(iterator)
        except StopIteration as e:  # noqa
            iterator = get_iterator()
            setattr(self, iter_name, iterator)
            batch = next(iterator)
        batch = self.to_device(batch)
        return batch

    def update(self):
        self.lr_scheduler.optimizer.step()

    def forward(self, batch):
        actions, activities, aux_loss = self.model(batch[0], batch[1], batch[4], batch[5])
        target_actions = batch[2].view(-1)
        target_activities = batch[3].view(-1)
        if isinstance(actions, list):
            actions_loss = sum([self.actions_criterion(action, target_actions) for action in actions]) / self.world_size
        else:
            actions_loss = self.actions_criterion(actions, target_actions) / self.world_size
        if isinstance(activities, list):
            activities_loss = sum(
                [self.activities_criterion(activity, target_activities) for activity in activities]) / self.world_size
        else:
            activities_loss = self.activities_criterion(activities, target_activities) / self.world_size
        loss = (actions_loss + activities_loss)

        return actions_loss, activities_loss, aux_loss, loss

    def backward(self, loss, aux_loss):
        reduced_loss = loss.clone()
        reduced_loss2 = aux_loss.clone()
        dist.all_reduce_multigpu([reduced_loss])
        dist.all_reduce_multigpu([reduced_loss2])
        # compute gradient and do SGD step
        self.lr_scheduler.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        aux_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
        average_gradients(self.model)
        return reduced_loss

    def train(self):
        rank = self.rank
        config = self.config
        model = self.model
        # lr_scheduler = self.lr_scheduler
        self.epoch = 0
        for epoch in range(self.start_epoch, self.total_epochs):
            self.epoch = epoch
            self.lr = self.lr_scheduler.get_lr()[0]
            self.train_loader.sampler.set_epoch(epoch)
            self.train_epoch()
            self.val()
            if rank == 0:
                save_checkpoint(self.get_dump_dict(), self.checkPointPath)

    def train_epoch(self):
        model = self.model
        self.train_epoch_iters = len(self.train_loader)
        for batch_idx in range(self.train_epoch_iters):
            # train for one epoch
            batch_time = AverageMeter()
            data_time = AverageMeter()
            actions_losses = AverageMeter()
            activities_losses = AverageMeter()
            aux_losses = AverageMeter()
            losses = AverageMeter()
            end = time.time()
            model.train()
            batch = self.get_batch("train")
            data_time.update(time.time() - end)
            actions_loss, activities_loss, aux_loss, loss = self.forward(batch)
            reduced_loss = self.backward(loss, aux_loss)

            dist.all_reduce(actions_loss)
            dist.all_reduce(activities_loss)
            dist.all_reduce(aux_loss)
            aux_losses.update(aux_loss.item())
            actions_losses.update(actions_loss.item())
            activities_losses.update(activities_loss.item())
            losses.update(reduced_loss.item())

            self.update()
            batch_time.update(time.time() - end)
            end = time.time()
            self.print_info(batch_idx, aux_losses, actions_losses, activities_losses, reduced_loss, batch_time,
                            data_time, losses, self.lr)

        self.lr_scheduler.step()

    def print_info(self, idx, aux_losses, actions_losses, activities_losses, reduced_loss, batch_time, data_time,
                   losses, lr):
        config = self.config
        tb_logger = self.tb_logger
        if idx % config.train.print_freq == 0 and self.rank == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})'
                        ' LR {lr:.8f}\t'
                        'Aux_Loss {aux_losses.val:.8f} ({aux_losses.avg:.8f})\t'
                        'Actions_Loss {action_loss.val:.4f} ({action_loss.avg:.4f})\t'
                        'Activities_Loss {activity_loss.val:.4f} ({activity_loss.avg:.4f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                self.epoch,
                idx,
                self.train_epoch_iters,
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                aux_losses=aux_losses,
                action_loss=actions_losses,
                activity_loss=activities_losses,
                lr=lr))
            curr_step = self.epoch * self.train_epoch_iters + idx + 1
            print_speed(curr_step, batch_time.val, self.total_epochs * self.train_epoch_iters)
            tb_logger.add_scalar('LR', lr, curr_step)
            tb_logger.add_scalar('data_time', data_time.avg, curr_step)
            tb_logger.add_scalar('batch_time', batch_time.avg, curr_step)
            tb_logger.add_scalar('aux_loss_train', aux_losses.avg, curr_step)
            tb_logger.add_scalar('Action_Loss_train', actions_losses.avg, curr_step)
            tb_logger.add_scalar('Activities_Loss_train', activities_losses.avg, curr_step)
            tb_logger.add_scalar('Loss_train', losses.avg, curr_step)
            # tb_logger.add_scalar('Acc', acces.avg, curr_step)

    @torch.no_grad()
    def val(self):
        model = self.model
        tb_logger = self.tb_logger
        rank = self.rank
        model.eval()

        num_actions_true = torch.tensor(0).cuda()
        num_actions_total = torch.tensor(0).cuda()
        num_activities_true = torch.tensor(0).cuda()
        num_activities_total = torch.tensor(0).cuda()

        # confusion_matrix = torch.zeros(8, 8).cuda()

        self.val_epoch_iter = len(self.val_loader)
        for batch_idx in range(self.val_epoch_iter):
            batch = self.get_batch('val')
            target_actions = batch[2].view(-1)
            target_activities = batch[3].view(-1)
            actions, activities, aux_loss = model(batch[0], batch[1], batch[4], batch[5])
            if isinstance(actions, list):
                actions = sum(actions)
            if isinstance(activities, list):
                activities = sum(activities)
            actions = F.softmax(actions, dim=1)
            activities = F.softmax(activities, dim=1)
            pred_actions = torch.argmax(actions, dim=1)
            pred_activities = torch.argmax(activities, dim=1)

            num_actions_true += (pred_actions == target_actions).sum()
            num_actions_total += target_actions.numel()

            num_activities_true += (pred_activities == target_activities).sum()
            num_activities_total += target_activities.numel()

            for pred_acty, acty in zip(pred_activities.view(-1), target_activities):
                acty = int(acty)
                pred_acty = int(pred_acty)
                # confusion_matrix[acty][pred_acty] += 1

        dist.barrier()
        dist.all_reduce(num_actions_true)
        dist.all_reduce(num_actions_total)
        dist.all_reduce(num_activities_true)
        dist.all_reduce(num_activities_total)
        # dist.all_reduce(confusion_matrix)

        if rank == 0:
            actions_acc = num_actions_true.float() / num_actions_total.float()
            activities_acc = num_activities_true.float() / num_activities_total.float()
            tb_logger.add_scalar('Validation Action Accuracy', actions_acc, self.epoch)
            tb_logger.add_scalar('Validation Activity Accuracy', activities_acc, self.epoch)
            print('Epoch: [%d] '
                  'Action Accuracy %.3f%%\t'
                  'Activity Accuracy %.3f%%\t'
                  % (self.epoch, actions_acc * 100, activities_acc * 100))
            # confusion_matrix = confusion_matrix.cpu().numpy()
            #with open('volleyball_confusion_matrix.npy', 'wb') as f:
            #    np.save(f, confusion_matrix)

