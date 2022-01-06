#import h5py
import torch
import torch.distributed as dist
from collections import OrderedDict
import shutil
import math
import os
from pathlib import Path
import logging

from group.utils.log_helper import init_log

init_log('group')
logger = logging.getLogger('group')

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)

def load_model(model, path):
    device = torch.cuda.current_device()
    checkpoint = torch.load(path, map_location = lambda storage, loc: storage.cuda(device))
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = 'backbone.' + k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict,False)
    ckpt_keys = set(new_state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    rank = 0
    rank = dist.get_rank()
    if rank == 0:
        for k in missing_keys:
            if 'num_batches_tracked' in k:
                continue
            print('missing keys from checkpoint {}: {}'.format(path, k))
    return model

def remove_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters share common prefix 'module."""
    def f(x):
        return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_DDPModel(model, path):
    device = torch.cuda.current_device()
    checkpoint = torch.load(path, map_location = lambda storage, loc: storage.cuda(device))
    state_dict = checkpoint['state_dict']
    if not hasattr(model, 'module'):
        state_dict = remove_prefix(state_dict, 'module.')
    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    for k in missing_keys:
        logger.warning('missing keys from checkpoint: {}'.format(k))
    return model

def save_checkpoint_freq(state, expPath='checkpoint', filename='checkpoint.pth.tar', snapshot=None):
    filepath = os.path.join(expPath, filename)
    torch.save(state, filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(expPath, 'checkpoint_{}.pth.tar'.format(state['epoch'])))

def save_checkpoint(state, savePath, filename='checkpoint.pth.tar'):
    filePath = savePath.joinpath(filename)
    torch.save(state, filePath)
    shutil.copyfile(filePath, savePath.joinpath('checkpoint_{}.pth.tar'.format(state['epoch'])))


def print_args(args):
    print('CONFIG', flush=True)
    print('------------------------------------------', flush=True)
    max_tab = 0
    for k, v in vars(args).items():
        max_tab = max(max_tab, math.floor(len(k) / 4))
    for k, v in vars(args).items():
        cur_tab = math.floor(len(k) / 4)
        print(k + '\t'*(max_tab - cur_tab + 1) + str(v))
    print('------------------------------------------', flush=True)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def print_speed(i, i_time, n):
    """print_speed(index, index_time, total_iteration)"""
    average_time = i_time
    remaining_time = (n - i) * average_time
    remaining_day = math.floor(remaining_time / 86400)
    remaining_hour = math.floor(remaining_time / 3600 - remaining_day * 24)
    remaining_min = math.floor(remaining_time / 60 - remaining_day * 1440 - remaining_hour * 60)
    print('Progress: %d / %d [%d%%], Speed: %.3f s/iter, ETA %d:%02d:%02d (D:H:M)\n' % (i, n, i/n*100, average_time, remaining_day, remaining_hour, remaining_min), flush=True)

