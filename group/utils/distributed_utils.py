import os
import math
import torch
import torch.distributed as dist
from torch.nn import Module
from torch.utils.data.distributed import DistributedSampler
import multiprocessing as mp
from torch.utils.data.sampler import Sampler
import numpy as np

class DistModule(Module):
    def __init__(self, module):
        super(DistModule, self).__init__()
        self.module = module
        broadcast_params(self.module)
        #dist._clear_group_cache()
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    def train(self, mode=True):
        #dist._clear_group_cache()
        super(DistModule, self).train(mode)
        self.module.train(mode)

def average_gradients(model):
    """ average gradients """
    """ test network """
    """
    for name, param in model.named_parameters():
        print(name,type(param.grad))
        dist.all_reduce(param.grad.data)
    """
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data)

def broadcast_params(model):
    """ broadcast model parameters """
    for p in model.state_dict().values():
        dist.broadcast(p, 0)

def dist_init(port):
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.get_context('spawn')
        # mp.set_start_method('spawn')
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id%num_gpus)

    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1,pos2)].replace('[', '')
    addr = node_list[10:].replace('-', '.')
    print(addr)
    os.environ['MASTER_PORT'] = port
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

def dist_init_pytorch(port):
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size



class CustomSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = list(torch.randperm(len(self.dataset), generator=g))
        offset = self.num_samples * self.rank
        indices = indices[offset:min(offset + self.num_samples, len(indices))]
        return iter(indices)

    def __len__(self):
        return self.num_samples

class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        world_size (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within world_size.
    """

    def __init__(self, dataset, world_size=None, rank=None, round_up=False):
        if world_size is None:
            world_size = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.round_up = round_up
        self.epoch = 0

        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.world_size))
        if self.round_up:
            self.total_size = self.num_samples * self.world_size
        else:
            self.total_size = len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = list(torch.randperm(len(self.dataset), generator=g))
        # indices = list(range(len(self.dataset)))
        # indices = self.dataset
        # print ('all: ', indices)
        # print (indices[:(self.total_size - len(indices))])
        # add extra samples to make it evenly divisible
        if self.round_up:
            indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:min(offset + self.num_samples, self.total_size)]
        if self.round_up or (not self.round_up and self.rank < self.world_size - 1):
            assert len(indices) == self.num_samples

        # print (self.rank, len(indices))

        # print (self.rank, indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedGivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, world_size=None, rank=None, last_iter=-1, shuffle_strategy=0):
        if world_size is None:
            world_size = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        assert rank < world_size
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter
        self.shuffle_strategy = shuffle_strategy

        self.total_size = self.total_iter * self.batch_size

        self.call = 0

    def gen_s2(self):
        length = len(self.dataset)
        print('using shuffle strategy 2, initializing index...')

        for i in range((self.last_iter + 1) * self.batch_size, self.total_size):
            yield np.random.randint(0, length)

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            if self.shuffle_strategy == 2:
                np.random.seed(self.rank)  # set different random seed
                return self.gen_s2()
            else:
                self.indices = self.gen_new_list()
                return iter(self.indices[(self.last_iter + 1) * self.batch_size:])
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def gen_new_list(self):

        # each process shuffle independently
        if self.shuffle_strategy == 0:

            np.random.seed(self.rank)

            indices = np.arange(len(self.dataset))
            indices = indices[:self.total_size]
            num_repeat = (self.total_size - 1) // indices.shape[0] + 1
            indices = np.tile(indices, num_repeat)
            indices = indices[:self.total_size]

            for beg in range(0, self.total_size, len(self.dataset)):
                end = min(beg + len(self.dataset), self.total_size)
                np.random.shuffle(indices[beg:end])

        # each process shuffle all list with same seed, and pick one piece according to rank
        elif self.shuffle_strategy == 1:

            np.random.seed(0)

            all_size = self.total_size * self.world_size
            indices = np.arange(len(self.dataset))
            indices = indices[:all_size]
            num_repeat = (all_size - 1) // indices.shape[0] + 1
            indices = np.tile(indices, num_repeat)
            indices = indices[:all_size]

            np.random.shuffle(indices)
            beg = self.total_size * self.rank
            indices = indices[beg:beg + self.total_size]

        else:
            raise RuntimeError('Unknown shuffle strategy')

        assert len(indices) == self.total_size

        return indices

    def __len__(self):
        return self.total_size - (self.last_iter + 1) * self.batch_size
