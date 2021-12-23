import torch
from torchvision import transforms
import numpy
import random

from group.datasets.volleyball import VolleyballDataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def dataset_entry(config, world_size, rank, evaluate=False, test_img=False):
    img_h = config.img_h
    img_w = config.img_w
    transform_train = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_set = VolleyballDataset(config.train,transform_train)
    val_set = VolleyballDataset(config.val,transform_val)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.train.batch_size, shuffle=False,
                                               num_workers=config.workers, pin_memory=True, sampler=train_sampler, worker_init_fn=seed_worker)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.val.batch_size, shuffle=False,
                                             num_workers=config.workers, pin_memory=True, sampler=val_sampler, worker_init_fn=seed_worker)
    return train_loader, val_loader
