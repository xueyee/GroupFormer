import yaml
import argparse
import os
from pathlib import Path
from pprint import pprint

from group.group import Group
from group.utils.distributed_utils import dist_init, dist_init_pytorch

def parse_args():
    """
    parse args
    :return:args
    """
    new_parser = argparse.ArgumentParser(
        description='PyTorch Density parser..')
    new_parser.add_argument('--config', help='model config file path')
    new_parser.add_argument('--checkpoint', default=None, help='the checkpoint file')
    new_parser.add_argument('--resume', default=None, help='the checkpoint file to resume from')
    new_parser.add_argument('--evaluate', action='store_true', default=False, help='train or test')
    return new_parser.parse_args()

def main():
    # rank, world_size = dist_init("23332")
    rank, world_size = dist_init_pytorch("23332")
    # parse args and load config
    args = parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['checkpoint'] = args.checkpoint
    config['resume'] = args.resume

    config['evaluate'] = args.evaluate
    
    config['basedir'] = os.getcwd() + '/experiments/' + Path(args.config).resolve().stem
    if rank == 0:
        pprint(config)

    group_helper = Group(config, work_dir=config['basedir'])


    if args.evaluate:
        group_helper.epoch=0
        group_helper.val()
    else:
        group_helper.train()

if __name__ == '__main__':
    main()
