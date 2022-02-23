import os
import yaml
import argparse
from IPython import embed
from easydict import EasyDict
from interfaces.super_resolution import TextSR
import torch
import random
import numpy as np

#5555555555

def main(config, args):

    seed=0
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    Mission = TextSR(config, args)
    if args.test:
        Mission.test()
    elif args.demo:
        Mission.demo()
    else:
        Mission.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--arch', default='mntsr', choices=['mntsr','tbsrn', 'tsrn', 'bicubic', 'srcnn', 'vdsr', 'srres', 'esrgan', 'rdn',
                                                           'edsr', 'lapsrn'])
    parser.add_argument('--text_focus', action='store_true')
    parser.add_argument('--exp_name', required=True, help='Type your experiment name')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_data_dir', type=str, default='./dataset/mydata/test')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--resume', type=str, default='./checkpoint/zxytrain/epoch_best.pth', help='')#./checkpoint/zxytrain/epoch303.pth
    parser.add_argument('--rec', default='crnn', choices=['crnn'])### aster	moran
    parser.add_argument('--STN', action='store_true', default=False, help='')
    parser.add_argument('--syn', action='store_true', default=False, help='use synthetic LR')
    parser.add_argument('--mixed', action='store_true', default=False, help='mix synthetic with real LR')
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--demo_dir', type=str, default='./demo')
    args = parser.parse_args()
    config_path = os.path.join('config', 'super_resolution.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)
    main(config, args)
