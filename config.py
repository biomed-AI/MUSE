#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@author:  Jiahua Rao
@license: BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
@contact: jiahua.rao@gmail.com
@time:    05/2023
'''


import os, sys
import random
import datetime

import torch
import numpy as np
import argparse

from omegaconf import OmegaConf


def set_rand_seed(seed=1, backends=True):
    print("Random Seed: ", seed)
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = backends
    torch.backends.cudnn.benchmark = backends
    torch.backends.cudnn.deterministic = not backends   


def parse_args():
    parser = argparse.ArgumentParser(description='Expectation-maximization algorithm For Multi-Scale Learning.')

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")

    parser.add_argument("--remark", type=str, default=None, help="experimental remark.")
    parser.add_argument("--job-id", type=str, default='now', help="job-id")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')

    parser.add_argument('--debug', action='store_true', default=False,
                        help='turn on debugging mode which uses a small number of data')

    parser.add_argument('--seed', type=int, default=2, help='')
    parser.add_argument('--num_workers', type=int, default=8, help='')
    parser.add_argument('--prefetch', type=int, default=4, help='')

    # ========================= Data Configs ==========================
    parser.add_argument('--inter-data-dir', type=str, default='/data/user/raojh/worksapce/MUSE_dev/data', help='path to training inter dataset')
    parser.add_argument('--intra-data-dir', type=str, default='/data/user/raojh/worksapce/MUSE_dev/data', help='path to training intra dataset')
    parser.add_argument('--results-dir', type=str, default='./outputs/', help='path to outputs dictionary')

    args = parser.parse_args()

    seed = args.seed
    set_rand_seed(seed=seed)

    job_id = args.job_id if args.job_id != 'now' else datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    args.res_dir = os.path.join(args.results_dir, f"{args.cfg_path.split('/')[-1].split('.')[0]}", job_id)
    args.logging_file = os.path.join(args.res_dir, f"logging.log")
    args.em_log_file = os.path.join(args.res_dir, "em_logging.log")

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    # save command line input & runing files
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)
    print('Command line input: ' + cmd_input + ' is saved.')
    os.system(f'cp *.py {args.res_dir}/')
    os.system(f'cp -r models/ {args.res_dir}/')
    os.system(f'cp {args.cfg_path} {args.res_dir}/')
    print(f'running files: *.py and *.yaml is saved to {args.res_dir}.')

    return args


class Config:
    def __init__(self, args):
        self.config = {}
        self.args = args

        config = OmegaConf.load(self.args.cfg_path)

        self.runner_cfg = config.get("run", None)
        if self.runner_cfg is None:
            raise KeyError(
                "Expecting 'run' as the root key for dataset configuration."
            )

        self.model_cfg = config.get("model", None)

        self.inter_dataset_cfg = config.get("inter_dataset", None)
        self.intra_dataset_cfg = config.get("intra_dataset", None)


    @property
    def run_config(self):
        return self.runner_cfg

    @property
    def intra_dataset_config(self):
        return self.intra_dataset_cfg

    @property
    def inter_dataset_config(self):
        return self.inter_dataset_cfg

    @property
    def model_config(self):
        return self.model_cfg


if __name__ == '__main__':
    # args = parse_args()
    # config = Config(args)

    # print(config.model_cfg)
    # print(config.inter_dataset_config)
    # print(config.intra_datasets_config)
    pass
