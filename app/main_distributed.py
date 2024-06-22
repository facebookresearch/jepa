# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import pprint
import yaml

import submitit

from app.scaffold import main as app_main
from src.utils.logging import get_logger

logger = get_logger(force=True)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--folder', type=str,
    help='location to save submitit logs',
    default='/fsx-jepa/massran/submitit/')
parser.add_argument(
    '--exclude', type=str,
    help='nodes to exclude from training',
    default=None)
parser.add_argument(
    '--batch-launch', action='store_true',
    help='whether fname points to a file to batch-lauch several config files')
parser.add_argument(
    '--fname', type=str,
    help='yaml file containing config file names to launch',
    default='configs.yaml')
parser.add_argument(
    '--partition', type=str,
    help='cluster partition to submit jobs on')
parser.add_argument(
    '--time', type=int, default=4300,
    help='time in minutes to run job')


class Trainer:

    def __init__(self, args_pretrain, load_model=None):
        self.app = args_pretrain['app']
        self.args_pretrain = args_pretrain
        self.load_model = load_model

    def __call__(self):
        app = self.app
        params = self.args_pretrain
        load_model = self.load_model

        logger.info('loaded pretrain params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

        # Launch app with loaded config
        resume_preempt = False if load_model is None else load_model
        app_main(app, args=params, resume_preempt=resume_preempt)

    def checkpoint(self):
        fb_trainer = Trainer(self.args_pretrain, True)
        return submitit.helpers.DelayedSubmission(fb_trainer,)


def launch_app_with_parsed_args(
    args_for_pretrain,
    submitit_folder,
    partition,
    timeout=4300,
    nodes=1,
    tasks_per_node=1,
    exclude_nodes=None
):
    executor = submitit.AutoExecutor(
        folder=os.path.join(submitit_folder, 'job_%j'),
        slurm_max_num_timeout=20)
    executor.update_parameters(
        slurm_partition=partition,
        slurm_mem_per_gpu='55G',
        timeout_min=timeout,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=12,
        gpus_per_node=tasks_per_node)

    if args.exclude is not None:
        executor.update_parameters(slurm_exclude=args.exclude)

    jobs, trainers = [], []
    with executor.batch():
        for ap in args_for_pretrain:
            fb_trainer = Trainer(ap)
            job = executor.submit(fb_trainer,)
            trainers.append(fb_trainer)
            jobs.append(job)

    for job in jobs:
        print(job.job_id)


def launch():

    # ---------------------------------------------------------------------- #
    # 1. Put config file names in a list
    # ---------------------------------------------------------------------- #
    config_fnames = [args.fname]

    # -- If batch-launch is True, then the args.fname yaml file is not a
    # -- config, but actually specifies a list of other config files
    # -- to run in a slurm job array
    if args.batch_launch:
        with open(args.fname, 'r') as y_file:
            config_fnames = yaml.load(y_file, Loader=yaml.FullLoader)
    # ---------------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    # 2. Parse each yaml config file as a dict and place in list
    # ---------------------------------------------------------------------- #
    nodes, tasks_per_node = None, None
    configs = []
    for f in config_fnames:
        with open(f, 'r') as y_file:
            _params = yaml.load(y_file, Loader=yaml.FullLoader)
            nodes = int(_params.get('nodes'))
            tasks_per_node = int(_params.get('tasks_per_node'))
            configs += [_params]
    logger.info(f'Loaded {len(configs)} config files')
    logger.info(f'Running all jobs with {nodes=} / {tasks_per_node=}')
    # ---------------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    # 3. Launch evals with parsed config files
    # ---------------------------------------------------------------------- #
    launch_app_with_parsed_args(
        args_for_pretrain=configs,
        submitit_folder=args.folder,
        partition=args.partition,
        timeout=args.time,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        exclude_nodes=args.exclude)
    # ---------------------------------------------------------------------- #


if __name__ == '__main__':
    args = parser.parse_args()
    launch()
