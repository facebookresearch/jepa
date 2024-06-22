# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging
import os
import pprint
import sys
import time
import yaml

import submitit

from evals.scaffold import main as eval_main

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

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

    def __init__(self, args_eval=None, resume_preempt=None):
        self.eval_name = args_eval['eval_name']
        self.args_eval = args_eval
        self.resume_preempt = resume_preempt

    def __call__(self):
        eval_name = self.eval_name
        args_eval = self.args_eval
        resume_preempt = self.resume_preempt

        logger.info('loaded eval params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(args_eval)

        eval_main(
            eval_name,
            args_eval=args_eval,
            resume_preempt=resume_preempt)

    def checkpoint(self):
        fb_trainer = Trainer(self.args_eval, True)
        return submitit.helpers.DelayedSubmission(fb_trainer,)


def launch_evals_with_parsed_args(
    args_for_evals,
    submitit_folder,
    partition='learnlab,learnfair',
    timeout=4300,
    nodes=1,
    tasks_per_node=1,
    delay_seconds=10,
    exclude_nodes=None
):
    if not isinstance(args_for_evals, list):
        logger.info(f'Passed in eval-args of type {type(args_for_evals)}')
        args_for_evals = [args_for_evals]

    time.sleep(delay_seconds)
    logger.info('Launching evaluations in separate jobs...')
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

    if exclude_nodes is not None:
        executor.update_parameters(slurm_exclude=exclude_nodes)

    jobs, trainers = [], []
    with executor.batch():
        for ae in args_for_evals:
            fb_trainer = Trainer(ae)
            job = executor.submit(fb_trainer,)
            trainers.append(fb_trainer)
            jobs.append(job)

    for job in jobs:
        logger.info(f'Launched eval job with id {job.job_id}')


def launch_evals():

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
    launch_evals_with_parsed_args(
        args_for_evals=configs,
        submitit_folder=args.folder,
        partition=args.partition,
        timeout=args.time,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        exclude_nodes=args.exclude)
    # ---------------------------------------------------------------------- #


if __name__ == '__main__':
    args = parser.parse_args()
    launch_evals()
