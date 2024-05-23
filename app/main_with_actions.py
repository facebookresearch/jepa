# In app/main_with_actions.py

# Copyright (c) NeoCybernetica, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import pprint
import yaml
import os
import logging

from app.scaffold import main as app_main
from src.utils.distributed import init_distributed
from app.vjepa.train_with_actions import main as train  # Import the main function from train_with_actions.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fname",
        type=str,
        help="name of config file to load",
        default="configs/pretrain/vith16_384.yaml",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cuda:0"],
        help="which devices to use on local machine",
    )

    args = parser.parse_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Called parameters: {args.fname}")

    # Load configuration from YAML file
    with open(args.fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info("Loaded configuration parameters.")

    # Pretty print the configuration parameters
    pprint.PrettyPrinter(indent=4).pprint(params)

    # Save the configuration parameters to a YAML file
    dump_file = os.path.join(params["logging"]["folder"], "params-pretrain.yaml")
    os.makedirs(os.path.dirname(dump_file), exist_ok=True)
    with open(dump_file, "w") as f:
        yaml.dump(params, f)

    # Initialize distributed training (for single GPU, world_size and rank will be 1 and 0 respectively)
    num_gpus = len(args.devices)
    rank = 0  # Since you're on a single GPU
    world_size, rank = init_distributed(rank_and_world_size=(rank, num_gpus))  # Update for single GPU
    logger.info(f"Running... (rank: {rank}/{world_size})")
    
    # Setup environment variables for GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices[rank].split(":")[-1])

    # Launch the app with loaded config
    try:
        train(args=params, world_size=world_size, rank=rank)
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise e

if __name__ == "__main__":
    main()
