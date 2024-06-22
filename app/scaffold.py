# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import importlib
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(app, args, resume_preempt=False):

    logger.info(f'Running pre-training of app: {app}')
    return importlib.import_module(f'app.{app}.train').main(
        args=args,
        resume_preempt=resume_preempt)
