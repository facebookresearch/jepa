# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from setuptools import setup

VERSION = "0.0.2"

from setuptools import setup, find_packages

def get_requirements():
    with open("./requirements.txt") as reqsf:
        reqs = reqsf.readlines()
    return reqs


if __name__ == "__main__":
    setup(
        name="vjepa_encoder",
        version=VERSION,
        description="JEPA research code.",
        author="Jonathan Koch",
        author_email="johnnykoch02@gmail.com",
        python_requires=">=3.9",
        packages=find_packages(),
        install_requires=get_requirements(),
    )