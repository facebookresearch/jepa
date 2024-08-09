from setuptools import setup

VERSION = "0.0.1"


def get_requirements():
    with open("./requirements.txt") as reqsf:
        reqs = reqsf.readlines()
    return reqs


if __name__ == "__main__":
    setup(
        name="nano-jepa",
        version=VERSION,
        description="nano-JEPA: a Video Joint Embedding Predictive Architecture that runs in a regular computer.",
        python_requires=">=3.9",
        install_requires=get_requirements(),
    )
