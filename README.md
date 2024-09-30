# nano-JEPA

nano-JEPA: a Video Joint Embedding Predictive Architecture that runs in a regular computer. Based on [V-JEPA](https://github.com/facebookresearch/jepa).

## Setup

```bash
(base) conda create -n nano-jepa python=3.9 
(base) conda activate nano-jepa

# Install PyTorch on hardware that contains GPUs
# (nano-jepa) conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install PyTorch only using CPUs 
(nano-jepa) conda install pytorch torchvision torchaudio cpuonly -c pytorch

(nano-jepa) python setup.py install
```

## System directories

Consider using the [nano-datasets](https://github.com/BHI-Research/nano-datasets) tool to create your local configuration. Here is how your local filesystem
should be organized:

```bash
(base) $ tree -d
.
├── image_datasets
│         ├── imagenet_full_size
│         ├── imagenet-mini
└── video_datasets
    └── k400
        ├── train
        └── val
```

At least, two paths must be provided.

### Dataset location path, k400 dataset example

* A Windows user path: C:\Users\your-user\Documents\ML-datasets\video_datasets\k400\k400_file_index.csv
* A Linux user path: /home/your-user/Documents/ML-datasets/video_datasets/k400/k400_file_index.csv

### Logging location path

* A Windows user path: C:\Users\your-user\Documents\ML-logging
* A Linux user path: /home/your-user/Documents/ML-logging

## Run

A set of config files are provided in this repo. Change the paths ins the *.yaml file using the above guidelines.

```bash
# unsupervised training
(nano-jepa)$ python -m app.train_nano_jepa --fname configs/pretrain/vitt.yaml

# video evaluation
(nano-jepa)$ python -m evals.eval_video_nano_jepa  --fname configs/evals/vitt16_k400_16x8x3.yaml

# image evaluation
(nano-jepa)$ python -m evals.eval_image_nano_jepa  --fname configs/evals/vitt16_in1k.yaml

# video inference
(nano-jepa)$ python -m evals.infer_video_classification --fname configs/infer/infer_vitt_k400x8x3.yaml

# visualize feature (work in progress)
(nano-jepa)$ python -m evals.eval_features 
```

## Checkpopints

Here is a list of checkpoints available for experimentation:

## Authors

Paper: 
* Title: "nano-JEPA: Democratizing Video Understanding with Personal Computers"
* Authors: Adrián Rostagno, Javier Iparraguirre, Joel Ermantraut, Lucas Tobio, Segundo Foissac, Santiago Aggio, Guillermo Friedrich
* Event: XXV WASI – Workshop Agentes y Sistemas Inteligentes, CACIC.
* Year: 2024.

Bibtex:
```
@inproceedings{ermantraut2020resolucion,
     title={nano-JEPA: Democratizing Video Understanding with Personal Computer},
     author={Adrian Rostagno and Javier Iparraguirre and Joel Ermantraut and Lucas Tobio and Segundo Foissac and Santiago Aggio and Guillermo Friedrich},
     booktitle={XXV WASI – Workshop Agentes y Sistemas Inteligentes, CACIC},
     year={2024}
}
```
