# nano-JEPA

nano-JEPA: a Video Joint Embedding Predictive Architecture that runs in a regular computer. Based on [V-JEPA](https://github.com/facebookresearch/jepa).

## Setup

```bash
(base) conda create -n nano-jepa python=3.9 
(base) conda activate nano-jepa

# GPU hardware
# (nano-jepa) conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# CPU only
(nano-jepa) conda install pytorch torchvision torchaudio cpuonly -c pytorch

(nano-jepa) python setup.py install
```

## Run

```bash
# unsupervised training
(nano-jepa)$ python -m app.train_nano_jepa --fname configs/pretrain/vitl16.yaml

# video evaluation
(nano-jepa)$ python -m evals.eval_video_nano_jepa  --fname configs/evals/vitt16_k400_16x8x3.yaml

# image evaluation
(nano-jepa)$ python -m evals.eval_video_nano_jepa  --fname configs/evals/vitt16_in1k.yaml

# video inference
(nano-jepa)$ python -m evals.infer_video_classification --fname configs/infer/infer_vitl16_k400x8x3.yaml

# visualize feature (work in progress)
(nano-jepa)$ python -m evals.eval_features 
```

## System directories

Consder usig the [nano-datasets](https://github.com/BHI-Research/nano-datasets) tool to create your local configuration.

```bash
(base) $ tree -d
.
├── image_datasets
│   ├── imagenet_full_size
│   ├── imagenet-mini
└── video_datasets
    └── k400
        ├── train
        └── val
```

## Authors

We are preparing a publication titlted "nano-JEPA: Democratizing Video Understanding with Personal Computers". Please cite us if you use this repo.

Contributors:

* Adrián Rostagno
* Javier Iparraguirre
* Joel Ermantraut
* Lucas Tobio
* Segundo Foissac
* Santiago Aggio
* Guillermo Friedrich
