# Extension of Jepa by Robot Perception and Action Laboratory, USF
#
# Non-Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Optional, Any
import multiprocessing as mp

import pprint
import yaml
import os

import torch

from jepa_src.utils.distributed import init_distributed

import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from vjepa_encoder.vjepa.utils import init_video_model
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
# from torch.nn.parallel import DistributedDataParallel
from jepa_src.utils.distributed import init_distributed, AllReduce
from jepa_src.utils.logging import get_logger

from vjepa_encoder.vjepa.utils import init_video_model

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

import logging
from jepa_src.utils.logging import get_logger
logger = get_logger(force=True)
logger.setLevel(logging.INFO)

class JepaEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder, self.predictor = None, None

    def preprocess_image(self, input_data: Any):
        """
        Preprocess the input image data.

        Args:
            input_data (Any): Input data in various formats.
                - str: Path to the image file.
                - list: List of image data (numpy arrays, PIL Images, or tensors).
                - numpy.ndarray: Image data as a numpy array.
                    - If the array has shape (batch_size, height, width, channels), it will be treated as a batch of images.
                    - If the array has shape (height, width, channels), it will be treated as a single image.
                - PIL.Image.Image: Image data as a PIL Image object.
                - torch.Tensor: Image data as a PyTorch tensor.

        Returns:
            torch.Tensor: Preprocessed image data as a tensor.
                - If the input is a batch of images, the output will have shape (batch_size, channels, height, width).
                - If the input is a single image, the output will have shape (1, channels, height, width).

        Raises:
            ValueError: If the input type is not supported.
        """
        if isinstance(input_data, str):
            img = Image.open(input_data).convert('RGB')
        
        elif isinstance(input_data, list):
            imgs = [
                self.preprocess_image(i).squeeze() for i in input_data
            ]
            preprocessed_input = torch.stack(imgs)
            return preprocessed_input

        elif isinstance(input_data, np.ndarray):
            if len(input_data.shape) == 4:
                input_data = input_data.transpose(0, 3, 1, 2)
                preprocessed_input = torch.from_numpy(input_data).float()
                preprocess = transforms.Compose([
                    transforms.Resize(self.args['data']['crop_size']),
                    transforms.CenterCrop(self.args['data']['crop_size']),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                preprocessed_input = preprocess(preprocessed_input)
                return preprocessed_input
            
            img = Image.fromarray(input_data.astype(np.uint8))

        elif isinstance(input_data, Image.Image):
            img = input_data
        
        elif isinstance(input_data, torch.Tensor):
            preprocessed_input = input_data
            preprocess = transforms.Compose([
                transforms.Resize(self.args['data']['crop_size']),
                transforms.CenterCrop(self.args['data']['crop_size']),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            preprocessed_input = preprocess(preprocessed_input)
            return preprocessed_input
        
        else:
            raise ValueError("Unsupported input type. Expected image path, image array, or PIL Image.")

        # Define the preprocessing transforms
        preprocess = transforms.Compose([
            transforms.Resize(self.args['data']['crop_size']),
            transforms.CenterCrop(self.args['data']['crop_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Apply preprocessing transforms
        preprocessed_input = preprocess(img)

        preprocessed_input = preprocessed_input.unsqueeze(0)  # Add batch dimension
        return preprocessed_input

    def embed_image(self, x):
        """
        Generate embeddings for the input image data.

        Args:
            x (Any): Input image data in various formats.
                - str: Path to the image file.
                - list: List of image data (numpy arrays, PIL Images, or tensors).
                - numpy.ndarray: Image data as a numpy array.
                    - If the array has shape (batch_size, height, width, channels), it will be treated as a batch of images.
                    - If the array has shape (height, width, channels), it will be treated as a single image.
                - PIL.Image.Image: Image data as a PIL Image object.
                - torch.Tensor: Image data as a PyTorch tensor.

        Returns:
            torch.Tensor: Embeddings for the input image data.
                - If the input is a batch of images, the output will have shape (batch_size, num_patches, embedding_size).
                - If the input is a single image, the output will have shape (1, num_patches, embedding_size).

        Notes:
            - The input image data is preprocessed using the `preprocess_image` method before generating embeddings.
            - If the preprocessed input has fewer than 5 dimensions, an additional dimension is added to represent the time dimension.
            - The embeddings are generated using the forward pass of the model.
            - The computation is performed on the available device (GPU if available, otherwise CPU).
        """
        x = self.preprocess_image(x)
        
        # Unsqueeze along the time Dimension
        if len(x.shape) < 5:
            x = x.unsqueeze(2)
        
        if not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:0')
        
        x = x.to(device)
        
        with torch.no_grad():
            embeddings = self.forward(x)
        
        return embeddings
    
    def load_encoder_checkpoint(
        self,
        r_path,
        encoder,
    ):
        try:
            checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        except Exception as e:
            logger.info(f'Encountered exception when loading checkpoint {e}')

        try:

            # -- loading encoder
            pretrained_dict = checkpoint['encoder']
            msg = encoder.load_state_dict(pretrained_dict)
            logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        except Exception as e:
            logger.info(f'Encountered exception when loading checkpoint {e}')
            epoch = 0

        return encoder


    def forward(self, clips: torch.Tensor, masks_enc: List[torch.Tensor], masks_pred: List[torch.Tensor]) -> List[torch.Tensor]:
        z = self.encoder(clips, masks_enc)
        h = self._forward_target(clips, masks_pred)
        z = self.predictor(z, h, masks_enc, masks_pred)
        return z

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.encoder(x)

    @classmethod
    def load_model(cls, config_file_path: str, device: Optional[List[str]] = None) -> "JepaEncoder":
        # TODO: Fix this so it works properly
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])
        
        args = None
        with open(config_file_path, 'r') as y_file:
            args = yaml.load(y_file, Loader=yaml.FullLoader)
            logger.info('loaded params...')
        
        pprint.PrettyPrinter(indent=4).pprint(args)
        dump = os.path.join(args['logging']['folder'], 'params-encoder.yaml')
        with open(dump, 'w') as f:
            yaml.dump(args, f)


        model = cls(args)

        world_size, rank = init_distributed()

        # -- META
        cfgs_meta = args.get('meta')
        load_model = cfgs_meta.get('load_checkpoint')
        assert load_model, "Cannot load model without checkpoint file specified"
        r_file = cfgs_meta.get('read_checkpoint', None)
        seed = cfgs_meta.get('seed', _GLOBAL_SEED)
        save_every_freq = cfgs_meta.get('save_every_freq', -1)
        skip_batches = cfgs_meta.get('skip_batches', -1)
        use_sdpa = cfgs_meta.get('use_sdpa', False)
        which_dtype = cfgs_meta.get('dtype')
        logger.info(f'{which_dtype}')
        if which_dtype.lower() == 'bfloat16':
            dtype = torch.bfloat16
            mixed_precision = True
        elif which_dtype.lower() == 'float16':
            dtype = torch.float16
            mixed_precision = True
        else:
            dtype = torch.float32
            mixed_precision = False

        # -- MASK
        cfgs_mask = args.get('mask')

        # -- MODEL
        cfgs_model = args.get('model')
        model_name = cfgs_model.get('model_name')
        pred_depth = cfgs_model.get('pred_depth')
        pred_embed_dim = cfgs_model.get('pred_embed_dim')
        uniform_power = cfgs_model.get('uniform_power', True)
        use_mask_tokens = cfgs_model.get('use_mask_tokens', True)
        zero_init_mask_tokens = cfgs_model.get('zero_init_mask_tokens', True)
        
        # -- DATA
        cfgs_data = args.get('data')
        num_clips = cfgs_data.get('num_clips')
        num_frames = cfgs_data.get('num_frames')
        tubelet_size = cfgs_data.get('tubelet_size')
        sampling_rate = cfgs_data.get('sampling_rate')
        duration = cfgs_data.get('clip_duration', None)
        crop_size = cfgs_data.get('crop_size', 224)
        patch_size = cfgs_data.get('patch_size')

        # -- LOGGING
        cfgs_logging = args.get('logging')
        folder = cfgs_logging.get('folder')
        tag = cfgs_logging.get('write_tag')

        # -- set device
        if not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)

        # -- log/checkpointing paths
        latest_file = f'{tag}-latest.pth.tar'
        latest_path = os.path.join(folder, latest_file)
        load_path = None
        if load_model:
            load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
            if not os.path.exists(load_path):
                load_path = r_file
                if not os.path.exists(load_path):
                    raise RuntimeError("Cannot load model. Ensure you specify the path to the model .tar file in the input config.")
        
        # -- Attempt to initialize model
        model.encoder, model.predictor = init_video_model(
            uniform_power=uniform_power,
            use_mask_tokens=use_mask_tokens,
            num_mask_tokens=len(cfgs_mask),
            zero_init_mask_tokens=zero_init_mask_tokens,
            device=device,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            model_name=model_name,
            crop_size=crop_size,
            pred_depth=pred_depth,
            pred_embed_dim=pred_embed_dim,
            use_sdpa=use_sdpa,
        )

        # model.encoder = DistributedDataParallel(model.encoder, static_graph=True)

        # -- load training checkpoint
        model.encoder = model.load_encoder_checkpoint(
                load_path, model.encoder
        )
        
        return model


