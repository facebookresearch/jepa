import argparse
import logging
import numpy as np
import datetime
import logging
import os
import pprint

import yaml
import torch
import torch.nn.functional as F

from evals.eval_video_nano_jepa import init_model, init_opt, load_checkpoint, make_dataloader
from evals.video_classification_frozen.utils import FrameAggregation, ClipAggregation
from src.models.attentive_pooler import AttentiveClassifier
from src.utils.logging import get_logger, AverageMeter

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')

# logger frecuency
log_freq = 10
checkpoint_freq = 1
logger = get_logger(force=True)
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


def execute_inference_work(fname):
    logger.info(f'called-params {fname}')

    # Load config
    args_eval = None
    with open(fname, 'r') as y_file:
        args_eval = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')

    # get a time stamp
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Log config
    pprint.PrettyPrinter(indent=4).pprint(args_eval)
    dump = os.path.join(args_eval['logging']['folder'], timestamp_str + '-params-eval.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args_eval, f)

    # -- PRETRAIN
    args_pretrain = args_eval.get('pretrain')
    checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    classifier_path = args_pretrain.get('classifier', None)
    tag = args_pretrain.get('write_tag', None)
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    tight_SiLU = args_pretrain.get('tight_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
    pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    # Optional [for Video model]:
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)

    # -- DATA
    args_data = args_eval.get('data')
    val_data_path = [args_data.get('dataset_val')]
    dataset_type = args_data.get('dataset_type', 'VideoDataset')
    num_classes = args_data.get('num_classes')
    eval_num_segments = args_data.get('num_segments', 1)
    eval_frames_per_clip = args_data.get('frames_per_clip', 16)
    eval_frame_step = args_pretrain.get('frame_step', 4)
    eval_duration = args_pretrain.get('clip_duration', None)
    eval_num_views_per_segment = args_data.get('num_views_per_segment', 1)

    # -- OPTIMIZATION
    args_opt = args_eval.get('optimization')
    resolution = args_opt.get('resolution', 224)
    attend_across_segments = args_opt.get('attend_across_segments', False)
    num_epochs = args_opt.get('num_epochs')
    wd = args_opt.get('weight_decay')
    start_lr = args_opt.get('start_lr')
    lr = args_opt.get('lr')
    final_lr = args_opt.get('final_lr')
    warmup = args_opt.get('warmup')
    use_bfloat16 = args_opt.get('use_bfloat16')
    resume_preempt = True
    # -- EXPERIMENT-ID/TAG (optional)
    resume_checkpoint = args_eval.get('resume_checkpoint', False) or resume_preempt
    eval_tag = args_eval.get('tag', None)

    # ----------------------------------------------------------------------- #

    # device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    logger.info(f'Using device {device}')

    # one cluster node
    world_size = 1
    rank = 0

    # -- pretrained encoder (frozen)
    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=pretrain_frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa)
    if pretrain_frames_per_clip == 1:
        # Process each frame independently and aggregate
        encoder = FrameAggregation(encoder).to(device)
    else:
        # Process each video clip independenty and aggregate
        encoder = ClipAggregation(
            encoder,
            tubelet_size=tubelet_size,
            attend_across_segments=attend_across_segments
        ).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # -- init classifier
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
        depth=1,
        num_classes=num_classes,
    ).to(device)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    val_loader = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        resolution=resolution,
        frames_per_clip=eval_frames_per_clip,
        frame_step=eval_frame_step,
        num_segments=eval_num_segments,
        eval_duration=eval_duration,
        num_views_per_segment=eval_num_views_per_segment,
        allow_segment_overlap=True,
        # batch_size=batch_size,
        batch_size=1,
        world_size=world_size,
        rank=rank,
        training=False)
    ipe = len(val_loader)
    logger.info(f'Dataloader created... iterations per epoch: {ipe}')

    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifier=classifier,
        wd=wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16)

    # -- load training checkpoint
    latest_path = pretrain_folder + classifier_path

    if resume_checkpoint:
        classifier, _, _, _ = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifier=classifier,
            opt=optimizer,
            scaler=scaler)
        logger.info(f'Loading classifier {classifier}')

    # infer
    try:
        res = run_one_epoch(
            device=device,
            attend_across_segments=attend_across_segments,
            encoder=encoder,
            classifier=classifier,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16)
    except Exception as e:
        logger.info('Exception during inference' + str(e))


def run_one_epoch(
        device,
        encoder,
        classifier,
        data_loader,
        use_bfloat16,
        attend_across_segments,
):
    result = []
    for itr, data in enumerate(data_loader):

        logger.info('Doing inference on video: ' + str(itr))
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):

            # Load data and put on GPU
            clips = [
                [dij.to(device, non_blocking=True) for dij in di]  # iterate over spatial views of clip
                for di in data[0]  # iterate over temporal index of clip
            ]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device)
            # Forward and prediction
            with torch.no_grad():
                outputs = encoder(clips, clip_indices)
                # if not training:
                if 1:
                    if attend_across_segments:
                        outputs = [classifier(o) for o in outputs]
                    else:
                        outputs = [[classifier(ost) for ost in os] for os in outputs]

        with torch.no_grad():
            if attend_across_segments:
                outputs = sum([F.softmax(o, dim=1) for o in outputs]) / len(outputs)
            else:
                outputs = sum([sum([F.softmax(ost, dim=1) for ost in os]) for os in outputs]) / len(outputs) / len(
                    outputs[0])

            aa = outputs.max(dim=1).indices
            result.append(aa)

    logger.info(f'Result: {result}')

    return result


if __name__ == '__main__':
    args = parser.parse_args()
    execute_inference_work(args.fname)
