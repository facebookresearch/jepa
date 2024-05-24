# Copyright (c) NeoCybernetica, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os


# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import time
import numpy as np
import traceback

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torchvision.transforms import ToPILImage

from einops import rearrange

from src.datasets.data_manager import init_data
from src.masks.random_tube import MaskCollatorWithActions as TubeMaskCollatorWithActions
from src.masks.multiblock3d import MaskCollatorWithActions as MB3DMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    get_logger,
    grad_logger,
    adamw_logger,
    AverageMeter,
)
from src.utils.tensors import repeat_interleave_batch, to_batch
from src.models.utils.combine_encodings import (
    combine_encodings_concat,
    combine_encodings_add,
    AttentionFusion,
)


from app.vjepa.utils import (
    load_checkpoint,
    init_video_model,
    init_opt,
)
from app.vjepa.transforms import make_image_transforms


# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__)

def main(args, world_size=1, rank=0, resume_preempt=False):
    # First let's go over the folders and generate the 
    
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    cfgs_meta = args.get("meta")
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    r_file = cfgs_meta.get("read_checkpoint", None)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MASK
    cfgs_mask = args.get("mask")

    # -- MODEL
    cfgs_model = args.get("model")
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    uniform_power = cfgs_model.get("uniform_power", True)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", True)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "egovehicle_imagedataset")
    mask_type = cfgs_data.get("mask_type", "multiblock3d")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights", None)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(
            dataset_paths
        ), "Must have one sampling weight specified for each dataset"
    batch_size = cfgs_data.get("batch_size")
    num_clips = cfgs_data.get("num_clips")
    num_frames = cfgs_data.get("num_frames")
    tubelet_size = cfgs_data.get("tubelet_size")
    sampling_rate = cfgs_data.get("sampling_rate")
    duration = cfgs_data.get("clip_duration", None)
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    filter_short_videos = cfgs_data.get("filter_short_videos", False)
    decode_one_clip = cfgs_data.get("decode_one_clip", True)
    log_resource_util_data = cfgs_data.get("log_resource_utilization", False)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")
    reg_coeff = cfgs_loss.get("reg_coeff")

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    clip_grad = cfgs_opt.get("clip_grad", None)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    ema = cfgs_opt.get("ema")
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)

    # -- LOGGING
    cfgs_logging = args.get("logging")
    folder = cfgs_logging.get("folder")
    tag = cfgs_logging.get("write_tag")

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #   

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
        
    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")
    latest_file = f"{tag}-latest.pth.tar"
    latest_path = os.path.join(folder, latest_file)
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "loss-jepa"),
        ("%.5f", "reg-loss"),
        ("%.5f", "enc-grad-norm"),
        ("%.5f", "pred-grad-norm"),
        ("%d", "gpu-time(ms)"),
        ("%d", "wall-time(ms)"),
    )

    # -- init model
    encoder, predictor, action_encoder = init_video_model(
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
    target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    if mask_type == "multiblock3d":
        logger.info("Initializing basic multi-block mask")
        mask_collator = MB3DMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask,
        )
    else:
        logger.info("Initializing random tube mask")
        mask_collator = TubeMaskCollatorWithActions(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask,
        )
    transform = make_image_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,        
        crop_size=crop_size,
    )

    # -- init data-loaders/samplers
    (unsupervised_loader, unsupervised_sampler) = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,
        clip_len=num_frames,
        frame_sample_rate=sampling_rate,
        filter_short_videos=filter_short_videos,
        decode_one_clip=decode_one_clip,
        duration=duration,
        num_clips=num_clips,
        transform=transform,
        datasets_weights=datasets_weights,
        collator=mask_collator,
        num_workers=num_workers,
        world_size=world_size,
        pin_mem=pin_mem,
        rank=rank,
        log_dir=folder if log_resource_util_data else None,
    )
    try:
        _dlen = len(unsupervised_loader)
    except Exception:  # Different interface for webdataset
        _dlen = unsupervised_loader.num_batches
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataest length: {ipe}/{_dlen}")

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    start_epoch = 0
    # -- load training checkpoint
    if load_model or os.path.exists(latest_path):
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {traceback.format_exc}")

    logger.info("Initializing loader...")
    loader = iter(unsupervised_loader)    

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        input_var_meter = AverageMeter()
        input_var_min_meter = AverageMeter()
        jepa_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()
        mask_meters = [AverageMeter() for _ in range(len(cfgs_mask))]
        gpu_time_meter = AverageMeter()
        wall_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            try:
                collated_images, collated_maneuvers, masks_enc, masks_pred = next(loader)

            except StopIteration:
                logger.info(
                    "Exhausted data loaders before completing all planned iterations. Ending epoch early..."
                )
                break  # Exit the current epoch loop if there are no more data points to process
            
            assert len(masks_enc) == len(
                masks_pred
            ), "Currently require num encoder masks = num predictor masks"

            def load_images_and_actions():
                try:
                    images = []
                    to_pil = ToPILImage()  # Create an instance of ToPILImage
                    
                    for i in range(len(collated_images)):
                        image = collated_images[i]
                        image = to_pil(image)  # Convert the PyTorch tensor to a PIL Image
                        image = transform(image)  # Apply the transformation to the PIL image
                        images.append(image)

                    # Stack the transformed images into a single batched tensor
                    images = torch.stack(images, dim=0).to(device, non_blocking=True)

                    # -- Encode actions
                    encoded_actions = action_encoder(collated_maneuvers)

                    # ... (load masks as before)
                    _masks_enc, _masks_pred = [], []
                    for _me, _mp in zip(masks_enc, masks_pred):
                        _me = _me.to(device, non_blocking=True)
                        _mp = _mp.to(device, non_blocking=True)
                        _masks_enc.append(_me)
                        _masks_pred.append(_mp)

                    return images, encoded_actions, _masks_enc, _masks_pred
                except Exception as e:
                    logger.error(f"Error in load_images_and_actions: {str(e)}")
                    raise e

            
            images, encoded_actions, masks_enc, masks_pred = load_images_and_actions()

            for _i, m in enumerate(mask_meters):
                m.update(masks_enc[_i][0].size(-1))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target(images):
                    """
                    Encodes the target images using the target encoder and returns the embeddings.

                    Args:
                        images (torch.Tensor): A tensor of shape [B, T, C, H, W] representing a batch
                            of image sequences.
                        
                    Returns:
                        torch.Tensor: A tensor of shape [B, T, D] representing the encoded image embeddings,
                            where D is the embedding dimension.
                    """
                    with torch.no_grad():
                        image_embeddings = target_encoder(images)
                        # Normalize the embeddings across the feature dimension
                        normalized_embeddings = F.layer_norm(image_embeddings, (image_embeddings.size(-1),))

                        # Extract the embeddings for the next frames as targets
                        next_frame_embeddings = normalized_embeddings[:, 1:, :]  # Assuming frames are sequential
                        return next_frame_embeddings

                def forward_context(images, encoded_actions, h):
                    """
                    Encodes context images with the encoder, combines with encoded actions,
                    and predicts masked regions using the predictor.

                    Args:
                        images (torch.Tensor): A tensor of shape [B, T, C, H, W] representing a batch
                            of image sequences.
                        encoded_actions (torch.Tensor): A tensor of shape [B, T, A] representing encoded actions,
                            where A is the action embedding dimension.
                        h (torch.Tensor): The hidden state from the target encoder. (Ground truth)

                    Returns:
                        torch.Tensor: A list of tensors representing the predicted values for the masked regions.
                    """
                    try:
                        image_embeddings = encoder(images, masks_enc)

                        # Combine image and action embeddings
                        combined_embeddings = combine_encodings_concat(image_embeddings, encoded_actions)

                        # Predict masked regions
                        predictions = predictor(combined_embeddings, h, masks_enc, masks_pred)
                        return predictions
                    except Exception as e:
                        logger.error(f"Error in forward_context: {str(e)}")
                        raise e


                def loss_fn(z_next, h_next):
                    loss = 0.0
                    # Compute loss between predicted next frames and ground truth next frames
                    for zi, hi in zip(z_next, h_next):
                        loss += torch.mean(torch.abs(zi - hi) ** loss_exp) / loss_exp
                    loss /= len(h_next)
                    return loss

                def reg_fn(z):
                    return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(
                        z
                    )

                # Step 1. Forward
                loss_jepa, loss_reg = 0.0, 0.0
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    h_next = forward_target(images)
                    z_next = forward_context(images, h_next, encoded_actions)
                    loss_jepa = loss_fn(z_next, h_next)  # jepa prediction loss
                    pstd_z = reg_fn(z_next)  # predictor variance across patches
                    loss_reg += torch.mean(F.relu(1.0 - pstd_z))
                loss = loss_jepa + reg_coeff * loss_reg

                # Step 2. Backward & step
                _enc_norm, _pred_norm = 0.0, 0.0
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if (epoch > warmup) and (clip_grad is not None):
                    _enc_norm = torch.nn.utils.clip_grad_norm_(
                        encoder.parameters(), clip_grad
                    )
                    _pred_norm = torch.nn.utils.clip_grad_norm_(
                        predictor.parameters(), clip_grad
                    )
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                grad_stats.global_norm = float(_enc_norm)
                grad_stats_pred = grad_logger(predictor.named_parameters())
                grad_stats_pred.global_norm = float(_pred_norm)
                optimizer.zero_grad()
                optim_stats = adamw_logger(optimizer)

                # Step 3. momentum update of target encoder
                m = next(momentum_scheduler)
                with torch.no_grad():
                    for param_q, param_k in zip(
                        encoder.parameters(), target_encoder.parameters()
                    ):
                        param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

                return (
                    float(loss),
                    float(loss_jepa),
                    float(loss_reg),
                    _new_lr,
                    _new_wd,
                    grad_stats,
                    grad_stats_pred,
                    optim_stats,
                )

            (
                loss,
                loss_jepa,
                loss_reg,
                _new_lr,
                _new_wd,
                grad_stats,
                grad_stats_pred,
                optim_stats,
            ), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            input_var = float(
                AllReduce.apply(images.view(images.shape[0], -1).var(dim=1).mean(dim=0))
            )
            input_var_min = float(
                AllReduce.apply(torch.min(images.view(images.shape[0], -1).var(dim=1)))
            )
            input_var_meter.update(input_var)
            input_var_min_meter.update(input_var_min)
            jepa_loss_meter.update(loss_jepa)
            reg_loss_meter.update(loss_reg)
            gpu_time_meter.update(gpu_etime_ms)
            wall_time_meter.update(iter_elapsed_time_ms)

            # -- Logging
            def log_stats():
                csv_logger.log(
                    epoch + 1,
                    itr,
                    loss,
                    loss_jepa,
                    loss_reg,
                    grad_stats.global_norm,
                    grad_stats_pred.global_norm,
                    gpu_etime_ms,
                    iter_elapsed_time_ms,
                )
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f | p%.3f r%.3f | "
                        "input_var: %.3f %.3f | "
                        "masks: %s "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[gpu: %.1f ms]"
                        "[wall: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            jepa_loss_meter.avg,
                            reg_loss_meter.avg,
                            input_var_meter.avg,
                            input_var_min_meter.avg,
                            "["
                            + ", ".join(["%.1f" % m.avg for m in mask_meters])
                            + "]",
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            gpu_time_meter.avg,
                            wall_time_meter.avg,
                        )
                    )

                    if optim_stats is not None:
                        logger.info(
                            "[%d, %5d] first moment: %.2e [%.2e %.2e] second moment: %.2e [%.2e %.2e]"
                            % (
                                epoch + 1,
                                itr,
                                optim_stats.get("exp_avg").avg,
                                optim_stats.get("exp_avg").min,
                                optim_stats.get("exp_avg").max,
                                optim_stats.get("exp_avg_sq").avg,
                                optim_stats.get("exp_avg_sq").min,
                                optim_stats.get("exp_avg_sq").max,
                            )
                        )

                    if grad_stats is not None:
                        logger.info(
                            "[%d, %5d] enc_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e"
                            % (
                                epoch + 1,
                                itr,
                                grad_stats.first_layer,
                                grad_stats.last_layer,
                                grad_stats.min,
                                grad_stats.max,
                                grad_stats.global_norm,
                            )
                        )

                    if grad_stats_pred is not None:
                        logger.info(
                            "[%d, %5d] pred_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e"
                            % (
                                epoch + 1,
                                itr,
                                grad_stats_pred.first_layer,
                                grad_stats_pred.last_layer,
                                grad_stats_pred.min,
                                grad_stats_pred.max,
                                grad_stats_pred.global_norm,
                            )
                        )

            log_stats()
            assert not np.isnan(loss), "loss is nan"

        # -- Save Checkpoint
        logger.info("avg. loss %.3f" % loss_meter.avg)
        # -- Save Last
        if epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"{tag}-e{epoch}.pth.tar"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)
