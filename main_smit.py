# Copyright 2021 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import shutil
import tempfile
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.data_utils import get_loader_v2, get_loader_v2_adv

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss
from monai.metrics import DiceMetric
# from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction

import torch.nn as nn
from torch.nn.modules.loss import _Loss
from monai.utils import DiceCEReduction, LossReduction, Weight, look_up_option
from typing import Callable, List, Optional, Sequence, Union
import warnings
from monai.networks import one_hot
from dataset_paths import get_default_data_root

from models import smit, configs_smit

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="testrun_tv1", type=str, help="directory to save the tensorboard logs")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument(
    "--data_dir",
    default=get_default_data_root(),
    type=str,
    help="centralized dataset directory",
)
parser.add_argument("--json_list", default="heartsub_master.json", type=str, help="dataset json file")
parser.add_argument(
    "--train_split",
    default="set1_cnc64_training",
    type=str,
    help="training split name inside the dataset json",
)
parser.add_argument(
    "--val_split",
    default="set1_cnc64_validation",
    type=str,
    help="validation split name inside the dataset json",
)
parser.add_argument("--label_key", default="label", type=str, help="primary label field name in the dataset json")
parser.add_argument(
    "--fallback_label_key",
    default="label_plus",
    type=str,
    help="alternate label field name used when present in the dataset json",
)
parser.add_argument(
    "--prefer_label_plus",
    action="store_true",
    help="rewrite old /label/ paths to /label_plus/ when loading labels",
)
parser.add_argument(
    "--pretrained_model_name",
    default="SwinUNETR.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=500, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=2e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=10, type=int, help="validation frequency")
parser.add_argument("--checkpoint_save_every", default=20, type=int, help="checkpoint save frequency in epochs")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=4, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument(
    "--dataset_backend",
    default="persistent",
    choices=["normal", "cache", "persistent"],
    help="dataset backend to use for training and validation",
)
parser.add_argument("--persistent_cache_root", default=None, type=str, help="persistent dataset cache root")
parser.add_argument(
    "--keep_persistent_cache",
    action="store_true",
    help="keep the persistent dataset cache after training finishes",
)
parser.add_argument(
    "--resume_last_checkpoint",
    action="store_true",
    help="resume from the latest checkpoint saved in the run logdir",
)
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--use_ssl_pretrained", action="store_true", help="use self-supervised pretrained weights")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--lambda_dice", default=1.0, type=float, help="value for dice loss in joint function")
parser.add_argument("--freeze_transformer", action="store_true", help="freeze transformer backbone weights")
parser.add_argument(
    "--augmentation_mode",
    default="adv",
    choices=["basic", "adv"],
    help="training augmentation pipeline to use",
)



def resolve_resume_checkpoint(args):
    if args.checkpoint:
        return args.checkpoint
    if not args.resume_last_checkpoint:
        return None

    candidates = [
        os.path.join(args.logdir, "model_final.pt"),
        os.path.join(args.logdir, "model.pt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No checkpoint found in {args.logdir} to resume from.")


def prepare_dataset_cache(args):
    if args.use_normal_dataset:
        args.dataset_backend = "normal"

    if args.dataset_backend != "persistent":
        args.train_cache_dir = None
        args.val_cache_dir = None
        return

    cache_root = args.persistent_cache_root
    if cache_root is None:
        run_name = os.path.basename(args.logdir.rstrip(os.sep)) or "train"
        cache_root = tempfile.mkdtemp(prefix=f"txseg_cache_{run_name}_")
    else:
        os.makedirs(cache_root, exist_ok=True)

    args.persistent_cache_root = cache_root
    args.train_cache_dir = os.path.join(cache_root, "train")
    args.val_cache_dir = os.path.join(cache_root, "val")
    os.makedirs(args.train_cache_dir, exist_ok=True)
    os.makedirs(args.val_cache_dir, exist_ok=True)


def cleanup_dataset_cache(args):
    cache_root = getattr(args, "persistent_cache_root", None)
    if args.dataset_backend == "persistent" and cache_root and not args.keep_persistent_cache:
        shutil.rmtree(cache_root, ignore_errors=True)
        print(f"Removed persistent dataset cache: {cache_root}")

def freeze_model_parameters(model, freeze_pattern="transformer"):
    """
    Freeze parameters matching the specified pattern.
    
    Args:
        model: The model whose parameters to freeze
        freeze_pattern: String pattern to match parameter names for freezing
    """
    frozen_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if freeze_pattern in name:
            param.requires_grad = False
            frozen_params += 1
            print(f"Frozen: {name}")
    
    print(f"\nFrozen {frozen_params}/{total_params} parameters matching pattern '{freeze_pattern}'")
    return frozen_params

def main():
    args = parser.parse_args()
    print(args)
    args.amp = not args.noamp
    args.logdir = "./runs/" + args.logdir
    prepare_dataset_cache(args)
    args.checkpoint = resolve_resume_checkpoint(args)

    try:
        if args.distributed:
            args.ngpus_per_node = torch.cuda.device_count()
            print("Found total gpus", args.ngpus_per_node)
            args.world_size = args.ngpus_per_node * args.world_size
            mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
        else:
            main_worker(gpu=0, args=args)
    finally:
        cleanup_dataset_cache(args)


def main_worker(gpu, args):

    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    loader_fn = get_loader_v2_adv if args.augmentation_mode == "adv" else get_loader_v2
    loader = loader_fn(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    pretrained_dir = args.pretrained_dir
    
    config = configs_smit.get_SMIT_128_bias_True()
    model = smit.SMIT_3D_Seg(config,out_channels=args.out_channels,norm_name='instance')
    print(model.transformer.patch_embed.norm.weight)

    if args.resume_ckpt:
        model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))["state_dict"]
        model.load_state_dict(model_dict)
        print("Use pretrained weights")

    if args.use_ssl_pretrained:
        try:
            model_dict = torch.load("./pretrained_models/model_smit.pth", map_location='cpu')
            pretrained_dict = model_dict['student']

            for key in list(pretrained_dict.keys()):
                    pretrained_dict[key.replace('module.backbone.', '')] = pretrained_dict.pop(key)
            model.load_state_dict(pretrained_dict,strict=False)
            
            print("Using pretrained self-supervised SMIT backbone weights !")
        except ValueError:
            raise ValueError("Self-supervised pre-trained weights not available for" + str(args.model_name))
    
    print(model.transformer.patch_embed.norm.weight)

    if args.freeze_transformer:
        print("\n=== Freezing Transformer Parameters ===")
        freeze_model_parameters(model, freeze_pattern="transformer")
        
        # Verify frozen parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    dice_loss = DiceCELoss(
        include_background=True, to_onehot_y=True, softmax=True,
        squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr,
        lambda_dice=args.lambda_dice
        )
                
    post_label = AsDiscrete(to_onehot=True, n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=args.out_channels)
    dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0
    checkpoint = None

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (next epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          output_device=args.gpu,
                                                          find_unused_parameters=True)
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None

    if checkpoint is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        print("=> restored optimizer and scheduler state")

    #print(model.transformer.patch_embed.norm.weight)
    
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    main()
