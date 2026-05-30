# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.transforms import OneOf
from monai.data import load_decathlon_datalist


def _resolve_label_entry(item, args):
    label_key = getattr(args, "label_key", "label")
    fallback_label_key = getattr(args, "fallback_label_key", "label_plus")
    prefer_label_plus = getattr(args, "prefer_label_plus", False)

    label_value = item.get(label_key)
    fallback_value = item.get(fallback_label_key)

    if fallback_value:
        item["label"] = fallback_value
        return item

    if label_value:
        resolved_label = label_value
        if prefer_label_plus:
            resolved_label = resolved_label.replace("/label/", "/label_plus/")
            resolved_label = resolved_label.replace("\\label\\", "\\label_plus\\")
        item["label"] = resolved_label
        return item

    return item


def _normalize_datalist(items, args):
    return [_resolve_label_entry(dict(item), args) for item in items]


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def _get_loader(args, use_adv=False):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    train_split = getattr(args, "train_split", "training")
    val_split = getattr(args, "val_split", "validation")
    train_transform_items = [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.AddChanneld(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        OneOf(transforms=[
            transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 2.0), mode=("bilinear", "nearest")),
            transforms.Spacingd(keys=["image", "label"], pixdim=(1.2, 1.2, 2.5), mode=("bilinear", "nearest")),
            transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 3.0), mode=("bilinear", "nearest")),
        ]),
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
        ),
        transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
        transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
        transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
        transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
        transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
    ]

    if use_adv:
        train_transform_items.extend(
            [
                transforms.RandRotated(
                    keys=["image", "label"],
                    range_x=0.52,
                    range_y=0.52,
                    range_z=0.1,
                    prob=0.25,
                    mode=("bilinear", "nearest"),
                    padding_mode="zeros",
                ),
                transforms.RandZoomd(
                    keys=["image", "label"],
                    min_zoom=0.8,
                    max_zoom=1.2,
                    prob=0.25,
                    mode=("trilinear", "nearest"),
                ),
                transforms.RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
                transforms.RandGaussianSmoothd(
                    keys=["image"],
                    sigma_x=(0.5, 1.5),
                    sigma_y=(0.5, 1.5),
                    sigma_z=(0.5, 1.5),
                    prob=0.15,
                ),
            ]
        )

    train_transform_items.extend(
        [
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
        ]
    )

    if use_adv:
        train_transform_items.append(
            transforms.RandAdjustContrastd(keys=["image"], prob=args.RandShiftIntensityd_prob, gamma=(0.7, 1.5))
        )

    train_transform_items.append(transforms.ToTensord(keys=["image", "label"]))
    train_transform = transforms.Compose(train_transform_items)

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(1.0, 1.0, 3.0), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image","label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    datalist = load_decathlon_datalist(datalist_json, True, train_split, base_dir=data_dir)
    datalist = _normalize_datalist(datalist, args)
    dataset_backend = "normal" if args.use_normal_dataset else getattr(args, "dataset_backend", "persistent")
    if dataset_backend == "normal":
        train_ds = data.Dataset(data=datalist, transform=train_transform)
    elif dataset_backend == "cache":
        train_ds = data.CacheDataset(
            data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
        )
    elif dataset_backend == "persistent":
        train_ds = data.PersistentDataset(
            data=datalist, transform=train_transform, cache_dir=args.train_cache_dir
        )
    else:
        raise ValueError(f"Unsupported dataset backend: {dataset_backend}")

    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )
    val_files = load_decathlon_datalist(datalist_json, True, val_split, base_dir=data_dir)
    val_files = _normalize_datalist(val_files, args)
    if dataset_backend == "persistent":
        val_ds = data.PersistentDataset(data=val_files, transform=val_transform, cache_dir=args.val_cache_dir)
    elif dataset_backend == "cache":
        val_ds = data.CacheDataset(
            data=val_files,
            transform=val_transform,
            cache_num=len(val_files),
            cache_rate=1.0,
            num_workers=args.workers,
        )
    else:
        val_ds = data.Dataset(data=val_files, transform=val_transform)
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
    )
    loader = [train_loader, val_loader]

    return loader


def get_loader_v2(args):
    return _get_loader(args, use_adv=False)


def get_loader_v2_adv(args):
    return _get_loader(args, use_adv=True)
