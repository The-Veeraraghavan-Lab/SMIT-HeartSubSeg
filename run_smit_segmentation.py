import argparse
import os

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from monai import data, transforms
from monai.data import decollate_batch, load_decathlon_datalist
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from dataset_paths import get_default_data_root, resolve_dataset_json
from models import configs_smit, smit


def infer_results_bucket(dataset_split):
    dataset_split_lower = dataset_split.lower()
    if "breast" in dataset_split_lower or "b66" in dataset_split_lower:
        return "model_outputs_breast"
    return "model_outputs_lung"


def resolve_results_dir(results_dir):
    if os.path.isabs(results_dir) or results_dir.startswith("analysis/results"):
        return results_dir
    if results_dir == "results":
        return os.path.join("analysis", "results")
    if results_dir.startswith("results/"):
        suffix = results_dir[len("results/") :]
        return os.path.join("analysis", "results", suffix)
    return os.path.join("analysis", "results", results_dir)


def setup_argparser():
    parser = argparse.ArgumentParser(description="SMIT segmentation inference")
    parser.add_argument("--data_root", "--data_dir", dest="data_root", default=get_default_data_root(), type=str,
                        help="dataset root containing manifests and images")
    parser.add_argument("--dataset_json", "--json_list", dest="dataset_json", default="heartsub_master.json", type=str,
                        help="dataset json manifest")
    parser.add_argument("--dataset_split", "--datasets", dest="dataset_split", default="set1_cnc64_validation", type=str,
                        help="dataset split inside the json manifest")
    parser.add_argument("--results_dir", default="results", type=str, help="main output directory")
    parser.add_argument("--output_dir", default=None, type=str, help="output subdirectory or explicit path")
    parser.add_argument("--model_name", default="smit", type=str, help="model name")
    parser.add_argument("--pretrained_model_path", default=None, type=str, help="path to the checkpoint to load")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=10, type=int, help="number of output channels")
    parser.add_argument("--norm", "--norm_name", dest="norm", default="instance", type=str,
                        help="normalization for encoder/decoder")
    parser.add_argument("--orientation", default="RAS", type=str, help="orientation code applied before inference")
    parser.add_argument("--a_min", default=-200.0, type=float, help="lower CT intensity bound")
    parser.add_argument("--a_max", default=300.0, type=float, help="upper CT intensity bound")
    parser.add_argument("--b_min", default=0.0, type=float, help="scaled intensity minimum")
    parser.add_argument("--b_max", default=1.0, type=float, help="scaled intensity maximum")
    parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
    parser.add_argument("--sw_batch_size", default=12, type=int, help="sliding window batch size")
    parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
    parser.add_argument("--distributed", action="store_true", help="use distributed inference")
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--rank", default=0, type=int, help="rank of the process")
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for distributed inference")
    parser.add_argument("--dist_url", default="env://", type=str, help="url for distributed inference")
    parser.add_argument("--dist_backend", default="nccl", type=str, help="distributed backend")
    return parser




def setup_distributed(args):
    if not args.distributed:
        return

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if args.world_size <= 1:
        args.distributed = False
        return

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args):
    return not args.distributed or args.rank == 0

def get_spacing(args):
    return (args.space_x, args.space_y, args.space_z)


def get_roi_size(args):
    return (args.roi_x, args.roi_y, args.roi_z)


def get_output_directory(args):
    results_root = resolve_results_dir(args.results_dir)
    results_bucket = infer_results_bucket(args.dataset_split)
    if os.path.basename(os.path.normpath(results_root)) != results_bucket:
        results_root = os.path.join(results_root, results_bucket)
    if args.output_dir:
        if os.path.isabs(args.output_dir) or args.output_dir.startswith("analysis/results"):
            return args.output_dir
        return os.path.join(results_root, args.output_dir)
    return results_root


def main():
    args = setup_argparser().parse_args()

    if args.model_name != "smit":
        raise ValueError(f"Unsupported model_name '{args.model_name}'. Only 'smit' is supported here.")

    setup_distributed(args)

    dataset_json = resolve_dataset_json(args.data_root, args.dataset_json)
    spacing = get_spacing(args)
    roi_size = get_roi_size(args)

    if args.distributed:
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.AddChanneld(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes=args.orientation),
            transforms.Spacingd(keys=["image"], pixdim=spacing, mode=("bilinear")),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            transforms.CropForegroundd(keys=["image"], source_key="image"),
            transforms.SpatialPadd(keys=["image"], spatial_size=list(roi_size)),
            transforms.ToTensord(keys=["image"]),
        ]
    )

    post_transforms = transforms.Compose(
        [
            transforms.EnsureTyped(keys="pred"),
            transforms.Invertd(
                keys="pred",
                transform=test_transform,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=True,
                to_tensor=True,
            ),
        ]
    )
    post_transforms_softmax = transforms.AsDiscreted(keys="pred", argmax=True)

    test_files = load_decathlon_datalist(dataset_json, True, args.dataset_split, base_dir=args.data_root)
    test_ds = data.Dataset(test_files, transform=test_transform)
    test_sampler = DistributedSampler(test_ds, shuffle=False) if args.distributed else None
    test_loader = data.DataLoader(test_ds, batch_size=1, shuffle=False, sampler=test_sampler)

    config = configs_smit.get_SMIT_128_bias_True()
    model = smit.SMIT_3D_Seg(config, out_channels=args.out_channels, norm_name=args.norm)
    model_dict = torch.load(args.pretrained_model_path, map_location="cpu")["state_dict"]
    model.load_state_dict(model_dict, strict=True)
    model.eval()
    model.to(device)

    output_directory = get_output_directory(args)
    if is_main_process(args):
        os.makedirs(output_directory, exist_ok=True)
    if args.distributed:
        dist.barrier()

    loader_iter = tqdm(test_loader) if is_main_process(args) else test_loader

    with torch.no_grad():
        for batch in loader_iter:
            val_inputs = batch["image"].to(device)

            img_name = batch["image_meta_dict"]["filename_or_obj"][0]
            img_name_tosave = os.path.basename(img_name)
            batch["pred"] = sliding_window_inference(
                inputs=val_inputs,
                roi_size=roi_size,
                sw_batch_size=args.sw_batch_size,
                predictor=model,
                overlap=args.infer_overlap,
                mode="gaussian",
                progress=False,
            )

            batch = [post_transforms(i) for i in decollate_batch(batch)]

            seg_batch = post_transforms_softmax(batch[0])
            seg_ori_size = seg_batch["pred"].cpu().numpy().astype(np.uint8)
            seg_ori_size = np.squeeze(seg_ori_size)
            image_meta = seg_batch["image_meta_dict"]
            affine = image_meta.get("original_affine", image_meta.get("affine", np.eye(4)))
            val_labels_ori_save = nib.Nifti1Image(seg_ori_size, affine)
            nib.save(val_labels_ori_save, os.path.join(output_directory, img_name_tosave))

    cleanup_distributed()


if __name__ == "__main__":
    main()
