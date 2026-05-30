import argparse
import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as label_connected_components
from tqdm import tqdm

from lits_metric_utils import compute_segmentation_scores, detect_lesions, detect_lesions_2

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
ALLDATASETS_ROOT = Path(__file__).resolve().parents[2] / 'data' / 'AllDatasets'

CLASS_MAP = {
    "aorta": 1,
    "pa": 2,
    "pv": 3,
    "svc": 4,
    "ivc": 5,
    "ra": 6,
    "rv": 7,
    "la": 8,
    "lv": 9,
}


def find_z_limits(volume):
    indices = np.where(volume > 0)
    z_min, z_max = np.min(indices[2]), np.max(indices[2])
    return z_min, z_max


def adjust_pred_limits(gt, pred):
    gt_z_min, gt_z_max = find_z_limits(gt)
    pred[:, :, :gt_z_min] = 0
    pred[:, :, gt_z_max + 1 :] = 0
    return pred


def compute_surface_dice(mask_gt, mask_pred, spacing, tolerance=1.0):
    from scipy.ndimage import binary_erosion

    surface_gt = mask_gt & ~binary_erosion(mask_gt)
    surface_pred = mask_pred & ~binary_erosion(mask_pred)

    if not np.any(surface_gt) or not np.any(surface_pred):
        return np.nan

    dt_gt = distance_transform_edt(~surface_gt, sampling=spacing)
    dt_pred = distance_transform_edt(~surface_pred, sampling=spacing)

    gt_border_distances = dt_pred[surface_gt]
    pred_border_distances = dt_gt[surface_pred]

    gt_within_tolerance = np.sum(gt_border_distances <= tolerance)
    pred_within_tolerance = np.sum(pred_border_distances <= tolerance)
    return (gt_within_tolerance + pred_within_tolerance) / (np.sum(surface_gt) + np.sum(surface_pred))


def compute_volume_ratio(mask_gt, mask_pred, spacing):
    voxel_volume = np.prod(spacing)
    vol_gt = np.sum(mask_gt) * voxel_volume
    vol_pred = np.sum(mask_pred) * voxel_volume
    if vol_gt == 0:
        return np.nan
    return vol_pred / vol_gt


def infer_prediction_root(args):
    if args.predictions_dir:
        return Path(args.predictions_dir)
    if args.source == 'totalseg':
        return RESULTS_ROOT / ('segmentations_totalseg_oar' if args.dataset == 'lung' else 'segmentations_totalseg_b66')
    if args.dataset == 'lung':
        return RESULTS_ROOT / 'model_outputs_lung' / args.config
    return RESULTS_ROOT / 'model_outputs_breast' / args.config


def infer_output_csv(args, prediction_root):
    if args.output_csv:
        return Path(args.output_csv)
    csv_root = RESULTS_ROOT / ('xcelrecords_oar' if args.dataset == 'lung' else 'xcelrecords_breast')
    run_name = prediction_root.name
    return csv_root / f'{run_name}_{args.classofi}.csv'


def resolve_breast_label_dir(args):
    if args.breast_label_dir:
        return Path(args.breast_label_dir)
    label_plus = ALLDATASETS_ROOT / 'Breast66' / 'label_plus'
    if label_plus.exists():
        return label_plus
    label_plus_peri = ALLDATASETS_ROOT / 'Breast66' / 'label_plus_peri'
    if label_plus_peri.exists():
        return label_plus_peri
    raise FileNotFoundError('Breast66 label_plus or label_plus_peri not found under ALLDATASETS_ROOT')


def lung_validation_items(args):
    with open(args.json_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    return payload[args.split]


def breast_validation_items(args):
    data_folder = resolve_breast_label_dir(args)
    return sorted(p for p in data_folder.iterdir() if p.name != '.DS_Store')


def process_lung_item(data_item, prediction_root, classofi):
    gt_rel = Path(data_item['label'])
    gt_path = ALLDATASETS_ROOT / gt_rel
    pred_path = prediction_root / gt_rel.name.replace('_label', '')
    case_name = gt_rel.name.replace('_label.nii.gz', '')
    return process_case(gt_path, pred_path, case_name, classofi)


def process_breast_item(gt_path, prediction_root, classofi):
    pred_path = prediction_root / gt_path.name.replace('_label', '')
    case_name = gt_path.name.replace('_label.nii.gz', '')
    return process_case(gt_path, pred_path, case_name, classofi)


def process_case(gt_path, pred_path, case_name, classofi):
    file_gt_nii_img = nib.load(str(gt_path))
    gt_spacing = file_gt_nii_img.header.get_zooms()
    gt_volume = file_gt_nii_img.get_fdata().astype(np.int8)

    file_pred_nii_img = nib.load(str(pred_path))
    pred_volume = file_pred_nii_img.get_fdata().astype(np.int8)

    ref_mask_lesion, num_reference = label_connected_components(
        gt_volume == CLASS_MAP[classofi],
        structure=generate_binary_structure(3, 3),
        output=np.int16,
    )

    if num_reference <= 0:
        return {
            'name': case_name,
            'dice': np.nan,
            'hd95': np.nan,
            'sdsc': np.nan,
            'volume_ratio': np.nan,
        }

    pred_mask_lesion, _ = label_connected_components(
        pred_volume == CLASS_MAP[classofi],
        structure=generate_binary_structure(3, 3),
        output=np.int16,
    )
    pred_mask_lesion = adjust_pred_limits(ref_mask_lesion, pred_mask_lesion.copy())

    try:
        detected_mask_lesion, mod_ref_mask, _ = detect_lesions(
            prediction_mask=pred_mask_lesion,
            reference_mask=ref_mask_lesion,
            min_overlap=0.1,
        )
    except ValueError:
        detected_mask_lesion, mod_ref_mask, _ = detect_lesions_2(
            prediction_mask=pred_mask_lesion,
            reference_mask=ref_mask_lesion,
            min_overlap=0.1,
        )

    lesion_scores = compute_segmentation_scores(
        prediction_mask=detected_mask_lesion,
        reference_mask=mod_ref_mask,
        voxel_spacing=gt_spacing,
    )

    binary_gt = (ref_mask_lesion > 0).astype(bool)
    binary_pred = (pred_mask_lesion > 0).astype(bool)

    return {
        'name': case_name,
        'dice': np.nanmean(lesion_scores['dice']),
        'hd95': np.nanmean(lesion_scores['hd95']),
        'sdsc': compute_surface_dice(binary_gt, binary_pred, gt_spacing, tolerance=1.0),
        'volume_ratio': compute_volume_ratio(binary_gt, binary_pred, gt_spacing),
    }


def main():
    parser = argparse.ArgumentParser(description='Score segmentation outputs for paper analyses.')
    parser.add_argument('--dataset', choices=['lung', 'breast'], required=True)
    parser.add_argument('--source', default='smit', choices=['smit', 'totalseg', 'nnunet'])
    parser.add_argument('--classofi', default='aorta', choices=sorted(CLASS_MAP))
    parser.add_argument('--config', default='run1_plus_cnc64_bnorm')
    parser.add_argument('--predictions_dir')
    parser.add_argument('--output_csv')
    parser.add_argument('--json_path', default='jsons/Trainval_set1.json')
    parser.add_argument('--split', default='validation')
    parser.add_argument('--breast_label_dir')
    parser.add_argument('--num_workers', type=int, default=None)
    args = parser.parse_args()

    if args.num_workers is None:
        args.num_workers = max(1, cpu_count() - 2)

    prediction_root = infer_prediction_root(args)
    output_csv = infer_output_csv(args, prediction_root)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset == 'lung':
        items = lung_validation_items(args)
        process_func = partial(process_lung_item, prediction_root=prediction_root, classofi=args.classofi)
    else:
        items = breast_validation_items(args)
        process_func = partial(process_breast_item, prediction_root=prediction_root, classofi=args.classofi)

    print(f"Processing {len(items)} {args.dataset} cases for class '{args.classofi}' using {args.num_workers} workers...")
    with Pool(processes=args.num_workers) as pool:
        score_records = list(tqdm(pool.imap(process_func, items), total=len(items), desc=f"Processing {args.classofi}"))

    pd.DataFrame(score_records).to_csv(output_csv, index=False)
    print(f'Saved scores to {output_csv}')


if __name__ == '__main__':
    main()
