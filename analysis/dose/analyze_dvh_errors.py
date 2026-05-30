"""
DVH Error Analysis - Parallelized Version - courtesy Claude
"""

import numpy as np
import nibabel as nib
import os
import os.path as osp
import glob
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import seaborn as sns


STRUCTURE_NAMES = ["AA", "PA", "PV", "SVC", "IVC", "RA", "RV", "LA", "LV"]

def extract_plot_data(results_df, model_name):
    """Extract median and IQR for plotting."""
    structures = ['AA', 'PA', 'SVC', 'IVC', 'RA', 'RV', 'LA', 'LV']
    
    medians = []
    q1s = []
    q3s = []
    
    for struct in structures:
        values = results_df[struct].dropna()
        medians.append(values.median())
        q1s.append(values.quantile(0.25))
        q3s.append(values.quantile(0.75))
    
    # Add overall
    overall = results_df['overall'].dropna()
    medians.append(overall.median())
    q1s.append(overall.quantile(0.25))
    q3s.append(overall.quantile(0.75))
    
    return {
        model_name: {
            'Median': medians,
            'Q1': q1s,
            'Q3': q3s
        }
    }

def compute_dvh_for_structure(segmentation_map, dose_map, class_label, num_bins=1000):
    """Compute DVH for a single structure."""
    dose_values = dose_map[segmentation_map == class_label]
    
    if dose_values.size == 0:
        return None, None
    
    hist, bin_edges = np.histogram(
        dose_values, 
        bins=num_bins, 
        range=(0, np.max(dose_map))
    )
    
    cumulative_volumes = np.cumsum(hist[::-1])[::-1] / len(dose_values)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return bin_centers, cumulative_volumes


def process_single_case(args):
    """Process a single case - designed for parallel execution."""
    ai_path, gt_folder, gt_label_subdir = args
    
    case_name = osp.basename(ai_path)
    
    # Construct paths
    gt_path = osp.join(
        gt_folder, 
        gt_label_subdir,
        case_name.replace('.nii.gz', '_label.nii.gz')
    )
    dose_path = osp.join(
        gt_folder,
        'dose',
        case_name.replace('.nii.gz', '_dose.nii.gz')
    )
    
    # Check if files exist
    if not osp.exists(gt_path) or not osp.exists(dose_path):
        return None
    
    try:
        # Load data
        ai_seg = nib.load(ai_path).get_fdata()
        gt_seg = nib.load(gt_path).get_fdata()
        dose_map = nib.load(dose_path).get_fdata()
        
        # Remove PV (label 3)
        ai_seg[ai_seg == 3] = 0
        gt_seg[gt_seg == 3] = 0
        
        mae_list = []
        result_row = {'case': case_name}
        
        for class_label in range(1, 10):
            bin_centers_ai, cumvol_ai = compute_dvh_for_structure(ai_seg, dose_map, class_label)
            bin_centers_gt, cumvol_gt = compute_dvh_for_structure(gt_seg, dose_map, class_label)
            
            if bin_centers_ai is None or bin_centers_gt is None:
                result_row[STRUCTURE_NAMES[class_label - 1]] = np.nan
                continue
            
            cumvol_ai_interp = np.interp(bin_centers_gt, bin_centers_ai, cumvol_ai)
            mae = np.mean(np.abs(cumvol_ai_interp - cumvol_gt)) * 100
            
            result_row[STRUCTURE_NAMES[class_label - 1]] = mae
            mae_list.append(mae)
        
        result_row['overall'] = np.mean(mae_list) if mae_list else np.nan
        return result_row
        
    except Exception as e:
        print(f"Error processing {case_name}: {e}")
        return None


def analyze_all_cases(ai_folder, gt_folder, gt_label_subdir='label_plus', output_file=None, n_workers=None, exclude_names=None):
    """Analyze DVH errors across all cases with parallel processing."""
    
    ai_files = sorted(glob.glob(osp.join(ai_folder, '*.nii.gz')))
    gt_files = sorted(glob.glob(osp.join(gt_folder, gt_label_subdir, '*_label.nii.gz')))
    
    if len(ai_files) == 0:
        raise ValueError(f"No .nii.gz files found in {ai_folder}")
    
    # Extract base names (without extensions) for matching
    ai_names = {osp.basename(f).replace('.nii.gz', ''): f for f in ai_files}
    gt_names = {osp.basename(f).replace('_label.nii.gz', ''): f for f in gt_files}
    
    # Find common names
    common_names = set(ai_names.keys()) & set(gt_names.keys())
    
    # Exclude specific filenames
    if exclude_names:
        common_names = common_names - set(exclude_names)
    
    # Print matching info
    print(f"\nAI files found: {len(ai_names)}")
    print(f"GT files found: {len(gt_names)}")
    print(f"Matched cases: {len(common_names)}")
    
    # Show unmatched files if any
    ai_only = set(ai_names.keys()) - set(gt_names.keys())
    gt_only = set(gt_names.keys()) - set(ai_names.keys())
    
    if ai_only:
        print(f"AI files without GT match: {sorted(ai_only)}")
    if gt_only:
        print(f"GT files without AI match: {sorted(gt_only)}")
    if exclude_names:
        print(f"Excluded: {exclude_names}")
    
    # Print first few matches to verify
    print(f"\nFirst 5 matched pairs:")
    for name in sorted(common_names)[:5]:
        print(f"  AI: {osp.basename(ai_names[name])}")
        print(f"  GT: {osp.basename(gt_names[name])}")
        print()
    
    # Filter to only common files
    ai_files = [ai_names[name] for name in sorted(common_names)]
    
    print(f"Proceeding with {len(ai_files)} cases\n")
    
    # Prepare arguments for parallel processing
    args_list = [(ai_path, gt_folder, gt_label_subdir) for ai_path in ai_files]
    
    # Set number of workers
    if n_workers is None:
        n_workers = min(cpu_count(), len(ai_files))
    
    print(f"Using {n_workers} workers")
    
    # Process in parallel
    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_case, args_list),
            total=len(ai_files),
            desc="Processing cases"
        ))
    
    # Filter out None results
    all_results = [r for r in results if r is not None]
    
    print(f"Successfully processed {len(all_results)}/{len(ai_files)} cases")
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Compute summary statistics
    summary = {}
    
    overall_values = results_df['overall'].dropna()
    summary['overall'] = {
        'mean': overall_values.mean(),
        'std': overall_values.std(),
        'median': overall_values.median(),
        'min': overall_values.min(),
        'max': overall_values.max(),
        'n_cases': len(overall_values)
    }
    
    summary['per_structure'] = {}
    for struct in STRUCTURE_NAMES:
        if struct in results_df.columns:
            values = results_df[struct].dropna()
            if len(values) > 0:
                summary['per_structure'][struct] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'median': values.median(),
                    'n_cases': len(values)
                }
    
    # Save results
    if output_file:
        # results_df.to_csv(output_file.replace('.txt', '.csv'), index=False)
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DVH ERROR ANALYSIS RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total cases analyzed: {summary['overall']['n_cases']}\n\n")
            f.write("OVERALL DVH ERROR (across all structures):\n")
            f.write(f"  Mean ± SD:  {summary['overall']['mean']:.2f} ± {summary['overall']['std']:.2f}%\n")
            f.write(f"  Median:     {summary['overall']['median']:.2f}%\n")
            f.write(f"  Range:      {summary['overall']['min']:.2f} - {summary['overall']['max']:.2f}%\n\n")
            f.write("PER-STRUCTURE DVH ERROR:\n")
            for struct, stats in summary['per_structure'].items():
                f.write(f"  {struct:4s}: {stats['mean']:.2f} ± {stats['std']:.2f}% (n={stats['n_cases']})\n")
            f.write("\n" + "="*80 + "\n")
            f.write("\nFOR MANUSCRIPT:\n")
            f.write(f"Mean absolute DVH difference: {summary['overall']['mean']:.1f} ± "
                   f"{summary['overall']['std']:.1f} percentage points\n")
            f.write("="*80 + "\n")
        
        print(f"\nResults saved to {output_file}")
        
    # Print summary
    print("\n" + "="*80)
    print("DVH ERROR ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nTotal cases analyzed: {summary['overall']['n_cases']}")
    print(f"\nOVERALL DVH ERROR: {summary['overall']['mean']:.2f} ± {summary['overall']['std']:.2f}%")
    print(f"Median: {summary['overall']['median']:.2f}%")
    
    print("\nPER-STRUCTURE DVH ERROR:")
    for struct, stats in summary['per_structure'].items():
        print(f"  {struct:4s}: {stats['mean']:.2f} ± {stats['std']:.2f}%")
    
    print("="*80 + "\n")
    
    return results_df, summary

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
ALLDATASETS_ROOT = Path(__file__).resolve().parents[2] / 'data' / 'AllDatasets'

def get_config(mode, model_name):
    configs = {
        'oar': {
            'ai_folder': str(RESULTS_ROOT / 'model_outputs_lung' / model_name),
            'gt_folder': str(ALLDATASETS_ROOT / 'HeartSubv2_substructs'),
            'gt_label_subdir': 'label_plus',
            'output_file': f'dvh_metrics_oar_{model_name}.txt'
        },
        'breast66': {
            'ai_folder': str(RESULTS_ROOT / 'model_outputs_breast' / model_name),
            'gt_folder': str(ALLDATASETS_ROOT / 'Breast66'),
            'gt_label_subdir': 'label_plus',
            'output_file': f'dvh_metrics_breast66_{model_name}.txt'
        }
    }
    return configs[mode]


if __name__ == "__main__":
    model_name = 'run1_plus_cnc64_bnorm'
    mode = 'oar'
    
    config = get_config(mode, model_name)
    results_df, summary = analyze_all_cases(
        ai_folder=config['ai_folder'],
        gt_folder=config['gt_folder'],
        gt_label_subdir=config['gt_label_subdir'],
        output_file=config['output_file'],
        n_workers=12,
        exclude_names=['XXXX']  # Add filenames to exclude here
    )
    
    #%%
    # Assuming results_df from your DVH analysis
    # Melt to long format
    df_long = results_df.melt(
        id_vars=['case'], 
        value_vars=['AA', 'PA', 'SVC', 'IVC', 'RA', 'RV', 'LA', 'LV'],
        var_name='Substructure', 
        value_name='DVH Error (%)'
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    vessels = ['AA', 'PA', 'SVC', 'IVC']
    chambers = ['RA', 'RV', 'LA', 'LV']
    
    # Box plots show median, IQR, and outliers properly
    sns.boxplot(data=df_long[df_long['Substructure'].isin(vessels)],
                x='Substructure', y='DVH Error (%)', 
                order=vessels, ax=ax1, palette='colorblind')
    ax1.set_xlabel('Great vessels', fontsize=14)
    ax1.set_ylabel('Mean DVH difference (percentage points)', fontsize=12)
    ax1.set_ylim(0, 2)
    
    sns.boxplot(data=df_long[df_long['Substructure'].isin(chambers)],
                x='Substructure', y='DVH Error (%)', 
                order=chambers, ax=ax2, palette='colorblind')
    ax2.set_xlabel('Chambers', fontsize=14)
    ax2.set_ylabel('')  # Remove duplicate y-label on right plot
    ax2.set_ylim(0, 2)
    
    plt.tight_layout()
    
    fig.text(0.5, -0.005, 
         'Box represents median and IQR (25th-75th percentile). Whiskers extend to 1.5×IQR. Points beyond this range are outliers. \nOutliers beyond 2 percentage points not shown.',
         ha='center', va='top', fontsize=10, wrap=True)
    # plt.savefig(f'tablesandfigures/dvh_metrics_boxplot_{model_name}_{mode}.pdf', bbox_inches='tight', dpi=300)
    
    structures = ['AA', 'PA', 'SVC', 'IVC', 'RA', 'RV', 'LA', 'LV']

    print("Structure | Median | IQR (Q1-Q3) | Min | Max")
    print("-" * 50)
    
    for struct in structures:
        values = results_df[struct].dropna()
        median = values.median()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        
        print(f"{struct:8s} | {median:5.2f}  | {q1:.2f} - {q3:.2f} ({iqr:.2f}) | {values.min():.2f} | {values.max():.2f}")
    
    # Overall summary
    print("\n" + "=" * 50)
    overall = results_df['overall'].dropna()
    print(f"Overall  | {overall.median():.2f}  | {overall.quantile(0.25):.2f} - {overall.quantile(0.75):.2f}")
