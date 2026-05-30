"""
CARDIAC/BREAST SEGMENTATION - COMPLETE ANALYSIS WITH PENALIZATION
Single script with fair comparison: penalizes failed segmentations instead of dropping

Penalization Strategy:
- ONLY penalize if at least ONE model has a valid (non-NaN) detection for a case
- If ALL models have NaN/missing for a case → exclude that case entirely
- Missing detections get penalty: DSC=0, HD95=20, sDSC=0.1

Reference Model: SMIT_Balanced_Frozen (all comparisons made against this)

Usage:
    python generate_figures_penalized.py --directory xcelrecords_oar
    python generate_figures_penalized.py --directory xcelrecords_breast
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ============================================================================
#                           CONFIGURATION
# ============================================================================

sns.set_palette("colorblind")
colorblind_colors = sns.color_palette("colorblind", n_colors=4)

MODELS = {
    'SMIT_Balanced': 'run1_plus_cnc64_bnorm',
    'SMIT_All': 'run1_plus_bnorm',
    'nnU-Net': 'nnunet_lung_cnc64',
    'TotalSegmentator': 'totalseg',
}

# Reference model for all comparisons
REFERENCE_MODEL = 'SMIT_Balanced'

MODEL_ORDER = ['SMIT_Balanced', 'SMIT_All', 'nnU-Net', 'TotalSegmentator']
MODEL_DISPLAY = ['SMIT-Balanced', 'SMIT-Oracle', 'nnU-Net', 'TotalSegmentator']

# Colors from seaborn colorblind palette
COLORS = {
    'SMIT_Balanced': colorblind_colors[0],
    'SMIT_All': colorblind_colors[1],
    'nnU-Net': colorblind_colors[2],
    'TotalSegmentator': colorblind_colors[3],
}

# Penalty values for failed segmentations.
# Three sentinel modes are supported (selected via --sentinel-mode):
#   - paper:    values reported in the manuscript (HD95=50, VR=10, DSC=sDSC=0)
#   - reviewer: DSC=sDSC=0, HD95/VR set to per-organ worst observed (Reviewer Q14)
#   - exclude:  failed cases are dropped entirely from paired tests
SENTINEL_MODES = {
    'paper':    {'dice': 0.0, 'hd95': 50.0, 'sdsc': 0.0,  'volume_ratio': 10.0},
    'reviewer': {'dice': 0.0, 'hd95': None, 'sdsc': 0.0,  'volume_ratio': None},  # None => per-organ worst observed
    'exclude':  {'dice': 0.0, 'hd95': 0.0,  'sdsc': 0.0,  'volume_ratio': 1.0},   # unused; cases dropped instead
}
# Default (kept for backward compatibility with any external import).
PENALTY_VALUES = SENTINEL_MODES['paper']

# Significance bracket display mode: 'tiered' (*/**/***), 'simple' (* only), 'none' (no brackets)
BRACKET_MODE = 'tiered'

def _sig_text(p_corr):
    """Format significance star text per BRACKET_MODE."""
    if BRACKET_MODE == 'none' or p_corr != p_corr:  # NaN check
        return ''
    if BRACKET_MODE == 'simple':
        return '*' if p_corr < 0.05 else ''
    if p_corr < 0.001: return '***'
    if p_corr < 0.01:  return '**'
    if p_corr < 0.05:  return '*'
    return ''

ORGAN_CONFIG = {
    'xcelrecords_oar': {
        'organs': ['aorta', 'pa', 'svc', 'ivc', 'ra', 'rv', 'la', 'lv'],
        'display': ['AA', 'PA', 'SVC', 'IVC', 'RA', 'RV', 'LA', 'LV'],
        'title': 'Cardiac Substructures'
    },
    'xcelrecords_breast': {
        'organs': ['aorta', 'pa', 'svc', 'ivc', 'ra', 'rv', 'la', 'lv'],
        'display': ['AA', 'PA', 'SVC', 'IVC', 'RA', 'RV', 'LA', 'LV'],
        'title': 'Cardiac Substructures'
    }
}

# Output directory
RESULTS_ROOT = Path(__file__).resolve().parents[1] / 'results'
OUTPUT_DIR = RESULTS_ROOT / 'tablesandfigures'

# ============================================================================
#                           DATA LOADING WITH SMART PENALIZATION
# ============================================================================

def compute_failure_rates(raw_data, organ_list, metric):
    """Return dict[(model, organ)] -> (n_failed, n_total) before any penalization."""
    fr = {}
    for model_name in MODELS.keys():
        for organ in organ_list:
            key = (model_name, organ)
            if key not in raw_data:
                fr[key] = (None, None)
                continue
            df = raw_data[key]
            if metric not in df.columns:
                fr[key] = (None, None)
                continue
            n_total = len(df)
            n_failed = int(df[metric].isna().sum())
            fr[key] = (n_failed, n_total)
    return fr


def print_failure_table(failure_rates, organ_list, organ_display, metric):
    print(f"\n{'='*80}")
    print(f"FAILURE-RATE TABLE (metric: {metric.upper()})")
    print(f"{'='*80}")
    header = f"{'Model':22s}" + ''.join([f"{d:>10s}" for d in organ_display])
    print(header)
    for model_name in MODEL_ORDER:
        row = f"{model_name:22s}"
        for organ in organ_list:
            nf, nt = failure_rates.get((model_name, organ), (None, None))
            if nf is None:
                row += f"{'N/A':>10s}"
            else:
                row += f"{f'{nf}/{nt}':>10s}"
        print(row)


def load_and_penalize_data(directory, organ_list, metric, sentinel_mode='paper'):
    """
    Load all data and apply penalization for missing detections.

    Sentinel modes:
      - 'paper':    static sentinels from manuscript text
      - 'reviewer': DSC=sDSC=0; HD95 and VR set to per-organ worst observed valid value
      - 'exclude':  drop any case where ANY model failed (intersection of valid cases)
    """
    print(f"\n{'='*80}")
    print(f"LOADING DATA (metric: {metric.upper()}, sentinel_mode: {sentinel_mode})")
    print(f"{'='*80}")
    if sentinel_mode == 'exclude':
        print("Strategy: drop cases where ANY model failed (paired-intersection)")
    else:
        print(f"Sentinel values: {SENTINEL_MODES[sentinel_mode]}")
        print(f"Strategy:")
        print(f"  - If ≥1 model has valid detection → penalize models with NaN/missing")
        print(f"  - If ALL models have NaN/missing → exclude case entirely")
    
    # Step 1: Load all raw data
    print(f"\nStep 1: Loading raw CSV files...")
    raw_data = {}
    for model_name, prefix in MODELS.items():
        for organ in organ_list:
            filepath = os.path.join(directory, f"{prefix}_{organ}.csv")
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, dtype={'name': str})
                    raw_data[(model_name, organ)] = df
                except Exception as e:
                    print(f"  Warning: Could not load {filepath}: {e}")
    
    print(f"  Loaded {len(raw_data)} model-organ combinations")
    
    # Step 2: For each organ, find cases where AT LEAST ONE model has valid detection
    print(f"\nStep 2: Finding valid cases per organ (at least one model detected)...")
    organ_valid_cases = {}
    organ_excluded_cases = {}
    
    for organ in organ_list:
        # Collect all case IDs and their validity across models
        case_validity = {}  # case_id -> {model: has_valid_value}
        
        for model_name in MODELS.keys():
            key = (model_name, organ)
            if key in raw_data:
                df = raw_data[key]
                # Check if metric column exists
                if metric not in df.columns:
                    print(f"  Warning: {model_name}/{organ} missing '{metric}' column, skipping")
                    continue
                for _, row in df.iterrows():
                    case_id = str(row['name'])
                    if case_id not in case_validity:
                        case_validity[case_id] = {}
                    # Check if this model has a valid (non-NaN) value
                    case_validity[case_id][model_name] = pd.notna(row[metric])
        
        # Determine which cases have at least one valid detection
        valid_cases = []
        excluded_cases = []
        
        for case_id, model_valid in case_validity.items():
            if any(model_valid.values()):
                # At least one model has valid value → include this case
                valid_cases.append(case_id)
            else:
                # ALL models have NaN → exclude this case
                excluded_cases.append(case_id)
        
        organ_valid_cases[organ] = sorted(valid_cases)
        organ_excluded_cases[organ] = excluded_cases
        
        n_excluded = len(excluded_cases)
        if n_excluded > 0:
            print(f"  {organ:10s}: {len(valid_cases)} valid cases, {n_excluded} excluded (all NaN)")
        else:
            print(f"  {organ:10s}: {len(valid_cases)} valid cases")
    
    # Step 2b: Mode-specific adjustments to valid_cases and per-organ sentinels.
    # In 'exclude' mode, restrict valid_cases to the intersection where ALL models detected.
    # In 'reviewer' mode, set HD95 / VR sentinels to per-organ worst-observed valid value.
    organ_sentinels = {}  # organ -> dict of metric -> sentinel value
    for organ in organ_list:
        if sentinel_mode == 'exclude':
            # Intersection: only cases where every model has a non-NaN value
            shared = None
            for model_name in MODELS.keys():
                key = (model_name, organ)
                if key not in raw_data or metric not in raw_data[key].columns:
                    shared = set()
                    break
                df = raw_data[key]
                detected = set(df.loc[df[metric].notna(), 'name'].astype(str).tolist())
                shared = detected if shared is None else (shared & detected)
            organ_valid_cases[organ] = sorted(shared) if shared else []

        if sentinel_mode == 'reviewer':
            # Per-organ worst-observed valid HD95 and VR (DSC, sDSC = 0 by definition)
            all_hd95, all_vr = [], []
            for model_name in MODELS.keys():
                key = (model_name, organ)
                if key in raw_data:
                    df = raw_data[key]
                    if 'hd95' in df.columns:
                        all_hd95.extend(df['hd95'].dropna().tolist())
                    if 'volume_ratio' in df.columns:
                        all_vr.extend(df['volume_ratio'].dropna().tolist())
            organ_sentinels[organ] = {
                'dice': 0.0,
                'sdsc': 0.0,
                'hd95': max(all_hd95) if all_hd95 else 50.0,
                'volume_ratio': max(all_vr) if all_vr else 10.0,
            }
        else:
            organ_sentinels[organ] = dict(SENTINEL_MODES[sentinel_mode])

    # Step 3: Apply penalization only for valid cases
    print(f"\nStep 3: Applying penalization for missing detections...")
    penalized_data = {}
    total_penalties = 0
    total_excluded = sum(len(v) for v in organ_excluded_cases.values())

    for organ in organ_list:
        valid_cases = set(organ_valid_cases[organ])
        sentinel = organ_sentinels[organ]
        
        for model_name, prefix in MODELS.items():
            key = (model_name, organ)
            
            if key in raw_data:
                df = raw_data[key].copy()
                
                # Check if metric column exists
                if metric not in df.columns:
                    # Treat as all missing - penalize all valid cases
                    print(f"  {model_name:20s}/{organ:10s}: {len(valid_cases)} penalties ('{metric}' column missing)")
                    total_penalties += len(valid_cases)
                    rows = []
                    for case_id in valid_cases:
                        rows.append({
                            'name': case_id,
                            'dice': sentinel['dice'],
                            'hd95': sentinel['hd95'],
                            'sdsc': sentinel['sdsc'],
                            'volume_ratio': sentinel['volume_ratio']
                        })
                    penalized_data[key] = pd.DataFrame(rows)
                    continue
                
                # Filter to only valid cases
                df = df[df['name'].isin(valid_cases)].copy()
                
                existing_cases = set(df['name'].tolist())
                missing_cases = valid_cases - existing_cases
                nan_cases = df[df[metric].isna()]['name'].tolist()
                
                n_penalties = len(missing_cases) + len(nan_cases)
                
                if n_penalties > 0:
                    print(f"  {model_name:20s}/{organ:10s}: {n_penalties} penalties "
                          f"({len(missing_cases)} missing, {len(nan_cases)} NaN)")
                    total_penalties += n_penalties
                    
                    # Fill NaN with penalty value
                    df[metric] = df[metric].fillna(sentinel[metric])
                    
                    # Add missing cases with penalty values
                    for case_id in missing_cases:
                        new_row = {
                            'name': case_id,
                            'dice': sentinel['dice'],
                            'hd95': sentinel['hd95'],
                            'sdsc': sentinel['sdsc'],
                            'volume_ratio': sentinel['volume_ratio']
                        }
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                
                penalized_data[key] = df
            else:
                # Model-organ file doesn't exist - create with all penalties for valid cases
                print(f"  {model_name:20s}/{organ:10s}: {len(valid_cases)} penalties (file missing)")
                total_penalties += len(valid_cases)
                
                rows = []
                for case_id in valid_cases:
                    rows.append({
                        'name': case_id,
                        'dice': sentinel['dice'],
                        'hd95': sentinel['hd95'],
                        'sdsc': sentinel['sdsc'],
                        'volume_ratio': sentinel['volume_ratio']
                    })
                penalized_data[key] = pd.DataFrame(rows)
    
    print(f"\n{'='*80}")
    print(f"PENALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total valid cases: {sum(len(v) for v in organ_valid_cases.values())}")
    print(f"Total excluded (all NaN): {total_excluded}")
    print(f"Total penalties applied: {total_penalties}")
    print(f"Result: All models now have identical case coverage per organ")

    return penalized_data, organ_valid_cases, raw_data, organ_sentinels

# ============================================================================
#                           HELPER FUNCTIONS
# ============================================================================

def compute_stats(values):
    """Compute mean ± std"""
    if len(values) == 0:
        return np.nan, np.nan, 0
    return np.mean(values), np.std(values, ddof=1), len(values)

def mann_whitney_test(vals1, vals2):
    """Perform Mann-Whitney U test (kept for unpaired use only)."""
    if len(vals1) > 0 and len(vals2) > 0:
        try:
            _, p = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
            return p
        except:
            return np.nan
    return np.nan


def wilcoxon_paired_test(ref_df, comp_df, metric):
    """
    Paired Wilcoxon signed-rank test on case-aligned data.

    Sorts both DataFrames by 'name', verifies case_id alignment, then runs
    scipy.stats.wilcoxon. Returns NaN if alignment fails, sample is empty,
    or both arrays are identical (Wilcoxon undefined).
    """
    if ref_df is None or comp_df is None:
        return np.nan
    if metric not in ref_df.columns or metric not in comp_df.columns:
        return np.nan
    ref_sorted = ref_df.sort_values('name').reset_index(drop=True)
    comp_sorted = comp_df.sort_values('name').reset_index(drop=True)
    if len(ref_sorted) != len(comp_sorted):
        return np.nan
    if not (ref_sorted['name'].astype(str).values ==
            comp_sorted['name'].astype(str).values).all():
        return np.nan
    v1 = ref_sorted[metric].values.astype(float)
    v2 = comp_sorted[metric].values.astype(float)
    if len(v1) == 0 or np.array_equal(v1, v2):
        return np.nan
    try:
        _, p = stats.wilcoxon(v1, v2, zero_method='wilcox', alternative='two-sided')
        return p
    except Exception:
        return np.nan


def apply_bonferroni(p_value, n_comparisons):
    """Apply Bonferroni correction"""
    if np.isnan(p_value):
        return np.nan
    return min(p_value * n_comparisons, 1.0)

def format_stats(mean, std, metric):
    """Format statistics"""
    if np.isnan(mean):
        return "N/A"
    if metric in ['dice', 'sdsc']:
        return f"{mean:.2f} ± {std:.2f}"
    else:  # hd95
        return f"{mean:.1f} ± {std:.1f}"

def format_pvalue(p):
    """Format p-value"""
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "< 0.001"
    else:
        return f"{p:.3f}"

# ============================================================================
#                           FIGURE 1: BOXPLOT
# ============================================================================

def generate_boxplot(penalized_data, organ_list, organ_display, title, metric, output_prefix):
    """Generate main comparison boxplot"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 1: Main Comparison Boxplot")
    print("="*80)
    
    all_data = []
    for model_name in MODELS.keys():
        for organ in organ_list:
            key = (model_name, organ)
            if key in penalized_data:
                df = penalized_data[key][['name', metric]].copy()
                df['group'] = model_name
                df['organ'] = organ
                all_data.append(df)
    
    if not all_data:
        print("  ✗ No data found")
        return
    
    final_df = pd.concat(all_data, ignore_index=True)
    
    print(f"  Total data points: {len(final_df)}")
    for model_name in MODEL_ORDER:
        model_df = final_df[final_df['group'] == model_name]
        print(f"    {model_name:20s}: {len(model_df)} measurements")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.boxplot(
        data=final_df,
        x='organ',
        y=metric,
        hue='group',
        order=organ_list,
        hue_order=MODEL_ORDER,
        palette=COLORS,
        dodge=True,
        ax=ax,
        linewidth=1.5,
        fliersize=3
    )
    
    ax.set_xlabel('', fontsize=14)
    if metric == 'dice':
        ax.set_ylabel('Dice Similarity Coefficient', fontsize=14)
        ax.set_ylim([-0.02, 1.0])
    elif metric == 'sdsc':
        ax.set_ylabel('Surface Dice Similarity Coefficient', fontsize=14)
        ax.set_ylim([-0.02, 1.0])
    else:  # hd95
        ax.set_ylabel('HD95 (mm)', fontsize=14)
        ax.set_ylim([-5, 62])
    
    ax.set_xticklabels(organ_display, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles=handles,
        labels=MODEL_DISPLAY,
        fontsize=10,
        loc='upper right',
        ncols=3,
        fancybox=True,
    )
    
    # ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()

    # Add caption below figure
    caption_text = {
        'hd95': 'HD95: Box: median, 25th-75th percentiles; whiskers: 1.5×IQR.',
        'dice': 'DSC: Box: median, 25th-75th percentiles; whiskers: 1.5×IQR.',
        'sdsc': 'sDSC: Box: median, 25th-75th percentiles; whiskers: 1.5×IQR.',
        'volume_ratio': 'VR: Box: median, 25th-75th percentiles; whiskers: 1.5×IQR.'
    }
    
    fig.text(0.8, -0.02, caption_text[metric], 
             ha='center', va='top', fontsize=10, 
             wrap=True, transform=fig.transFigure)
    
    # Adjust bottom margin to accommodate caption
    plt.subplots_adjust(bottom=0.04)

    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    filename = os.path.join(OUTPUT_DIR, f'{output_prefix}_figure1_boxplot_{metric}')
    plt.savefig(f'{filename}.pdf', bbox_inches='tight')
    # print(f"  ✓ Saved: {filename}.pdf")
    plt.close()

def generate_boxplot(penalized_data, organ_list, organ_display, title, metric, output_prefix):
    """Generate main comparison boxplot with significance annotations"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 1: Main Comparison Boxplot")
    print("="*80)
    
    all_data = []
    for model_name in MODELS.keys():
        for organ in organ_list:
            key = (model_name, organ)
            if key in penalized_data:
                df = penalized_data[key][['name', metric]].copy()
                df['group'] = model_name
                df['organ'] = organ
                all_data.append(df)
    
    if not all_data:
        print("  ✗ No data found")
        return
    
    final_df = pd.concat(all_data, ignore_index=True)
    
    print(f"  Total data points: {len(final_df)}")
    for model_name in MODEL_ORDER:
        model_df = final_df[final_df['group'] == model_name]
        print(f"    {model_name:20s}: {len(model_df)} measurements")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.boxplot(
        data=final_df,
        x='organ',
        y=metric,
        hue='group',
        order=organ_list,
        hue_order=MODEL_ORDER,
        palette=COLORS,
        dodge=True,
        ax=ax,
        linewidth=1.5,
        fliersize=3
    )
    
    # Compute significance and add brackets with stars
    n_comparisons = len(MODEL_ORDER) - 1
    ref_model_idx = MODEL_ORDER.index(REFERENCE_MODEL)
    
    # First pass: count significant comparisons per organ
    sig_counts = {}
    for organ_idx, organ in enumerate(organ_list):
        ref_key = (REFERENCE_MODEL, organ)
        if ref_key not in penalized_data:
            continue
        
        ref_values = penalized_data[ref_key][metric].values
        sig_count = 0
        
        for comp_idx, model_name in enumerate(MODEL_ORDER):
            if model_name == REFERENCE_MODEL:
                continue
            
            comp_key = (model_name, organ)
            if comp_key not in penalized_data:
                continue
            
            comp_values = penalized_data[comp_key][metric].values
            p_raw = wilcoxon_paired_test(penalized_data[ref_key], penalized_data[comp_key], metric)
            p_corr = apply_bonferroni(p_raw, n_comparisons)

            if p_corr < 0.05:
                sig_count += 1
        
        sig_counts[organ] = sig_count
    
    # Second pass: draw brackets with conditional stacking
    for organ_idx, organ in enumerate(organ_list):
        ref_key = (REFERENCE_MODEL, organ)
        if ref_key not in penalized_data:
            continue
        
        ref_values = penalized_data[ref_key][metric].values
        
        # Calculate box positions
        n_models = len(MODEL_ORDER)
        box_width = 0.8 / n_models
        
        # Reference model x position
        ref_x_offset = (ref_model_idx - n_models/2 + 0.5) * box_width
        ref_x_pos = organ_idx + ref_x_offset
        
        # Track which comparison this is (for stacking if needed)
        comparison_counter = 0
        
        # For each comparison model
        for comp_idx, model_name in enumerate(MODEL_ORDER):
            if model_name == REFERENCE_MODEL:
                continue
            
            comp_key = (model_name, organ)
            if comp_key not in penalized_data:
                continue
            
            comp_values = penalized_data[comp_key][metric].values

            # Paired Wilcoxon signed-rank test with Bonferroni correction
            p_raw = wilcoxon_paired_test(penalized_data[ref_key], penalized_data[comp_key], metric)
            p_corr = apply_bonferroni(p_raw, n_comparisons)
            
            # Convert to significance stars (respects BRACKET_MODE)
            sig_text = _sig_text(p_corr)

            if sig_text:
                # Comparison model x position
                comp_x_offset = (comp_idx - n_models/2 + 0.5) * box_width
                comp_x_pos = organ_idx + comp_x_offset
                
                # Get y position (max of both models + offset)
                ref_max = ref_values.max()
                comp_max = comp_values.max()
                y_base = max(ref_max, comp_max)
                
                # Only use comparison_number for stacking if multiple significant comparisons
                use_stacking = sig_counts[organ] > 1
                comparison_number = comparison_counter if use_stacking else 0
                
                # Brackets are placed in a dedicated zone above the natural data ceiling
                # so they never collide with boxes, the legend, or outliers.
                if metric == 'dice':
                    bracket_zone_floor = 1.03
                    y_bracket = bracket_zone_floor + comparison_number * 0.06
                    bar_height = 0.005
                    y_limit = 1.27
                    y_base = bracket_zone_floor - 0.02  # vertical line bottom
                    gap = 0.0
                elif metric == 'sdsc':
                    bracket_zone_floor = 1.03
                    y_bracket = bracket_zone_floor + comparison_number * 0.06
                    bar_height = 0.005
                    y_limit = 1.27
                    y_base = bracket_zone_floor - 0.02
                    gap = 0.0
                elif metric == 'volume_ratio':
                    bracket_zone_floor = 2.55
                    y_bracket = bracket_zone_floor + comparison_number * 0.20
                    bar_height = 0.04
                    y_limit = 3.3
                    y_base = bracket_zone_floor - 0.05
                    gap = 0.0
                else:  # hd95
                    bracket_height = 1.5 + comparison_number * 3
                    gap = 1.5
                    y_base = max(y_base, 50.0)  # don't let brackets sit on outliers below 50
                    y_bracket = y_base + gap + bracket_height
                    bar_height = 1
                    y_limit = 73
                
                # Only draw if bracket fits within reasonable range
                if y_bracket > y_limit:
                    continue  # Skip this annotation
                
                # Draw bracket with gap from whiskers
                # Left vertical (starts above the data)
                ax.plot([ref_x_pos, ref_x_pos], [y_base + gap, y_bracket], 
                       'k-', linewidth=1, zorder=10)
                
                # Horizontal line
                ax.plot([ref_x_pos, comp_x_pos], [y_bracket, y_bracket], 
                       'k-', linewidth=1, zorder=10)
                
                # Right vertical (starts above the data)
                ax.plot([comp_x_pos, comp_x_pos], [y_base + gap, y_bracket], 
                       'k-', linewidth=1, zorder=10)
                
                # Star annotation at center of bracket
                x_center = (ref_x_pos + comp_x_pos) / 2
                
                # Place text ABOVE the horizontal bracket line
                if metric in ['dice', 'sdsc']:
                    text_y = y_bracket + 0.005 # Fixed offset above bracket
                elif metric == 'volume_ratio':
                    text_y = y_bracket + 0.05
                else:
                    text_y = y_bracket + bar_height - 0.2
                   
                ax.text(x_center, text_y, sig_text, 
                       ha='center', va='bottom', 
                       fontsize=9, fontweight='bold',
                       color='black', zorder=11)
                
                comparison_counter += 1
    
    ax.set_xlabel('', fontsize=14)
    if metric == 'dice':
        ax.set_ylabel('Dice Similarity Coefficient', fontsize=14)
        ax.set_ylim([-0.02, 1.30])  # Extended bracket headroom; brackets sit above 1.0
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    elif metric == 'sdsc':
        ax.set_ylabel('Surface Dice Similarity Coefficient', fontsize=14)
        ax.set_ylim([-0.02, 1.30])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    elif metric == 'volume_ratio':
        ax.set_ylabel('Volume Ratio (Pred/GT)', fontsize=14)
        ax.set_ylim([0, 3.5])  # Headroom for outliers + brackets
    else:  # hd95
        ax.set_ylabel('HD95 (mm)', fontsize=14)
        ax.set_ylim([-5, 75])  # Headroom for stacked brackets

    ax.set_xticklabels(organ_display, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    handles, labels = ax.get_legend_handles_labels()
    # Legend placed above the axes in a single horizontal row (out of bracket region)
    legend = ax.legend(
        handles=handles,
        labels=MODEL_DISPLAY,
        fontsize=10,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncols=4,
        fancybox=True,
        frameon=True,
    )
    
    ax.set_axisbelow(True)
    
    plt.tight_layout()

    # Add caption below figure
    caption_text = {
        'hd95': 'HD95: Box: median, 25th-75th percentiles; whiskers: 1.5×IQR. Brackets show comparisons to SMIT-Balanced. * p≤0.05, ** p≤0.01,*** p≤0.001, **** p≤0.0001.',
        'dice': 'DSC: Box: median, 25th-75th percentiles; whiskers: 1.5×IQR. Brackets show comparisons to SMIT-Balanced.* p≤0.05, ** p≤0.01,*** p≤0.001, **** p≤0.0001.',
        'sdsc': 'sDSC: Box: median, 25th-75th percentiles; whiskers: 1.5×IQR. Brackets show comparisons to SMIT-Balanced. * p≤0.05, ** p≤0.01,*** p≤0.001, **** p≤0.0001.',
        'volume_ratio': 'VR: Box: median, 25th-75th percentiles; whiskers: 1.5×IQR. Brackets show comparisons to SMIT-Balanced. * p≤0.05, ** p≤0.01,*** p≤0.001, **** p≤0.0001.'
    }
    
    fig.text(0.5, -0.02, caption_text[metric], 
             ha='center', va='top', fontsize=10, 
             wrap=True, transform=fig.transFigure)
    
    plt.subplots_adjust(bottom=0.08)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    filename = os.path.join(OUTPUT_DIR, f'{output_prefix}_figure1_boxplot_{metric}')
    plt.savefig(f'{filename}.pdf', bbox_inches='tight')
# %%
    # plt.close()

    
# ============================================================================
#                           STATISTICS TABLE
# ============================================================================

def generate_statistics(penalized_data, organ_list, organ_display, metric, output_prefix):
    """Generate comprehensive statistics table in publication format - organs as columns"""
    print("\n" + "="*80)
    print(f"GENERATING STATISTICS TABLE (Reference: {REFERENCE_MODEL})")
    print("="*80)
    
    n_comparisons = len(MODEL_ORDER) - 1
    
    results_list = []
    
    # Each MODEL gets a row
    for model_name in MODEL_ORDER:
        row = {'Model': MODEL_DISPLAY[MODEL_ORDER.index(model_name)]}
        
        # Each ORGAN gets a column
        for organ, display_name in zip(organ_list, organ_display):
            key = (model_name, organ)
            if key in penalized_data:
                values = penalized_data[key][metric].values
                mean, std, n = compute_stats(values)
                
                # Format: mean ± std
                if metric in ['dice', 'sdsc']:
                    row[display_name] = f"{mean:.2f} ± {std:.2f}"
                else:  # hd95
                    row[display_name] = f"{mean:.1f} ± {std:.1f}"
            else:
                row[display_name] = "N/A"
        
        results_list.append(row)
        
        # Add p-value row right after each model (except reference)
        if model_name != REFERENCE_MODEL:
            pval_row = {'Model': '  p-value'}  # Indented to show it belongs to model above
            
            for organ, display_name in zip(organ_list, organ_display):
                ref_key = (REFERENCE_MODEL, organ)
                model_key = (model_name, organ)
                
                if ref_key in penalized_data and model_key in penalized_data:
                    ref_values = penalized_data[ref_key][metric].values
                    model_values = penalized_data[model_key][metric].values
                    
                    p_raw = wilcoxon_paired_test(penalized_data[ref_key], penalized_data[model_key], metric)
                    p_corr = apply_bonferroni(p_raw, n_comparisons)
                    pval_row[display_name] = format_pvalue(p_corr)
                else:
                    pval_row[display_name] = "N/A"
            
            results_list.append(pval_row)
            
    
    # Create DataFrame
    df = pd.DataFrame(results_list)
    
    # Reorder columns: Model first, then organs in display order
    column_order = ['Model'] + organ_display
    df = df[[c for c in column_order if c in df.columns]]
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Print to console
    print("\n")
    print(f"TABLE: {metric.upper()} Statistics (Reference: {REFERENCE_MODEL})")
    print("\n")
    print(df.to_string(index=False))
    print("\n")
    print("Notes:")
    print(f"  - Reference model: {REFERENCE_MODEL}")
    print(f"  - P-values: Bonferroni-corrected (×{n_comparisons})")
    print(f"  - Cases excluded if ALL models have NaN (no valid detection)")
    print(f"  - Failed detections penalized: DSC=0, HD95=50, sDSC=0.1")
    print("\n")
    
    return df

# ============================================================================
#                           MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate figures with penalization for failed segmentations'
    )
    parser.add_argument('--directory', type=str, default=str(RESULTS_ROOT / 'xcelrecords_breast'),
                       help='Data directory (xcelrecords_oar or xcelrecords_breast)')
    parser.add_argument('--metric', type=str, default='all',
                       choices=['hd95', 'dice', 'sdsc', 'volume_ratio', 'all'],
                       help='Metric to analyze (default: hd95, use "all" for all three)')
    parser.add_argument('--organs', type=str, nargs='+',
                       help='Custom organ list (optional)')
    parser.add_argument('--bracket-mode', type=str, default='simple',
                       choices=['tiered', 'simple', 'none'],
                       help="Bracket display: tiered (*/**/***), simple (* only), or none.")
    parser.add_argument('--sentinel-mode', type=str, default='paper',
                       choices=['paper', 'reviewer', 'exclude'],
                       help="Sentinel mode for failed segs: paper (HD95=50,VR=10), "
                            "reviewer (HD95/VR = per-organ worst observed), "
                            "exclude (drop cases where any model failed).")

    args = parser.parse_args()

    # Apply bracket mode globally before any plot generation
    global BRACKET_MODE
    BRACKET_MODE = args.bracket_mode

    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' not found")
        sys.exit(1)
    
    basename = os.path.basename(args.directory)
    
    if basename in ORGAN_CONFIG:
        config = ORGAN_CONFIG[basename]
        organ_list = config['organs']
        organ_display = config['display']
        title = config['title']
    elif args.organs:
        organ_list = args.organs
        organ_display = [o.upper() for o in args.organs]
        title = 'Structures'
    else:
        print(f"Error: Unknown directory. Please specify --organs")
        sys.exit(1)
    
    metrics = ['hd95', 'dice', 'sdsc', 'volume_ratio'] if args.metric == 'all' else [args.metric]
    dataset_name = os.path.basename(args.directory).replace('xcelrecords_', '')
    
    print("\n" + "="*80)
    print("CARDIAC/BREAST SEGMENTATION WITH SMART PENALIZATION")
    print("="*80)
    print(f"Directory: {args.directory}")
    print(f"Output prefix: {dataset_name}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Organs: {', '.join(organ_display)}")
    print(f"Metrics: {', '.join([m.upper() for m in metrics])}")
    print(f"Models: {len(MODEL_ORDER)}")
    print(f"Reference Model: {REFERENCE_MODEL}")
    print(f"\nSentinel mode: {args.sentinel_mode}")
    if args.sentinel_mode == 'exclude':
        print("  -> Cases where ANY model failed are dropped from paired tests")
    else:
        print(f"  -> Sentinel template: {SENTINEL_MODES[args.sentinel_mode]}")
    print("="*80)

    # Update output prefix with sentinel mode for traceability
    dataset_name = f"{dataset_name}_{args.sentinel_mode}"

    for metric in metrics:
        print(f"\n{'#'*80}")
        print(f"# PROCESSING METRIC: {metric.upper()}")
        print(f"{'#'*80}")

        penalized_data, organ_valid_cases, raw_data, organ_sentinels = load_and_penalize_data(
            args.directory, organ_list, metric, sentinel_mode=args.sentinel_mode
        )

        # Failure-rate report based on raw (pre-penalization) data
        failure_rates = compute_failure_rates(raw_data, organ_list, metric)
        print_failure_table(failure_rates, organ_list, organ_display, metric)

        generate_boxplot(penalized_data, organ_list, organ_display, title, metric, dataset_name)
        generate_statistics(penalized_data, organ_list, organ_display, metric, dataset_name)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in '{OUTPUT_DIR}/':")
    print(f"  - {dataset_name}_figure1_boxplot_*.pdf")
    print(f"  - {dataset_name}_table_*.csv/tex/docx")
    print(f"\nReference model: {REFERENCE_MODEL}")
    print("Cases with ALL NaN excluded; others penalized if missing detection")
    print("="*80)

if __name__ == '__main__':
    main()