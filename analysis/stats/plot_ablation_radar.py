"""
CARDIAC SEGMENTATION - RADAR CHART ANALYSIS
Shows performance profiles across organs for different SMIT training conditions
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
#                           CONFIGURATION
# ============================================================================

sns.set_palette("colorblind")
colorblind_colors = sns.color_palette("colorblind", n_colors=4)

MODELS = {
    'SMIT_Balanced': 'run1_plus_cnc64_bnorm',
    'Frozen backbone': 'run1_plus_cnc64_frozen',
    'CECT-only': 'run1_plus_onlycontrast_bnorm',
    'NCCT-only': 'run1_plus_onlynoncontrast_inorm',
}

BASELINE_MODEL = 'SMIT_Balanced'

# All models including baseline
MODEL_ORDER = ['SMIT_Balanced', 'Frozen backbone', 'CECT-only', 'NCCT-only']
MODEL_DISPLAY = {
    'SMIT_Balanced': 'SMIT-Balanced',
    'Frozen backbone': 'Frozen backbone',
    'CECT-only': 'CECT-Only',
    'NCCT-only': 'NCCT-only'
}

# Use seaborn colorblind palette for all models
COLORS = {
    'SMIT_Balanced': colorblind_colors[0],
    'Frozen backbone': colorblind_colors[1],
    'CECT-only': colorblind_colors[2],
    'NCCT-only': colorblind_colors[3],
}

LINE_STYLES = {
    'SMIT_Balanced': '-',
    'Frozen backbone': '--',
    'CECT-only': '-.',
    'NCCT-only': ':'
}

LINE_WIDTHS = {
    'SMIT_Balanced': 3,
    'Frozen backbone': 2,
    'CECT-only': 2,
    'NCCT-only': 2
}

PENALTY_VALUES = {
    'dice': 0.0,
    'hd95': 50.0,
    'sdsc': 0.1,
    'volume_ratio': 2
}

ORGAN_CONFIG = {
    'xcelrecords_oar': {
        'organs': ['aorta', 'pa', 'svc', 'ivc', 'ra', 'rv', 'la', 'lv'],
        'display': ['AA', 'PA', 'SVC', 'IVC', 'RA', 'RV', 'LA', 'LV'],
    },
    'xcelrecords_breast': {
        'organs': ['aorta', 'pa', 'svc', 'ivc', 'ra', 'rv', 'la', 'lv'],
        'display': ['AA', 'PA', 'SVC', 'IVC', 'RA', 'RV', 'LA', 'LV'],
    }
}

RESULTS_ROOT = Path(__file__).resolve().parents[1] / 'results'
OUTPUT_DIR = RESULTS_ROOT / 'tablesandfigures'

# ============================================================================
#                           DATA LOADING
# ============================================================================

def load_and_penalize_data(directory, organ_list, metric):
    """Load all data and apply penalization for missing detections."""
    print(f"\nLoading data for {metric.upper()}...")
    
    raw_data = {}
    for model_name, prefix in MODELS.items():
        for organ in organ_list:
            filepath = os.path.join(directory, f"{prefix}_{organ}.csv")
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, dtype={'name': str})
                    raw_data[(model_name, organ)] = df
                except Exception as e:
                    print(f"  Warning: {filepath}: {e}")
    
    # Find valid cases per organ
    organ_valid_cases = {}
    for organ in organ_list:
        case_validity = {}
        for model_name in MODELS.keys():
            key = (model_name, organ)
            if key in raw_data:
                df = raw_data[key]
                if metric not in df.columns:
                    continue
                for _, row in df.iterrows():
                    case_id = str(row['name'])
                    if case_id not in case_validity:
                        case_validity[case_id] = {}
                    case_validity[case_id][model_name] = pd.notna(row[metric])
        
        valid_cases = [cid for cid, mv in case_validity.items() if any(mv.values())]
        organ_valid_cases[organ] = sorted(valid_cases)
    
    # Apply penalization
    penalized_data = {}
    for organ in organ_list:
        valid_cases = set(organ_valid_cases[organ])
        for model_name, prefix in MODELS.items():
            key = (model_name, organ)
            
            if key in raw_data:
                df = raw_data[key].copy()
                if metric not in df.columns:
                    rows = [{'name': cid, metric: PENALTY_VALUES[metric]} 
                           for cid in valid_cases]
                    penalized_data[key] = pd.DataFrame(rows)
                    continue
                
                df = df[df['name'].isin(valid_cases)].copy()
                existing_cases = set(df['name'].tolist())
                missing_cases = valid_cases - existing_cases
                
                df[metric] = df[metric].fillna(PENALTY_VALUES[metric])
                
                for case_id in missing_cases:
                    new_row = {'name': case_id, metric: PENALTY_VALUES[metric]}
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                
                penalized_data[key] = df
            else:
                rows = [{'name': cid, metric: PENALTY_VALUES[metric]} 
                       for cid in valid_cases]
                penalized_data[key] = pd.DataFrame(rows)
    
    return penalized_data

# ============================================================================
#                           RADAR CHART
# ============================================================================

def generate_radar_chart(penalized_data, organ_list, organ_display, metric, output_prefix):
    """Radar chart showing performance profile across organs"""
    print(f"\nGenerating radar chart for {metric.upper()}...")
    
    # Number of variables
    num_vars = len(organ_list)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    
    # ADD THESE TWO LINES TO PUT AA AT THE TOP
    ax.set_theta_offset(np.pi / 2)  # Rotate so first label is at top
    ax.set_theta_direction(-1)  # Go clockwise
    
    # Collect all values to determine appropriate y-axis limits
    all_values = []
    for model_name in MODEL_ORDER:
        for organ in organ_list:
            key = (model_name, organ)
            if key in penalized_data:
                all_values.extend(penalized_data[key][metric].values)
    
    # Set appropriate y-axis range
    if metric in ['dice', 'sdsc']:
        y_min = max(0, np.min(all_values) - 0.05)
        y_max = min(1.0, np.max(all_values) + 0.05)
    elif metric == 'hd95':
        y_min = 0
        y_max = 22
    else:
        y_min = max(0, np.min(all_values) - 0.1)
        y_max = np.max(all_values) + 0.1
    
    # Plot each model
    for model_name in MODEL_ORDER:
        model_values = []
        for organ in organ_list:
            key = (model_name, organ)
            if key in penalized_data:
                model_values.append(np.mean(penalized_data[key][metric].values))
            else:
                model_values.append(np.nan)
        
        model_values += model_values[:1]  # Complete the circle
        
        # Determine marker size and alpha
        if model_name == BASELINE_MODEL:
            markersize = 10
            alpha = 1.0
            zorder = 10
        else:
            markersize = 7
            alpha = 0.8
            zorder = 5
        
        ax.plot(angles, model_values, 
               marker='o',
               linestyle=LINE_STYLES[model_name],
               linewidth=LINE_WIDTHS[model_name],
               label=MODEL_DISPLAY[model_name],
               color=COLORS[model_name], 
               alpha=alpha, 
               markersize=markersize,
               zorder=zorder)
        
        # Add fill for baseline
        if model_name == BASELINE_MODEL:
            ax.fill(angles, model_values, alpha=0.15, color=COLORS[model_name])
    
    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(organ_display, fontsize=14)
    ax.set_ylim([y_min, y_max])
    
    # Add value labels on radial axis
    ax.set_yticks(np.linspace(y_min, y_max, 5))
    if metric in ['dice', 'sdsc']:
        ax.set_yticklabels([f'{val:.2f}' for val in np.linspace(y_min, y_max, 5)], 
                          fontsize=14)
    else:
        ax.set_yticklabels([f'{val:.1f}' for val in np.linspace(y_min, y_max, 5)], 
                          fontsize=14)
    
    # Grid styling
    ax.grid(True, alpha=0.1, linestyle='--', linewidth=0.5)
    ax.spines['polar'].set_color('gray')
    ax.spines['polar'].set_linewidth(1)
    
    # Legend
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.2), 
                       ncols = 2, fontsize=10, frameon=True,
                       fancybox=False)
    legend.get_frame().set_alpha(0.9)

    plt.tight_layout()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, f'{output_prefix}_radar_{metric}')
    plt.savefig(f'{filename}.pdf', bbox_inches='tight', dpi=300)
    print(f"  ✓ Saved: {filename}.pdf and .png")
    # plt.close()

# ============================================================================
#                           MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate radar chart analysis for SMIT training conditions'
    )
    parser.add_argument('--directory', type=str, default=str(RESULTS_ROOT / 'xcelrecords_breast'),
                       help='Data directory')
    parser.add_argument('--metric', type=str, default='hd95',
                       choices=['hd95', 'dice', 'sdsc', 'volume_ratio', 'all'],
                       help='Metric to analyze')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' not found")
        sys.exit(1)
    
    basename = os.path.basename(args.directory)
    config = ORGAN_CONFIG[basename]
    organ_list = config['organs']
    organ_display = config['display']
    
    metrics = ['dice', 'hd95', 'sdsc'] if args.metric == 'all' else [args.metric]
    dataset_name = basename.replace('xcelrecords_', '')
    
    print("\n" + "="*80)
    print("RADAR CHART ANALYSIS - SMIT TRAINING CONDITIONS")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Models: {', '.join([MODEL_DISPLAY[m] for m in MODEL_ORDER])}")
    print(f"Metrics: {', '.join([m.upper() for m in metrics])}")
    print("="*80)
    
    for metric in metrics:
        print(f"\n{'#'*80}")
        print(f"# METRIC: {metric.upper()}")
        print(f"{'#'*80}")
        
        penalized_data = load_and_penalize_data(args.directory, organ_list, metric)
        generate_radar_chart(penalized_data, organ_list, organ_display, metric, dataset_name)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print(f"Output directory: {OUTPUT_DIR}/")
    print("="*80)

if __name__ == '__main__':
    main()