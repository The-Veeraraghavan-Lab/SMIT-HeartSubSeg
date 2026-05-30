import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from statannotations.Annotator import Annotator

ORGAN_NAMES = ['aorta', 'pa', 'svc', 'ivc', 'ra', 'rv', 'la', 'lv']
ORGAN_LABELS = ['AA', 'PA', 'SVC', 'IVC', 'RA', 'RV', 'LA', 'LV']
RESULTS_ROOT = Path(__file__).resolve().parents[1] / 'results'

MODEL_LABELS = {
    'run1_plus_cnc64_bnorm': 'SMIT-Balanced',
    'run1_plus_cnc64_frozen': 'SMIT-Frozen',
    'run1_plus_onlycontrast_bnorm': 'SMIT-OnlyC',
    'run1_plus_onlynoncontrast_inorm': 'SMIT-OnlyNC',
}


def metric_root(dataset):
    return RESULTS_ROOT / ('xcelrecords_oar' if dataset == 'lung' else 'xcelrecords_breast')


def read_metric_table(dataset, modelname, organ_name):
    return pd.read_csv(metric_root(dataset) / f'{modelname}_{organ_name}.csv', dtype={'name': str})


def save_current_figure(output_path, dpi=300):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f'Saved {output_path}')


def apply_annotations(ax, final_df, pairs, x, y, hue, order, hue_order, test):
    if not pairs:
        print('No valid pairs for statistical testing')
        return
    annotator = Annotator(
        ax,
        pairs,
        data=final_df,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
    )
    annotator.configure(
        test=test,
        text_format='star',
        show_test_name=False,
        line_offset_to_group=0.15,
        line_height=0.02 if test == 'Mann-Whitney' else 0.03,
        fontsize=10,
    )
    annotator.apply_and_annotate()


def run_boxplot_mode(args, metadata_file, metadata_column, group_map, dataset, output_stem, legend_title, ylim, test):
    info_df = pd.read_csv(metadata_file, dtype={'name': str})
    all_data = []
    for organ_name in ORGAN_NAMES:
        df = read_metric_table(dataset, args.modelname, organ_name)
        merged_df = pd.merge(df, info_df, on='name', how='inner')
        for raw_value, group_label in group_map.items():
            group_df = merged_df[merged_df[metadata_column] == raw_value].copy()
            group_df['group'] = group_label
            group_df['substructure'] = organ_name
            all_data.append(group_df)

    final_df = pd.concat(all_data, ignore_index=True)
    group_order = list(group_map.values())

    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(6, 6), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    sns.boxplot(
        data=final_df,
        x='substructure',
        y=args.metric,
        hue='group',
        order=ORGAN_NAMES,
        hue_order=group_order,
        dodge=True,
        ax=ax,
    )

    plt.xlabel('Substructure', fontsize=14)
    plt.ylabel(f'{args.metric.upper()} (mm)' if args.metric == 'hd95' else args.metric, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(ticks=range(len(ORGAN_LABELS)), labels=ORGAN_LABELS, fontsize=14)
    plt.ylim(*ylim)

    valid_pairs = []
    for organ in ORGAN_NAMES:
        organ_data = final_df[final_df['substructure'] == organ]
        values = []
        for group_label in group_order:
            data = organ_data[(organ_data['group'] == group_label) & (organ_data[args.metric].notna())][args.metric]
            values.append(data)
        if all(len(v) >= 2 for v in values):
            valid_pairs.append(((organ, group_order[0]), (organ, group_order[1])))

    apply_annotations(ax, final_df, valid_pairs, 'substructure', args.metric, 'group', ORGAN_NAMES, group_order, test)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title=legend_title, loc='upper right', fontsize=12, title_fontsize=12, frameon=True, fancybox=False)
    caption_text = 'HD95: Box: median and IQR. Significance: * p≤0.05, ** p≤0.01, *** p≤0.001, **** p≤0.0001.'
    ax.text(0, -0.15, caption_text, ha='left', va='top', fontsize=10, wrap=True, transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2)
    save_current_figure(args.output_dir / f'{output_stem}_{args.modelname}.pdf', dpi=args.dpi)
    plt.close()


def run_orientation_mode(args):
    position_df = pd.read_csv(args.metadata_dir / 'b66_split_maria.csv', dtype={'name': str})
    all_data = []
    for organ_name in ORGAN_NAMES:
        df = read_metric_table('breast', args.modelname, organ_name)
        df['name'] = df['name'].astype(str)
        df['substructure'] = organ_name
        all_data.append(df.merge(position_df, on='name', how='inner'))
    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.groupby(['substructure', 'name']).filter(lambda g: not g[args.metric].isna().any())
    available_organs = final_df['substructure'].unique().tolist()

    for organ in ORGAN_NAMES:
        if organ not in available_organs:
            for orientation in ['Prone', 'Supine']:
                final_df = pd.concat([
                    final_df,
                    pd.DataFrame({'substructure': [organ], 'orientation': [orientation], args.metric: [float('nan')]})
                ], ignore_index=True)

    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(6, 6), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    sns.boxplot(data=final_df, x='substructure', y=args.metric, hue='orientation', order=ORGAN_NAMES, dodge=True, ax=ax)
    plt.xticks(ticks=range(len(ORGAN_LABELS)), labels=ORGAN_LABELS, fontsize=14)
    plt.xlabel('Substructure', fontsize=16)
    plt.ylabel('HD95 (mm)', fontsize=14)
    plt.ylim(0, 50)

    pairs = []
    for organ in ORGAN_NAMES:
        if organ in available_organs:
            sub_df = final_df[final_df['substructure'] == organ]
            orientations = sub_df.dropna(subset=[args.metric])['orientation'].unique()
            if 'Prone' in orientations and 'Supine' in orientations:
                pairs.append(((organ, 'Prone'), (organ, 'Supine')))

    apply_annotations(ax, final_df, pairs, 'substructure', args.metric, 'orientation', ORGAN_NAMES, ['Prone', 'Supine'], 'Mann-Whitney')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Factors (orientation)', loc='upper right', fontsize=12, title_fontsize=12, frameon=True, fancybox=False)
    caption_text = 'HD95: Box: median and IQR. Significance: * p≤0.05, ** p≤0.01, *** p≤0.001, **** p≤0.0001.'
    ax.text(0, -0.15, caption_text, ha='left', va='top', fontsize=10, wrap=True, transform=ax.transAxes)
    plt.subplots_adjust(bottom=0.2)
    save_current_figure(args.output_dir / f'orientation_{args.modelname}.pdf', dpi=args.dpi)
    plt.close()


def run_age_bmi_mode(args):
    age_df = pd.read_csv(args.metadata_dir / 'N240_age.csv', dtype={'name': str})
    bmi_df = pd.read_csv(args.metadata_dir / 'N240_BMI.csv', dtype={'name': str})
    age_corr = []
    bmi_corr = []

    for organ_name in ORGAN_NAMES:
        scores = read_metric_table('lung', args.modelname, organ_name)
        age_merged = pd.merge(scores, age_df, on='name', how='inner').dropna(subset=[args.metric, 'age'])
        bmi_merged = pd.merge(scores, bmi_df, on='name', how='inner').dropna(subset=[args.metric, 'bmi'])
        age_corr.append(spearmanr(age_merged[args.metric], age_merged['age'])[0] if len(age_merged) else np.nan)
        bmi_corr.append(spearmanr(bmi_merged['bmi'], bmi_merged[args.metric])[0] if len(bmi_merged) else np.nan)

    angles = np.linspace(0, 2 * np.pi, len(ORGAN_LABELS), endpoint=False).tolist()
    angles += angles[:1]
    age_correlations = np.append(np.asarray(age_corr), age_corr[0])
    bmi_correlations = np.append(np.asarray(bmi_corr), bmi_corr[0])

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'}, facecolor='white')
    ax.set_facecolor('white')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    age_valid = ~np.isnan(age_correlations)
    bmi_valid = ~np.isnan(bmi_correlations)
    angles_arr = np.array(angles)

    ax.plot(angles_arr[age_valid], age_correlations[age_valid], label='Age', marker='o', linestyle='', markersize=6, linewidth=2)
    for i in range(len(age_correlations) - 1):
        if age_valid[i] and age_valid[i + 1]:
            ax.plot([angles_arr[i], angles_arr[i + 1]], [age_correlations[i], age_correlations[i + 1]], color='C0', linestyle='-', linewidth=2)
    if np.any(age_valid):
        ax.fill(angles_arr, np.ma.masked_where(~age_valid, age_correlations), alpha=0.25, color='C0')

    ax.plot(angles_arr[bmi_valid], bmi_correlations[bmi_valid], label='BMI', marker='s', linestyle='', markersize=6, linewidth=2)
    for i in range(len(bmi_correlations) - 1):
        if bmi_valid[i] and bmi_valid[i + 1]:
            ax.plot([angles_arr[i], angles_arr[i + 1]], [bmi_correlations[i], bmi_correlations[i + 1]], color='C1', linestyle='--', linewidth=2)
    if np.any(bmi_valid):
        ax.fill(angles_arr, np.ma.masked_where(~bmi_valid, bmi_correlations), alpha=0.15, color='C1')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(ORGAN_LABELS, fontsize=14)
    ax.set_ylim(-0.4, 0.4)
    ax.plot(np.linspace(0, 2 * np.pi, 100), np.zeros(100), color='gray', linestyle=':', linewidth=1.5, zorder=1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.12, 1.12), fontsize=12, title='Factors', title_fontsize=12, frameon=True, fancybox=False)
    ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.5)
    fig.text(0.1, 0.02, 'Radial axis shows Spearman correlation coefficient. Missing segments indicate structures not segmented by the model.', ha='left', va='bottom', fontsize=10, wrap=True, transform=fig.transFigure)
    plt.subplots_adjust(bottom=0.11)
    save_current_figure(args.output_dir / f'age_bmi_{args.modelname}.pdf', dpi=args.dpi)
    plt.close(fig)


def run_contrast_models_mode(args):
    models = {
        'SMIT_Balanced': 'run1_plus_cnc64_bnorm',
        'SMIT_Frozen': 'run1_plus_cnc64_frozen',
        'SMIT_OnlyC': 'run1_plus_onlycontrast_bnorm',
        'SMIT_OnlyNC': 'run1_plus_onlynoncontrast_inorm',
    }
    info_df = pd.read_csv(args.metadata_dir / 'N240_contrast.csv', dtype={'name': str})
    all_model_data = []
    for model_key, modelname in models.items():
        for organ_name in ORGAN_NAMES:
            organ_file = metric_root('lung') / f'{modelname}_{organ_name}.csv'
            if not organ_file.exists():
                continue
            df = pd.read_csv(organ_file, dtype={'name': str})
            merged_df = pd.merge(df, info_df, on='name', how='inner')
            merged_df['model'] = model_key
            merged_df['substructure'] = organ_name
            all_model_data.append(merged_df)
    final_df = pd.concat(all_model_data, ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    group_order = ['contrast', 'noncontrast']
    model_order = list(models.keys())

    for idx, model_key in enumerate(model_order):
        ax = axes[idx]
        model_data = final_df[final_df['model'] == model_key].copy()
        model_data['group'] = model_data['setting']
        sns.boxplot(data=model_data, x='substructure', y=args.metric, hue='group', order=ORGAN_NAMES, hue_order=group_order, dodge=True, ax=ax)
        ax.set_xlabel('Substructure' if idx >= 2 else '', fontsize=12)
        ax.set_ylabel('HD95 (mm)' if idx % 2 == 0 else '', fontsize=12)
        ax.set_title(MODEL_LABELS[modelname], fontsize=14, fontweight='bold')
        ax.set_xticklabels(ORGAN_LABELS, fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        valid_pairs = []
        for organ in ORGAN_NAMES:
            organ_data = model_data[model_data['substructure'] == organ]
            contrast_data = organ_data[(organ_data['group'] == 'contrast') & (organ_data[args.metric].notna())][args.metric]
            noncontrast_data = organ_data[(organ_data['group'] == 'noncontrast') & (organ_data[args.metric].notna())][args.metric]
            if len(contrast_data) >= 2 and len(noncontrast_data) >= 2:
                valid_pairs.append(((organ, 'contrast'), (organ, 'noncontrast')))
        apply_annotations(ax, model_data, valid_pairs, 'substructure', args.metric, 'group', ORGAN_NAMES, group_order, 'Mann-Whitney')
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, ['Contrast', 'Non-contrast'], loc='upper right', fontsize=10, frameon=True)
        else:
            ax.get_legend().remove()

    plt.tight_layout()
    save_current_figure(args.output_dir / 'contrast_all_models_faceted.pdf', dpi=args.dpi)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description='Generate subgroup robustness plots used in the paper.')
    parser.add_argument('--mode', required=True, choices=['contrast', 'sex', 'orientation', 'age_bmi', 'contrast_models'])
    parser.add_argument('--modelname', default='run1_plus_cnc64_bnorm')
    parser.add_argument('--metric', default='hd95')
    parser.add_argument('--metadata_dir', type=Path, default=Path(__file__).resolve().parents[1] / 'metadata')
    parser.add_argument('--output_dir', type=Path, default=RESULTS_ROOT / 'tablesandfigures')
    parser.add_argument('--dpi', type=int, default=300)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == 'contrast':
        run_boxplot_mode(args, args.metadata_dir / 'N240_contrast.csv', 'setting', {'contrast': 'Contrast', 'noncontrast': 'Non-contrast'}, 'lung', 'contrast', 'CT Acquisition', (0, 70), 'Mann-Whitney')
    elif args.mode == 'sex':
        run_boxplot_mode(args, args.metadata_dir / 'N240_gender.csv', 'sex', {0: 'Male', 1: 'Female'}, 'lung', 'sex', 'Factors (sex)', (0, 50), 'Mann-Whitney')
    elif args.mode == 'orientation':
        run_orientation_mode(args)
    elif args.mode == 'age_bmi':
        run_age_bmi_mode(args)
    elif args.mode == 'contrast_models':
        run_contrast_models_mode(args)


if __name__ == '__main__':
    main()
