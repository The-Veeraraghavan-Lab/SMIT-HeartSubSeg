import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
from pathlib import Path
ALLDATASETS_ROOT = Path(__file__).resolve().parents[2] / 'data' / 'AllDatasets'
import nibabel as nib
from matplotlib.colors import ListedColormap
import seaborn as sns

def compute_dvh_multiclass(segmentation_map, dose_map, num_classes, num_bins=1000):
    dvh_data = {}
    for class_label in range(1, num_classes + 1):  # Exclude background (label 0)
        dose_values = dose_map[segmentation_map == class_label]
        if dose_values.size == 0:
            continue
        hist, bin_edges = np.histogram(dose_values, bins=num_bins, range=(0, np.max(dose_map)))
        cumulative_volumes = np.cumsum(hist[::-1])[::-1] / len(dose_values)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        dvh_data[class_label] = (bin_centers, cumulative_volumes)
    return dvh_data

def plot_dvh_comparison_two_groups(dvh_data_predicted, dvh_data_truth, class_labels, title1, title2, name):
    idxx = 0
    plt.rcParams.update({'font.size': 12})
    
    # Create figure with white background
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), facecolor='white')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    
    # Use seaborn colorblind palette
    sns.set_palette("colorblind")
    colors = sns.color_palette("colorblind", n_colors=10)
    
    # First plot (Classes 1-5) - Great vessels
    for idx, class_label in enumerate(range(1, 6)):
        if class_label in dvh_data_predicted:
            bin_centers_pred, cumulative_vol_pred = dvh_data_predicted[class_label]
            ax1.plot(bin_centers_pred, cumulative_vol_pred, 
                    label=f"{class_labels[class_label - 1]}", 
                    linestyle="-", color=colors[idxx], linewidth=2)
        if class_label in dvh_data_truth:
            bin_centers_truth, cumulative_vol_truth = dvh_data_truth[class_label]
            ax1.plot(bin_centers_truth, cumulative_vol_truth, 
                    linestyle="--", color=colors[idxx], linewidth=1.5, alpha=0.7)
            idxx = idxx + 1
    
    ax1.set_xlabel("Dose (Gy)", fontsize=14)
    ax1.set_ylabel("Relative volume", fontsize=14)
    ax1.tick_params(axis='both', labelsize=12)
    # ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)  # REMOVED per IEEE TMI
    ax1.set_title(title1, fontsize=14)
    ax1.legend(fontsize=10, frameon=True, fancybox=False)
    
    # Second plot (Classes 6-9) - Chambers
    for idx, class_label in enumerate(range(6, 10)):
        if class_label in dvh_data_predicted:
            bin_centers_pred, cumulative_vol_pred = dvh_data_predicted[class_label]
            ax2.plot(bin_centers_pred, cumulative_vol_pred, 
                    label=f"{class_labels[class_label - 1]}", 
                    linestyle="-", color=colors[idx], linewidth=2)
        if class_label in dvh_data_truth:
            bin_centers_truth, cumulative_vol_truth = dvh_data_truth[class_label]
            ax2.plot(bin_centers_truth, cumulative_vol_truth, 
                    linestyle="--", color=colors[idx], linewidth=1.5, alpha=0.7)
    
    ax2.set_xlabel("Dose (Gy)", fontsize=14)
    ax2.set_ylabel("Relative volume", fontsize=14)
    ax2.tick_params(axis='both', labelsize=12)
    # ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)  # REMOVED per IEEE TMI
    ax2.set_title(title2, fontsize=14)
    ax2.legend(fontsize=10, frameon=True, fancybox=False)
    
    plt.tight_layout()
    
    # Add caption below - LEFT ALIGNED
    caption_text = ('DVH: Solid lines: AI-segmentation; dashed lines: manual delineations.')
    
    fig.text(0.1, -0.02, caption_text,
             ha='left', va='top',
             fontsize=10,
             wrap=True,
             transform=fig.transFigure)
    
    plt.subplots_adjust(bottom=0.15)
    
    plt.savefig(RESULTS_ROOT / 'tablesandfigures' / '{}_dvh_comparison.pdf'.format(name.replace(".nii.gz","")),
                bbox_inches='tight',
                pad_inches=0.1, dpi=600)
    # plt.close()

# Example Usage
if __name__ == "__main__":
    num_classes = 9  # Now considering all 9 classes
    folder_of_i = str(RESULTS_ROOT / 'model_outputs_breast' / 'run1_plus_cnc64_bnorm')
    setting = 'run1_plus_cnc64_bnorm'
    
    name = 'XXXX.nii.gz' # example case

    ai_seg = osp.join(folder_of_i, name)
    ai_seg = nib.load(ai_seg).get_fdata()
    ai_seg[ai_seg == 3] = 0
    
    gt_folder = str(ALLDATASETS_ROOT / 'Breast66')
    gt_seg = osp.join(gt_folder, 'label_plus', name.replace(".nii.gz","_label.nii.gz"))
    gt_seg = nib.load(gt_seg).get_fdata()
    gt_seg[gt_seg == 3] = 0
    
    dose_map = osp.join(gt_folder, 'dose', name.replace(".nii.gz", "_dose.nii.gz"))
    dose_map = nib.load(dose_map).get_fdata()
    
    dvh_data_predicted = compute_dvh_multiclass(ai_seg, dose_map, num_classes)
    dvh_data_truth = compute_dvh_multiclass(gt_seg, dose_map, num_classes)
    
    class_labels = ["AA", "PA", "PV", "SVC", "IVC", "RA", "RV", "LA", "LV"]
    
    plot_dvh_comparison_two_groups(dvh_data_predicted, dvh_data_truth,
                                   class_labels, title1="Great vessels",
                                   title2="Chambers", name=name)