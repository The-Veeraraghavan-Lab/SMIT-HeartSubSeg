import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from glob import glob

def plot_dose_overlay(image_path, dose_path, output_path):
    # Load volumes
    img = nib.load(image_path).get_fdata()
    dose = nib.load(dose_path).get_fdata()
    
    # Find slice with max dose (most expressive)
    dose_per_slice = dose.sum(axis=(0, 1))  # sum over x, y
    max_slice = np.argmax(dose_per_slice)
    
    # Extract slices
    img_slice = img[:, :, max_slice]
    dose_slice = dose[:, :, max_slice]
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image only
    axes[0].imshow(img_slice.T, cmap='gray', origin='lower')
    axes[0].set_title(f'CT (slice {max_slice})')
    axes[0].axis('off')
    
    # Dose only
    axes[1].imshow(dose_slice.T, cmap='jet', origin='lower')
    axes[1].set_title(f'Dose (slice {max_slice})')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(img_slice.T, cmap='gray', origin='lower')
    dose_masked = np.ma.masked_where(dose_slice < dose_slice.max() * 0.1, dose_slice)  # threshold 10%
    axes[2].imshow(dose_masked.T, cmap='jet', alpha=0.5, origin='lower')
    axes[2].set_title(f'Overlay (slice {max_slice})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


ALLDATASETS_ROOT = Path(__file__).resolve().parents[2] / 'data' / 'AllDatasets'

# Usage
image_dir = str(ALLDATASETS_ROOT / 'Breast66' / 'imgs')
dose_dir = str(ALLDATASETS_ROOT / 'Breast66' / 'dose')
output_dir = str(ALLDATASETS_ROOT / 'Breast66' / 'dose_check')
os.makedirs(output_dir, exist_ok=True)

image_files = sorted(glob(os.path.join(image_dir, '*.nii.gz')))

for image_path in image_files:
    name = os.path.basename(image_path).replace('.nii.gz', '')
    dose_path = os.path.join(dose_dir, f'{name}_dose.nii.gz')
    
    if os.path.exists(dose_path):
        output_path = os.path.join(output_dir, f'{name}_overlay.png')
        plot_dose_overlay(image_path, dose_path, output_path)
    else:
        print(f"Dose not found: {dose_path}")