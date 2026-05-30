import os
import os.path as osp
from pathlib import Path
ALLDATASETS_ROOT = Path(__file__).resolve().parents[2] / 'data' / 'AllDatasets'
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


image_dir = str(ALLDATASETS_ROOT / 'Breast66' / 'imgs')
dose_dir = str(ALLDATASETS_ROOT / 'Breast66' / 'dose')
gt_label_dir = str(ALLDATASETS_ROOT / 'Breast66' / 'label_plus')

RESULTS_ROOT = Path(__file__).resolve().parents[1] / 'results'

ai_label_dir = str(RESULTS_ROOT / 'model_outputs_breast' / 'run1_plus_cnc64_bnorm'); setting = 'run1_plus_cnc64_bnorm'

# filename = 'XXXX.nii.gz'
# lim_x1, lim_x2, lim_y1, lim_y2 = 150, 380, 100, 410
# z_slice = 92
# annotations = {
#     "AA": (135, 78),
#     "PA": (160, 75),
#     # "PV": (200, 80),
#     "SVC": (120, 95),
#     # "IVC": (170, 140),
# }

# filename = 'XXXX.nii.gz'
# lim_x1, lim_x2, lim_y1, lim_y2 = 130, 360, 90, 410
# z_slice = 53
# annotations = {
#     "AA": (160, 80),
#     "PA": (190, 70),
#     # "PV": (150, 30),
#     "SVC": (135,103),

#     # "IVC": (170, 140),
# }

# filename = 'XXXX.nii.gz'
# lim_x1, lim_x2, lim_y1, lim_y2 = 150, 350, 110, 400
# z_slice = 80
# annotations = {
#     "AA": (130, 55), #1
#     "PA": (180, 48), #2
#     "SVC": (114,78), #4
#     "LA": (134, 115), #8
#     "LV": (195, 85), #8
# }


filename = 'XXXX.nii.gz'
lim_x1, lim_x2, lim_y1, lim_y2 = 100, 360, 60, 360
z_slice = 69
annotations = {
    "AA": (80, 65), #1
    "RA": (150, 95), #2
    "RV": (150,155), #4
    "LA": (110, 85), #8
    "LV": (70, 99), #8
}


# Load NIfTI files
image_path = osp.join(image_dir, filename) # Replace with your image file path
dose_path = osp.join(dose_dir, filename.replace(".nii", "_dose.nii"))

gt_label_path = osp.join(gt_label_dir, filename.replace(".nii", "_label.nii"))
pred_label_path = osp.join(ai_label_dir, filename)



# Load data
image = nib.load(image_path).get_fdata()
dose = nib.load(dose_path).get_fdata()

gt_label = nib.load(gt_label_path).get_fdata()
pred_label = nib.load(pred_label_path).get_fdata()

# image = np.fliplr(image)
# dose = np.fliplr(dose)
# gt_label = np.fliplr(gt_label)
# pred_label = np.fliplr(pred_label)



# Extract the corresponding slices
image_slice = image[lim_x1:lim_x2, lim_y1:lim_y2, z_slice]
dose_slice = dose[lim_x1:lim_x2, lim_y1:lim_y2, z_slice]

gt_label_slice = gt_label[lim_x1:lim_x2, lim_y1:lim_y2, z_slice]
gt_label_slice[gt_label_slice==3] = 0
pred_label_slice = pred_label[lim_x1:lim_x2, lim_y1:lim_y2, z_slice]
pred_label_slice[pred_label_slice==3] = 0


# # Normalize image for better visualization
image_slice = np.clip(image_slice, -400, 400)
image_slice = (image_slice - (-400)) / (400 - (-400))

# # Mask dose values below 5
dose_masked = np.ma.masked_less(dose_slice, 10)  # Mask values less than 5


# Define a colormap for the dose map
dose_colors = ["red", "yellow", "lime", "cyan", "blue", "magenta"]
cmap = ListedColormap(dose_colors)

# Get min and max of dose_masked
min_dose = np.nanmin(dose_masked)  # Ignore NaN values
max_dose = np.nanmax(dose)

# Generate evenly spaced dose levels
dose_levels = np.linspace(min_dose, max_dose, len(dose_colors) + 1)
dose_levels = np.round(dose_levels).astype(int)  # Round to the nearest integer and convert to int


# Plot the image
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(image_slice, cmap="gray", interpolation="none")

# Overlay dose map with transparency
dose_overlay = ax.imshow(dose_masked,
                         cmap=cmap,
                         interpolation="none",
                         alpha=0.4,
                         vmin=min(dose_levels),
                         vmax=max(dose_levels)
                         )

if setting == 'run1_plus_cnc64_bnorm':
    # Adjust the colorbar to match the size of the image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.02)  # Reduce padding between the image and the colorbar
    
    cbar = plt.colorbar(dose_overlay, cax = cax)
    cbar.set_label("Dose Levels (Gy)", fontsize = 20)
    cbar.set_ticks(dose_levels)
    cbar.ax.tick_params(labelsize=20)

# Overlay GT segmentations as dotted lines
gt_contours = ax.contour(
    gt_label_slice,
    levels=np.unique(gt_label_slice)[0:],  # Exclude background
    colors="white",
    linestyles="dotted",
    linewidths=3,
)

# Overlay AI-segmented annotations as solid lines
ai_contours = ax.contour(
    pred_label_slice,
    levels=np.unique(pred_label_slice)[0:],  # Exclude background
    colors="white",
    linestyles="solid",
    linewidths=2,
)

for label, coord in annotations.items():
    ax.text(coord[0],
            coord[1],
            label,
            color="white",
            fontsize=20,
            weight="bold",
            ha="center")

ax.axis("off")
plt.tight_layout()
# plt.show()
plt.savefig(RESULTS_ROOT / 'viz_images' / '{}_img_{}.pdf'.format(filename.replace(".nii.gz",""), setting),
            bbox_inches = 'tight', pad_inches = 0, dpi = 600)
