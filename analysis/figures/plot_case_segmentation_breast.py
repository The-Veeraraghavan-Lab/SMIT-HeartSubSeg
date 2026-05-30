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
gt_label_dir = str(ALLDATASETS_ROOT / 'Breast66' / 'label_plus')


# ai_label_dir = 'model_outputs/b66_onlysubst'; setting = 'onlysubst'
# ai_label_dir = 'model_outputs/b66_onlysubst_onlyc'; setting = 'onlysubstc'
# ai_label_dir = 'model_outputs/b66_onlysubst_cnc64'; setting = 'onlysubstc64'
# ai_label_dir = 'segmentations_totalseg_b66'; setting = 'totalseg'
# ai_label_dir = str(RESULTS_ROOT / 'model_outputs_breast' / 'run1_plus_cnc64_frozen'); setting = 'run1_plus_cnc64_frozen'
ai_label_dir = str(RESULTS_ROOT / 'model_outputs_breast' / 'nnunet_lung_cnc64'); setting = 'nnunet'

filename = 'XXXX.nii.gz'
lim_x1, lim_x2, lim_y1, lim_y2 = 80, 240, 110, 370
z_slice = 69
annotations = {
    "AA": (140,35), #1
    # "PA": (150, 70), #2
    "SVC": (125, 70), #4
    # "IVC": (88, 90), #5
    "RA": (123, 45), #6
    "RV": (178, 20), #7
    "LA": (192, 70), #8
    "LV": (200, 30)  #9
}

# filename = 'XXXX.nii.gz'
# lim_x1, lim_x2, lim_y1, lim_y2 = 200, 350, 140, 360
# z_slice = 78
# annotations = {
#     "AA": (136,78), #1
#     # "PA": (180, 80), #2
#     # "SVC": (114,110), #4
#     "IVC": (90, 65), #5
#     "RA": (98, 45), #6
#     "RV": (155, 25), #7
# #     "LA": (192, 70), #8
#     "LV": (175, 35)  #9
# }


# # Load NIfTI files
image_path = osp.join(image_dir, filename) # Replace with your image file path

gt_label_path = osp.join(gt_label_dir, filename.replace(".nii", "_label.nii"))
pred_label_path = osp.join(ai_label_dir, filename)

# Load data
image = nib.load(image_path).get_fdata()

gt_label = nib.load(gt_label_path).get_fdata()
if setting == 'totalseg':
    import nibabel as nib
    
    # Label mapping
    LABEL_MAP = {
        "aorta": 1,
        "pulmonary_artery": 2,
        "pulmonary_vein": 3,
        "superior_vena_cava": 4,
        "inferior_vena_cava": 5,
        "heart_atrium_right": 6,
        "heart_ventricle_right": 7,
        "heart_atrium_left": 8,
        "heart_ventricle_left": 9
    }
    
    def combine_subject_segmentations(subject_folder):
        combined = None
        for filename in os.listdir(subject_folder):
            if not filename.endswith(".nii.gz"):
                continue
            
            key = filename.replace(".nii.gz", "")
            if key not in LABEL_MAP:
                continue
    
            label_value = LABEL_MAP[key]
            filepath = os.path.join(subject_folder, filename)
            img = nib.load(filepath)
            data = img.get_fdata()
    
            if combined is None:
                combined = np.zeros_like(data, dtype=np.uint8)
    
            combined[data > 0] = label_value  # overwrite where structure exists
    
        return combined  # This is your labeled NumPy array
    
    # Example usage
    pred_label = combine_subject_segmentations(osp.join(ai_label_dir,
                                                        filename.replace(".nii.gz","")))
    pred_label = pred_label[:,:,::-1]
    pred_label = np.rot90(pred_label, k=1, axes=(0, 1))
else:        
    pred_label = nib.load(pred_label_path).get_fdata()
# pred_label[pred_label>5] = 0


# Extract the corresponding slices
image_slice = image[lim_x1:lim_x2, lim_y1:lim_y2, z_slice]

gt_label_slice = gt_label[lim_x1:lim_x2, lim_y1:lim_y2, z_slice]
pred_label_slice = pred_label[lim_x1:lim_x2, lim_y1:lim_y2, z_slice]

# image_slice = np.fliplr(np.flipud(image_slice))
# gt_label_slice = np.fliplr(np.flipud(gt_label_slice))
# pred_label_slice = np.fliplr(np.flipud(pred_label_slice))

# image_slice = np.flipud(image_slice)
# gt_label_slice = np.flipud(gt_label_slice)
# pred_label_slice = np.flipud(pred_label_slice)


# # Normalize image for better visualization
image_slice = np.clip(image_slice, -400, 400)
image_slice = (image_slice - (-400)) / (400 - (-400))


# Plot the image
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(image_slice, cmap="gray", interpolation="none")

# Get unique labels (excluding background, assuming background is 0)
unique_labels = np.unique(gt_label_slice)
unique_labels = unique_labels[unique_labels > 0]  # Exclude background if necessary

# Generate a color palette with the same number of unique labels
color_palette = [(255,0,0), (0,255,0), (0,0,255), (255,255,0),
                 (0,255,255), (255,0,255), (255,239,213),
                 (0,0,190),(205,133,63)]
color_palette = [(r/255, g/255, b/255) for r, g, b in color_palette]


# Create a mapping of label values to specific colors
label_color_map = {int(label): color_palette[int(label) - 1] for label in unique_labels if label - 1 < len(color_palette)}


# Plot contours manually to ensure correct color mapping
for label in unique_labels:
    ax.contour(
        gt_label_slice == label,  # Binary mask for each label
        levels=[0.5],  # Single contour level
        colors=[label_color_map[label]],  # Assign the correct color
        linestyles="dotted",
        linewidths=4,
    )
    
# Overlay AI-segmented annotations as solid lines
for label in unique_labels:
    ax.contour(
        pred_label_slice == label,
        levels=[0.5], # Exclude background
        colors=[label_color_map[label]],  # Assign the correct color
        linestyles="solid",
        linewidths=3,
    )

# for label, coord in annotations.items():
#     ax.text(coord[0],
#             coord[1],
#             label,
#             color="white",
#             fontsize=20,
#             weight="bold",
#             ha="center")


ax.axis("off")
plt.tight_layout()
plt.show()
# plt.savefig('viz_images/breast_{}_img_{}.png'.format(filename.replace(".nii.gz",""),setting),
#             bbox_inches = 'tight', pad_inches = 0, dpi = 600)
