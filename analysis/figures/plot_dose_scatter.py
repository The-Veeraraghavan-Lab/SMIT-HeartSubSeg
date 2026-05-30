from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

fontsize = 20

# Load the CSV files
df1 = pd.read_csv(RESULTS_ROOT / 'xcelrecords_dose' / 'oar' / 'gt.csv')
df2 = pd.read_csv(RESULTS_ROOT / 'xcelrecords_dose' / 'oar' / 'run1_plus_cnc64_bnorm.csv')

# Get colorblind palette
colors = sns.color_palette("colorblind", n_colors=10)

# Define variable groups
vessels = ['aorta', 'pa', 'svc', 'ivc']
chambers = ['ra', 'rv', 'la', 'lv']
labels = {
    'aorta': 'AA', 'pa': 'PA', 'svc': 'SVC', 'ivc': 'IVC',
    'ra': 'RA', 'rv': 'RV', 'la': 'LA', 'lv': 'LV'
}

# ===== PLOT 1: Great Vessels (AA-IVC) =====
fig = plt.figure(figsize=(8, 8))

for idx, variable in enumerate(vessels):
    x_values = df1[variable]
    y_values = df2[variable]
    
    min_length = min(len(x_values), len(y_values))
    x_values = x_values[:min_length]
    y_values = y_values[:min_length]
    
    sns.scatterplot(x=x_values, y=y_values, color=colors[idx], s=75, 
                   label=labels[variable], alpha=1.0)

# Find global min/max for reference line
all_vessels_x = pd.concat([df1[v] for v in vessels])
all_vessels_y = pd.concat([df2[v] for v in vessels])
min_val = min(all_vessels_x.min(), all_vessels_y.min())
max_val = max(all_vessels_x.max(), all_vessels_y.max())

plt.plot([min_val, max_val], [min_val, max_val],
         color="black", linestyle="--", linewidth=2.5, label="y = x")

plt.xlabel("MD (Gy)", fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.ylabel("AI (Gy)", fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(loc="upper left", fontsize=fontsize-4, frameon=True, fancybox=False)

# Add caption
fig.text(0.5, -0.02, 
         'Scatter plot comparing mean dose values. MD: manual delineation; AI: AI-segmentation.',
         ha='center', va='top', fontsize=10, wrap=True, transform=fig.transFigure)

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)

plt.tight_layout()
# plt.savefig('tablesandfigures/dose_great_vessels.pdf', bbox_inches='tight', dpi=300)
# plt.close()

# ===== PLOT 2: Chambers (RA-LV) =====
fig = plt.figure(figsize=(8, 8))

for idx, variable in enumerate(chambers):
    x_values = df1[variable]
    y_values = df2[variable]
    
    min_length = min(len(x_values), len(y_values))
    x_values = x_values[:min_length]
    y_values = y_values[:min_length]
    
    sns.scatterplot(x=x_values, y=y_values, color=colors[idx], s=75, 
                   label=labels[variable], alpha=1.0)

# Find global min/max for reference line
all_chambers_x = pd.concat([df1[v] for v in chambers])
all_chambers_y = pd.concat([df2[v] for v in chambers])
min_val = min(all_chambers_x.min(), all_chambers_y.min())
max_val = max(all_chambers_x.max(), all_chambers_y.max())

plt.plot([min_val, max_val], [min_val, max_val],
         color="black", linestyle="--", linewidth=2.5, label="y = x")

plt.xlabel("MD (Gy)", fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.ylabel("AI (Gy)", fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(loc="upper left", fontsize=fontsize-4, frameon=True, fancybox=False)
# plt.title("Chambers", fontsize=fontsize)
# Add caption
fig.text(0.5, -0.02, 
         'Scatter plot comparing mean dose values. MD: manual delineation; AI: AI-segmentation.',
         ha='center', va='top', fontsize=10, wrap=True, transform=fig.transFigure)
plt.tight_layout()
# plt.savefig('tablesandfigures/dose_chambers.pdf', bbox_inches='tight', dpi=300)
# plt.close()

print("Figures saved!")