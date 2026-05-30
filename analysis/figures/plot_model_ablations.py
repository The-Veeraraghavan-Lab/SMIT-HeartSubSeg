import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = {
    'Substructure': ['AA', 'PA', 'SVC', 'IVC', 'RA', 'RV', 'LA', 'LV'],
    '8+8': [6.53, 7.12, 7.28, 8.19, 8.10, 9.26, 6.8, 6.7],
    '16+16': [5.88, 6.8, 6.58, 7.69, 8.01, 9.05, 6.68, 6.5],
    '32+32': [4.81, 6.35, 5.14, 6.34, 7.81, 8.83, 6.25, 6.38],
    '48+48': [4.52, 6.25, 5.08, 6.24, 7.56, 8.53, 5.89, 6.18],
    '56+56': [4.24, 5.95, 4.92, 6.87, 7.35, 8.14, 5.68, 5.79],
}
std_data = {
    'Substructure': ['AA', 'PA', 'SVC', 'IVC', 'RA', 'RV', 'LA', 'LV'],
    '8+8': [4.26, 3.58, 4.81, 5.47, 6.65, 3.25, 5.12, 4.28],
    '16+16': [5.47, 4.74, 3.86, 4.58, 4.45, 3.75, 4.85, 4.58],
    '32+32': [5.46, 4.32, 3.56, 3.02, 3.25, 4.69, 4.45, 3.82],
    '48+48': [5.23, 4.12, 3.26, 3.12, 3.02, 3.28, 2.81, 2.66],
    '56+56': [1.25, 3.84, 3.56, 3.95, 3.14, 3.01, 2.36, 1.68],
}

df_mean = pd.DataFrame(data)
df_std = pd.DataFrame(std_data)
configs = ['8+8', '16+16', '32+32', '48+48', '56+56']
x_vals = [8, 16, 32, 48, 56]


agg_mean = np.array([df_mean[c].mean() for c in configs])
agg_std = np.array([df_std[c].mean() for c in configs])

#%%
sns.set_style("white")
fig, ax = plt.subplots(figsize=(7, 4))

# Shaded std region
ax.fill_between(x_vals, agg_mean - agg_std, agg_mean + agg_std,
                color='steelblue', alpha=0.15)

# Line + points
ax.plot(x_vals, agg_mean, '-o', color='steelblue', markersize=7, linewidth=1.8, zorder=3)

# Red vertical line at 32
ax.axvline(x=32, color='red', linewidth=1.2, linestyle='--', zorder=1)

ax.set_xticks(x_vals)
ax.set_xticklabels(configs, fontsize=11)
ax.set_ylabel('HD95 (mm)', fontsize=12)
ax.set_xlabel('Training configuration', fontsize=12)
ax.grid(False)
ax.tick_params(labelsize=11)

caption = ('Shaded region indicates aggregate mean ± standard deviation. The dashed red line marks the\n'
           '32+32 configuration, beyond which marginal improvement diminishes.')
fig.text(0.5, -0.02, caption, ha='center', va='top', fontsize=9,
         wrap=True, transform=fig.transFigure)

plt.tight_layout()
plt.show()