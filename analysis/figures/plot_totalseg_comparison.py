"""
HD95 Comparison: SMIT vs TotalSegmentator
Penalizes failed segmentations (NaN) with HD95 = 20mm
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import wilcoxon

# ============================================================================
# Configuration
# ============================================================================
STRUCTURES = ['aorta', 'pa', 'svc', 'ivc', 'ra', 'rv', 'la', 'lv']
STRUCTURE_NAMES = {
    'aorta': 'AA', 'pa': 'PA', 'svc': 'SVC', 'ivc': 'IVC',
    'ra': 'RA', 'rv': 'RV', 'la': 'LA', 'lv': 'LV'
}
PENALTY_HD95 = 50.0  # matches manuscript sentinel

# Colors - colorblind friendly
sns.set_palette("colorblind")
colors = sns.color_palette("colorblind", n_colors=4)
COLOR_SMIT = colors[0]  # blue
COLOR_TS = colors[3]    # red/pink

# ============================================================================
# Data Loading
# ============================================================================
import sys
RESULTS_ROOT = Path(__file__).resolve().parents[1] / 'results'
# Usage: python plot_totalseg_comparison.py [oar|breast]
_cohort = sys.argv[1] if len(sys.argv) > 1 else 'breast'
data_dir = RESULTS_ROOT / f'xcelrecords_{_cohort}'
_panel_label = '3e' if _cohort == 'oar' else '3f'

results = []
merged_data = {}  # Store merged data for significance testing

for struct in STRUCTURES:
    smit = pd.read_csv(data_dir / f'smit_totalseg_{struct}.csv')
    totalseg = pd.read_csv(data_dir / f'totalseg_{struct}.csv')
    
    smit['name'] = smit['name'].astype(str)
    totalseg['name'] = totalseg['name'].astype(str)
    
    merged = pd.merge(
        smit[['name', 'hd95']], 
        totalseg[['name', 'hd95']], 
        on='name', 
        suffixes=('_smit', '_ts'), 
        how='inner'
    )
    
    merged['hd95_smit'] = merged['hd95_smit'].fillna(PENALTY_HD95)
    merged['hd95_ts'] = merged['hd95_ts'].fillna(PENALTY_HD95)
    
    # Store for plotting
    merged_data[struct] = merged
    
    # Paired Wilcoxon signed-rank test (same cases per row -> paired)
    v_smit = merged['hd95_smit'].values
    v_ts = merged['hd95_ts'].values
    if np.array_equal(v_smit, v_ts):
        stat, pval = np.nan, np.nan
    else:
        stat, pval = wilcoxon(v_smit, v_ts, zero_method='wilcox', alternative='two-sided')
    
    results.append({
        'struct': struct,
        'name': STRUCTURE_NAMES[struct],
        'smit_mean': merged['hd95_smit'].mean(),
        'smit_std': merged['hd95_smit'].std(),
        'smit_max': merged['hd95_smit'].max(),
        'ts_mean': merged['hd95_ts'].mean(),
        'ts_std': merged['hd95_ts'].std(),
        'ts_max': merged['hd95_ts'].max(),
        'n': len(merged),
        'U_stat': stat,
        'p_value': pval,
    })

df = pd.DataFrame(results)

print(df[['name', 'smit_mean', 'ts_mean', 'U_stat', 'p_value']])

#%%
#OAR: 12, 4, **
#Breast: 16, _, **

# ============================================================================
# Plot
# ============================================================================
def sig_stars(p):
    if p < 0.0001: return '****'
    elif p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return ''

fig, ax = plt.subplots(figsize=(8, 8))

x = np.arange(len(df))
width = 0.35

bars1 = ax.bar(
    x - width/2, df['smit_mean'], width, 
    yerr=df['smit_std'], 
    label='SMIT', 
    color=COLOR_SMIT, 
    alpha=0.85, 
    capsize=4, 
    error_kw={'linewidth': 1.5}
)

bars2 = ax.bar(
    x + width/2, df['ts_mean'], width, 
    yerr=df['ts_std'], 
    label='nnU-Net', 
    color=COLOR_TS, 
    alpha=0.85, 
    capsize=4, 
    error_kw={'linewidth': 1.5}
)

# Add significance brackets
for i, row in df.iterrows():
    sig_text = sig_stars(row['p_value'])
    
    if sig_text:  # Only draw bracket if significant
        # Bar positions
        x_left = i - width/2
        x_right = i + width/2
        
        # Y position: clamp the bracket zone above whisker top but cap at 56 mm so it fits
        # within the 70 mm axis (sentinel bars top out at 50 mm).
        y_max_raw = max(row['smit_mean'] + row['smit_std'],
                        row['ts_mean'] + row['ts_std'])
        y_max = min(y_max_raw, 55.0)
        gap = 1.5
        bracket_height = y_max + gap
        bar_drop = 0.4  # How far down the vertical lines go
        
        # Draw bracket
        # Left vertical line
        ax.plot([x_left, x_left], [y_max + 0.3, bracket_height], 
                'k-', linewidth=1, zorder=10)
        # Horizontal line
        ax.plot([x_left, x_right], [bracket_height, bracket_height], 
                'k-', linewidth=1, zorder=10)
        # Right vertical line
        ax.plot([x_right, x_right], [y_max + 0.3, bracket_height], 
                'k-', linewidth=1, zorder=10)
        
        # Stars above bracket
        ax.text((x_left + x_right) / 2, bracket_height + 0.2, sig_text, 
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('HD95 (mm)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(df['name'], fontsize=11)
ax.legend(fontsize=12, title='Data generalization', title_fontsize=12)
# ax.legend(fontsize=12)
ax.set_ylim(0, 70)  # Headroom for sentinel=50 bars + bracket annotations
ax.grid(False) 

caption_text = 'HD95 (mm): Bars show mean ± standard deviation. Failed segmentations penalized as 50 mm. *p<0.05, **p<0.01, ***p<0.001, ****p<0.0001'
fig.text(0.5, -0.005, caption_text, 
         ha='center', va='top', fontsize=10, 
         wrap=True, transform=fig.transFigure)

plt.tight_layout()
plt.savefig(RESULTS_ROOT / 'tablesandfigures' / f'figure_{_panel_label}.pdf', dpi=300, bbox_inches='tight', facecolor='white')