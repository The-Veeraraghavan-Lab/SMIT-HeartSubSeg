from pathlib import Path

import pandas as pd
import scipy.stats as stats

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

mode = 'breast66'

# Load the datasets
# Load with 'name' as string
gt = pd.read_csv(RESULTS_ROOT / 'xcelrecords_dose' / mode / 'gt.csv', dtype={'name': str})
run1_plus_cnc64 = pd.read_csv(RESULTS_ROOT / 'xcelrecords_dose' / mode / 'run1_plus_cnc64_bnorm.csv', dtype={'name': str})

# Zero-pad to 8 characters
gt['name'] = gt['name'].str.zfill(8)
run1_plus_cnc64['name'] = run1_plus_cnc64['name'].str.zfill(8)

EXCLUDED_CASE_IDS = {'XXXX'}
gt = gt[~gt['name'].isin(EXCLUDED_CASE_IDS)]
common_names = set(gt['name']) & set(run1_plus_cnc64['name'])
gt = gt[gt['name'].isin(common_names)]
run1_plus_cnc64 = run1_plus_cnc64[run1_plus_cnc64['name'].isin(common_names)]

# Extract vessel names (excluding 'name' column)
vessels = gt.columns[1:]

# Initialize lists to store results
results = []

# Compute mean, std, and perform significance test
for vessel in vessels:
    gt_mean, gt_std = gt[vessel].mean(), gt[vessel].std()
    cnc64_mean, cnc64_std = run1_plus_cnc64[vessel].mean(), run1_plus_cnc64[vessel].std()
    
    # Perform Mann-Whitney Wilcoxon test
    stat, p_value = stats.mannwhitneyu(gt[vessel], run1_plus_cnc64[vessel], alternative='two-sided')
    
    # Store the results
    results.append({
        "Vessel": vessel,
        "GT Mean": round(gt_mean, 3),
        "GT Std": round(gt_std, 3),
        "CNC64 Mean": round(cnc64_mean, 3),
        "CNC64 Std": round(cnc64_std, 3),
        "Mann-Whitney U": round(stat, 3),
        "p-value": round(p_value, 5)
    })

# Convert results into a DataFrame and display
df_results = pd.DataFrame(results)

# # Print results
# import ace_tools as tools
# tools.display_dataframe_to_user(name="GT vs CNC64 Comparison", dataframe=df_results)
# Print results individually
for result in results:
    print(f"Vessel: {result['Vessel']}")
    print(f"{result['GT Mean']:.1f} {result['GT Std']:.1f} {result['CNC64 Mean']:.1f} {result['CNC64 Std']:.1f} {result['p-value']:.3f}")