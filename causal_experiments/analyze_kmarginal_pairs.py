import os
import pandas as pd
import numpy as np
import itertools
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Discretization logic (copied from _discretize_for_kmarginal)
def discretize_for_kmarginal(real_data, synthetic_data, cat_cols):
    numeric_features = [
        col for col in real_data.columns
        if col not in cat_cols and real_data[col].nunique() >= 20
    ]
    real_binned = real_data.copy()
    for col in numeric_features:
        not_na_mask = real_binned[col].notna()
        ranked_pct = real_binned.loc[not_na_mask, col].rank(pct=True)
        real_binned.loc[not_na_mask, col] = ranked_pct.apply(lambda x: int(20 * x) if x < 1 else 19)
        real_binned.loc[real_data[col].isna(), col] = -1
    syn_binned = synthetic_data.copy()
    for col in numeric_features:
        syn_not_na_mask = syn_binned[col].notna()
        syn_numeric_values = pd.to_numeric(syn_binned.loc[syn_not_na_mask, col])
        binned_syn_values = syn_numeric_values.copy()
        max_value_of_previous_bin = -np.inf
        unique_bins = sorted([b for b in real_binned[col].unique() if b != -1])
        for i, bin_val in enumerate(unique_bins):
            indices_for_bin = real_binned[real_binned[col] == bin_val].index
            max_value_in_bin = real_data.loc[indices_for_bin, col].max()
            min_value_of_current_bin = max_value_of_previous_bin
            condition = (syn_numeric_values > min_value_of_current_bin)
            if i < len(unique_bins) - 1:
                condition &= (syn_numeric_values <= max_value_in_bin)
            binned_syn_values.loc[condition] = bin_val
            max_value_of_previous_bin = max_value_in_bin
        syn_binned.loc[syn_not_na_mask, col] = binned_syn_values
        syn_binned.loc[synthetic_data[col].isna(), col] = -1
    all_cols = numeric_features + cat_cols
    return real_binned[all_cols].astype(int), syn_binned[all_cols].astype(int)

def compute_pairwise_kmarginal_tvd(real, synth, cat_cols):
    real_binned, synth_binned = discretize_for_kmarginal(real, synth, cat_cols)
    features = real_binned.columns.tolist()
    results = []
    heatmap = np.zeros((len(features), len(features)))
    for i, j in itertools.combinations(range(len(features)), 2):
        f1, f2 = features[i], features[j]
        marg = [f1, f2]
        t_den = real_binned.groupby(marg).size() / len(real_binned)
        s_den = synth_binned.groupby(marg).size() / len(synth_binned)
        abs_den_diff = t_den.subtract(s_den, fill_value=0).abs()
        tvd = abs_den_diff.sum() / 2.0
        results.append({'feature1': f1, 'feature2': f2, 'tvd': tvd})
        heatmap[i, j] = tvd
        heatmap[j, i] = tvd
    return pd.DataFrame(results), features, heatmap

def main():
    parser = argparse.ArgumentParser(description='Analyze k-marginal TVD for all feature pairs.')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to all_samples directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save results')
    parser.add_argument('--cat_cols', type=str, default='', help='Comma-separated categorical column names (optional)')
    parser.add_argument('--prefix', type=str, default='', help='Prefix to filter files (e.g., dag_dag_min_3_edges_size100)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # Find all test/synth pairs
    files = os.listdir(args.input_dir)
    synth_files = [f for f in files if f.endswith('_synth.csv') and args.prefix in f]
    for synth_file in synth_files:
        base = synth_file[:-10]  # remove _synth.csv
        test_file = base + '_test.csv'
        if test_file not in files:
            print(f"Warning: test file {test_file} not found for {synth_file}")
            continue
        synth_path = os.path.join(args.input_dir, synth_file)
        test_path = os.path.join(args.input_dir, test_file)
        synth = pd.read_csv(synth_path)
        test = pd.read_csv(test_path)
        cat_cols = [c for c in args.cat_cols.split(',') if c] if args.cat_cols else []
        df, features, heatmap = compute_pairwise_kmarginal_tvd(test, synth, cat_cols)
        out_prefix = os.path.join(args.output_dir, base)
        df.to_csv(out_prefix + '_pairwise_kmarginal.csv', index=False)
        # Save heatmap
        plt.figure(figsize=(len(features)*0.8, len(features)*0.8))
        sns.heatmap(heatmap, xticklabels=features, yticklabels=features, annot=True, fmt='.3f', cmap='viridis')
        plt.title(f'Pairwise k-marginal TVD: {base}')
        plt.tight_layout()
        plt.savefig(out_prefix + '_pairwise_kmarginal_heatmap.png')
        plt.close()
        print(f"Saved: {out_prefix}_pairwise_kmarginal.csv and heatmap.")

if __name__ == '__main__':
    main() 