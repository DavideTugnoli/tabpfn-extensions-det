import pandas as pd
import numpy as np
import itertools
import random
import warnings
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import re

# Add parent directory to path for imports
# This allows us to import from the 'utils' and 'viz_config' directories
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causal_experiments.utils.scm_data import get_dag_and_config
from causal_experiments.utils.dag_utils import convert_indices_dag_to_named, get_ordering_strategies
from causal_experiments.viz_config.viz_config import setup_plotting, FONT_SIZES, DPI

# --- Core Analysis Functions ---

def _discretize_for_kmarginal(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cat_cols: list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Internal function to replicate the percentile-rank-based binning logic from SynthEval.
    This ensures our standalone calculation is consistent with the original metric.
    """
    numeric_features = [
        col for col in real_data.columns
        if col not in cat_cols and real_data[col].nunique() >= 20
    ]

    real_binned = real_data.copy()
    for col in numeric_features:
        not_na_mask = real_binned[col].notna()
        # Rank data and create 20 bins based on percentiles
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

        # Map synthetic data values to the bins defined by the real data
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

def calculate_pairwise_tvd(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cat_cols: list = None,
    k: int = 2
) -> dict:
    """
    Calculates the Total Variation Distance (TVD) for all k-wise marginals (pairs of variables).
    
    This is an adaptation of the `calculate_kmarginal_tvd` function, but instead of returning
    the mean TVD, it returns a dictionary with the TVD for each individual pair.

    Returns:
        A dictionary where keys are tuples of variable names (e.g., ('X1', 'X2'))
        and values are the corresponding TVD scores.
    """
    if cat_cols is None:
        cat_cols = []
    
    # Use the same discretization as the original metric to ensure comparability
    real_processed, syn_processed = _discretize_for_kmarginal(real_data, synthetic_data, cat_cols)
    features = real_processed.columns.tolist()

    if len(features) < k:
        return {}

    # Get all combinations of k variables
    marginals = list(itertools.combinations(sorted(features), k))
    
    if not marginals:
        return {}

    pairwise_tvd = {}
    for marg in marginals:
        marg_list = list(marg)
        
        # Calculate probability densities for the real and synthetic data for the current marginal
        t_den = real_processed.groupby(marg_list).size() / len(real_processed)
        s_den = syn_processed.groupby(marg_list).size() / len(syn_processed)
        
        # Calculate the Total Variation Distance (TVD)
        # TVD = 0.5 * sum(|P(i) - Q(i)|)
        abs_den_diff = t_den.subtract(s_den, fill_value=0).abs()
        total_density_diff = abs_den_diff.sum()
        tvd = total_density_diff / 2.0
        
        # Store the result
        pairwise_tvd[marg] = tvd
        
    return pairwise_tvd

def get_correlated_pairs_from_dag(dag: dict, col_names: list) -> list:
    """
    Extract the pairs of variables that are correlated according to the DAG structure.
    
    Args:
        dag: DAG structure as {node_index: [parent_indices]}
        col_names: List of column names
        
    Returns:
        List of tuples representing correlated variable pairs
    """
    correlated_pairs = []
    
    # Get all edges from the DAG
    for child_idx, parent_indices in dag.items():
        for parent_idx in parent_indices:
            # Add the parent-child pair
            parent_name = col_names[parent_idx]
            child_name = col_names[child_idx]
            pair = tuple(sorted([parent_name, child_name]))  # Sort for consistency
            if pair not in correlated_pairs:
                correlated_pairs.append(pair)
    
    return correlated_pairs

def analyze_all_pairs_with_differences():
    """
    Main function to analyze ALL variable pairs and show DIFFERENCES between DAGs.
    This provides a complete view while highlighting the correlated pairs from the DAG.
    """
    # Configuration
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / 'experiment_4_det_copy' / 'data_samples' / 'all_samples'
    output_dir = script_dir / 'all_pairs_differences_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Get DAG structure dynamically from utility functions
    dag, col_names, cat_cols_indices = get_dag_and_config(include_categorical=False)
    cat_cols = [col_names[i] for i in cat_cols_indices]
    
    # Get correlated pairs from DAG structure
    correlated_pairs = get_correlated_pairs_from_dag(dag, col_names)
    
    print("=== COMPLETE PAIRS ANALYSIS WITH DIFFERENCES ===")
    print(f"True DAG structure: {dag}")
    print(f"Column names: {col_names}")
    print(f"Correlated pairs according to DAG: {correlated_pairs}")
    print()
    
    # Convert DAG to named format for better readability
    named_dag = convert_indices_dag_to_named(dag, col_names)
    print("DAG structure (named):")
    for child, parents in named_dag.items():
        if parents:
            print(f"  {child} ← {', '.join(parents)}")
        else:
            print(f"  {child} (independent)")
    print()
    
    dag_types_to_compare = ['dag_1_3_edges', 'dag_2_3_edges_correct']
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Find all synthetic data files
    all_files = list(data_dir.glob('*_synth.csv'))
    print(f"Found {len(all_files)} synthetic data files.")

    # Store all results by repetition to calculate differences
    all_results = {}
    
    # Process each file
    for synth_file in all_files:
        # Parse filename
        match = re.match(r"(?P<dag_type>.*)_size(?P<size>\d+)_rep(?P<rep>\d+)_synth", synth_file.stem)
        
        if not match:
            print(f"Warning: Filename format not recognized, skipping: {synth_file.name}")
            continue

        parsed = match.groupdict()
        full_dag_name = parsed['dag_type']
        train_size = int(parsed['size'])
        repetition = int(parsed['rep'])

        # Normalize the extracted DAG type
        dag_type = full_dag_name.removeprefix('dag_')
        dag_type = dag_type.replace(' (correct)', '_correct')
        
        # We only care about the two DAG types of interest
        if dag_type not in dag_types_to_compare:
            continue
            
        # Corresponding real data (from the test set)
        test_file = Path(str(synth_file).replace('_synth.csv', '_test.csv'))
        if not test_file.exists():
            print(f"Warning: Test file not found for {synth_file.name}")
            continue
            
        # Load data
        real_df = pd.read_csv(test_file)
        synth_df = pd.read_csv(synth_file)
        
        # Ensure column names are consistent
        real_df.columns = col_names
        synth_df.columns = col_names
        
        # Calculate pairwise TVD for ALL pairs
        all_tvd_results = calculate_pairwise_tvd(real_df, synth_df, cat_cols, k=2)
        
        # Store results by repetition
        key = (train_size, repetition)
        if key not in all_results:
            all_results[key] = {}
        
        all_results[key][dag_type] = all_tvd_results

    # Calculate differences between DAGs
    difference_results = []
    
    for (train_size, repetition), dag_results in all_results.items():
        if len(dag_results) == 2:  # Both DAGs present
            dag1_results = dag_results['dag_1_3_edges']
            dag2_results = dag_results['dag_2_3_edges_correct']
            
            # Calculate differences for all pairs
            all_pairs = set(dag1_results.keys()) | set(dag2_results.keys())
            
            for pair in all_pairs:
                dag1_tvd = dag1_results.get(pair, 0)
                dag2_tvd = dag2_results.get(pair, 0)
                difference = dag1_tvd - dag2_tvd  # DAG1 - DAG2
                
                # Check if this pair is correlated according to DAG
                is_correlated = pair in correlated_pairs
                
                difference_results.append({
                    'train_size': train_size,
                    'repetition': repetition,
                    'variable_pair': f"{pair[0]}-{pair[1]}",
                    'dag1_tvd': dag1_tvd,
                    'dag2_tvd': dag2_tvd,
                    'difference': difference,
                    'is_correlated': is_correlated
                })

    if not difference_results:
        print("No results were generated. Please check the data directory and filenames.")
        return

    # Convert results to a DataFrame for easier plotting
    results_df = pd.DataFrame(difference_results)
    
    print("\n--- ANALYSIS COMPLETE ---")
    print(f"Generated {len(results_df)} data points for differences.")
    print(f"Total pairs analyzed: {len(results_df['variable_pair'].unique())}")
    print(f"Correlated pairs: {correlated_pairs}")
    print("First 5 rows of results:")
    print(results_df.head())
    
    # --- Statistical Analysis ---
    print("\n=== STATISTICAL ANALYSIS OF DIFFERENCES ===")
    
    # Group by pair and calculate statistics
    for pair in sorted(results_df['variable_pair'].unique()):
        pair_data = results_df[results_df['variable_pair'] == pair]
        is_correlated = pair_data['is_correlated'].iloc[0]
        
        mean_diff = pair_data['difference'].mean()
        std_diff = pair_data['difference'].std()
        
        status = "*** CORRELATED ***" if is_correlated else "uncorrelated"
        print(f"\nPair: {pair} ({status})")
        print(f"  Mean difference (DAG1 - DAG2): {mean_diff:.4f} ± {std_diff:.4f}")
        print(f"  Mean DAG1 TVD: {pair_data['dag1_tvd'].mean():.4f}")
        print(f"  Mean DAG2 TVD: {pair_data['dag2_tvd'].mean():.4f}")
    
    # --- Plotting ---
    setup_plotting()
    
    # Get the sorted order of all pairs, with correlated ones first
    all_pairs = sorted(results_df['variable_pair'].unique())
    correlated_pairs_str = [f"{p[0]}-{p[1]}" for p in correlated_pairs]
    
    # Sort: correlated pairs first, then others
    sorted_pairs = sorted(all_pairs, key=lambda x: (x not in correlated_pairs_str, x))
    train_sizes = sorted(results_df['train_size'].unique())

    # --- Summary Statistics Table (for max diff) ---
    summary_stats = results_df.groupby(['variable_pair', 'is_correlated'])['difference'].agg(['mean', 'std', 'count']).round(4)
    print("\n=== SUMMARY STATISTICS TABLE ===")
    print(summary_stats)
    # Save summary to CSV
    summary_csv = output_dir / 'tvd_differences_summary.csv'
    summary_stats.to_csv(summary_csv)
    print(f"\nSummary statistics saved to: {summary_csv}")

    # Trova la coppia con la differenza media massima
    max_diff_row = summary_stats['mean'].abs().idxmax()
    max_diff_pair = max_diff_row[0]

    # Generate a separate plot for each training size
    for size in train_sizes:
        plt.figure(figsize=(14, 8))
        ax = plt.gca()

        # Filter data for the current training size
        size_df = results_df[results_df['train_size'] == size]

        # Create separate dataframes for correlated and uncorrelated pairs
        correlated_df = size_df[size_df['is_correlated'] == True]
        uncorrelated_df = size_df[size_df['is_correlated'] == False]
        
        # Plot uncorrelated pairs first (background)
        if not uncorrelated_df.empty:
            sns.boxplot(
                data=uncorrelated_df,
                x='variable_pair',
                y='difference',
                order=sorted_pairs,
                ax=ax,
                color='lightblue'
            )
        
        # Plot correlated pairs on top (highlighted)
        if not correlated_df.empty:
            sns.boxplot(
                data=correlated_df,
                x='variable_pair',
                y='difference',
                order=sorted_pairs,
                ax=ax,
                color='#FF6B6B'
            )

        # Add horizontal line at zero
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No difference')

        # Customize plot aesthetics
        ax.set_title(f'TVD Differences: DAG1 - DAG2 (Training Size: {size})', fontsize=FONT_SIZES['title'], pad=15)
        ax.set_xlabel("Variable Pairs", fontsize=FONT_SIZES['label'])
        ax.set_ylabel("TVD Difference (DAG1 - DAG2)", fontsize=FONT_SIZES['label'])
        ax.tick_params(axis='x', rotation=0, labelsize=FONT_SIZES['tick'])  # Horizontal labels
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Improve layout and remove top/right spines
        sns.despine(left=True)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='Correlated pairs (from DAG)'),
            Patch(facecolor='lightblue', label='Uncorrelated pairs')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=FONT_SIZES['legend'])

        # Add text box with DAG structure
        dag_text = "DAG Structure:\n" + "\n".join([f"{child} ← {', '.join(parents)}" if parents else f"{child} (independent)" 
                                                  for child, parents in named_dag.items()])
        plt.figtext(0.02, 0.02, dag_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        # Evidenzia le label delle coppie causali e quella con la differenza massima
        for tick_label in ax.get_xticklabels():
            label = tick_label.get_text()
            if label == max_diff_pair:
                tick_label.set_color('blue')
                tick_label.set_fontweight('bold')
            elif label in correlated_pairs_str:
                tick_label.set_color('red')
                tick_label.set_fontweight('bold')
            else:
                tick_label.set_color('black')
                tick_label.set_fontweight('normal')

        # Adjust layout to prevent labels overlapping
        plt.tight_layout()

        # Save the plot
        output_filename = output_dir / f'tvd_differences_analysis_size_{size}.png'
        plt.savefig(output_filename, dpi=DPI, bbox_inches='tight')
        print(f"Plot for size {size} saved to: {output_filename}")
        plt.close()

if __name__ == '__main__':
    analyze_all_pairs_with_differences() 