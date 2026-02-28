# src/metrics/statistics.py
"""
Statistical analysis functions: t-tests, ANOVA, effect sizes, etc.
"""

import numpy as np
import pandas as pd
from scipy import stats


def interpret_cohens_d(d):
    """
    Interpret Cohen's d effect size magnitude.
    
    Parameters
    ----------
    d : float
        Cohen's d value
    
    Returns
    -------
    str
        Interpretation: 'negligible', 'small', 'medium', or 'large'
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size between two groups.
    
    Parameters
    ----------
    group1 : array-like
        First group of values
    group2 : array-like
        Second group of values
    
    Returns
    -------
    float
        Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0
    
    mean_diff = np.mean(group1) - np.mean(group2)
    return mean_diff / pooled_std


def paired_ttest(baseline_values, comparison_values):
    """
    Perform paired t-test between baseline and comparison.
    
    Parameters
    ----------
    baseline_values : array-like
        Values from baseline algorithm
    comparison_values : array-like
        Values from comparison algorithm
    
    Returns
    -------
    dict
        Dictionary with t-statistic, p-value, and significance flag
    """
    t_stat, p_value = stats.ttest_rel(baseline_values, comparison_values)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'significance_level': 'p < 0.05' if p_value < 0.05 else 'p ≥ 0.05'
    }


def anova_test(data_groups):
    """
    Perform one-way ANOVA test across multiple groups.
    
    Parameters
    ----------
    data_groups : list of arrays
        List of groups to compare
    
    Returns
    -------
    dict
        Dictionary with F-statistic, p-value, and significance flag
    """
    f_stat, p_value = stats.f_oneway(*data_groups)
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def advanced_statistical_analysis(results_dict, baseline_algo='MOEA/D'):
    """
    Perform comprehensive statistical analysis on algorithm results.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of corridor_name -> algorithm_results
    baseline_algo : str
        Name of baseline algorithm for comparisons
    
    Returns
    -------
    dict
        Detailed statistical analysis results
    """
    analysis_results = {}
    
    for corridor_name, results in results_dict.items():
        corridor_analysis = {}
        
        # Convert to DataFrame for easier analysis
        all_data = []
        for algo_name, algo_results in results.items():
            df_algo = pd.DataFrame(algo_results)
            df_algo['Algorithm'] = algo_name
            all_data.append(df_algo)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # ANOVA tests
        algorithms = list(results.keys())
        cost_groups = [combined_df[combined_df['Algorithm'] == algo]['AverageCost'].values 
                      for algo in algorithms]
        time_groups = [combined_df[combined_df['Algorithm'] == algo]['AverageTime'].values 
                      for algo in algorithms]
        
        anova_cost = anova_test(cost_groups)
        anova_time = anova_test(time_groups)
        
        # Effect sizes (Cohen's d) relative to baseline
        effect_sizes = {}
        baseline_data = combined_df[combined_df['Algorithm'] == baseline_algo]
        
        for algo_name in algorithms:
            if algo_name == baseline_algo:
                continue
                
            algo_data = combined_df[combined_df['Algorithm'] == algo_name]
            
            if len(baseline_data) > 0 and len(algo_data) > 0:
                # Cohen's d for cost
                d_cost = cohens_d(baseline_data['AverageCost'].values, 
                                  algo_data['AverageCost'].values)
                
                # Cohen's d for time
                d_time = cohens_d(baseline_data['AverageTime'].values, 
                                  algo_data['AverageTime'].values)
                
                effect_sizes[algo_name] = {
                    'cost_effect_size': d_cost,
                    'time_effect_size': d_time,
                    'cost_effect_magnitude': interpret_cohens_d(d_cost),
                    'time_effect_magnitude': interpret_cohens_d(d_time)
                }
        
        corridor_analysis['anova_cost'] = anova_cost
        corridor_analysis['anova_time'] = anova_time
        corridor_analysis['effect_sizes'] = effect_sizes
        analysis_results[corridor_name] = corridor_analysis
    
    return analysis_results


def generate_performance_ranking(results_df, weights=None):
    """
    Generate overall performance ranking of algorithms.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns: algorithm, HV, robustness_score, survival_rate
    weights : dict, optional
        Weights for different metrics (default: 40% HV, 40% robustness, 20% survival)
    
    Returns
    -------
    pd.DataFrame
        Ranked algorithms with combined scores
    """
    if weights is None:
        weights = {'HV': 0.4, 'robustness_score': 0.4, 'survival_rate': 0.2}
    
    # Group by algorithm and compute means
    summary = results_df.groupby('algorithm').agg({
        'HV': 'mean',
        'robustness_score': 'mean',
        'survival_rate': 'mean'
    }).reset_index()
    
    # Normalize metrics
    for metric in ['HV', 'robustness_score', 'survival_rate']:
        max_val = summary[metric].max()
        if max_val > 0:
            summary[f'{metric}_norm'] = summary[metric] / max_val
        else:
            summary[f'{metric}_norm'] = 0
    
    # Calculate combined score
    summary['combined_score'] = (
        weights['HV'] * summary['HV_norm'] * 100 +
        weights['robustness_score'] * summary['robustness_score_norm'] * 100 +
        weights['survival_rate'] * summary['survival_rate_norm'] * 100
    )
    
    # Sort by combined score
    summary = summary.sort_values('combined_score', ascending=False).reset_index(drop=True)
    summary['rank'] = summary.index + 1
    
    return summary