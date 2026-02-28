# src/visualization/plotting.py
"""
Visualization functions for Pareto fronts, sensitivity analysis, and performance comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D


def plot_pareto_fronts(results, problem_name, save_path=None):
    """
    Plot Pareto fronts from different algorithms.
    
    Parameters
    ----------
    results : dict
        Dictionary of algorithm_name -> result object with F attribute
    problem_name : str
        Name of the problem for title
    save_path : str, optional
        Path to save the figure
    """
    if problem_name not in results:
        print(f"No results for {problem_name}")
        return

    fig = plt.figure(figsize=(12, 8))
    num_objectives = None
    
    # Determine number of objectives
    for result in results[problem_name].values():
        if result is not None and hasattr(result, 'F') and result.F is not None and result.F.size > 0:
            num_objectives = result.F.shape[1]
            break

    if num_objectives == 2:
        for name, result in results[problem_name].items():
            if result is not None and hasattr(result, 'F') and result.F is not None:
                plt.scatter(result.F[:, 0], result.F[:, 1], label=name, alpha=0.7, s=30)
        plt.xlabel("Objective 1")
        plt.ylabel("Objective 2")
        
    elif num_objectives == 3:
        ax = fig.add_subplot(111, projection='3d')
        for name, result in results[problem_name].items():
            if result is not None and hasattr(result, 'F') and result.F is not None:
                ax.scatter(result.F[:, 0], result.F[:, 1], result.F[:, 2], 
                          label=name, alpha=0.7, s=30)
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_zlabel("Objective 3")
        
    else:
        print(f"Cannot visualize {num_objectives} objectives")
        return

    plt.title(f"Pareto Fronts: {problem_name}", fontweight='bold', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sensitivity_analysis(performance_data, param_names, save_path=None):
    """
    Plot parameter sensitivity analysis results.
    
    Parameters
    ----------
    performance_data : dict
        Dictionary of parameter -> list of (value, HV, Survival) tuples
    param_names : list
        List of parameter names for titles
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Parameter Sensitivity Analysis for En-MOEA/D', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e']
    titles = ['Mean Weight (α)', 'Variance Weight (β)', 'Balance Parameter (ω)', 
              'Variance Weight (γ)', 'Convergence Window (τ)', 'MC Samples']
    
    for idx, (param, title) in enumerate(zip(param_names, titles)):
        row, col = idx // 3, idx % 3
        param_data = performance_data[param]
        values = [d['value'] for d in param_data]
        
        ax1 = axes[row, col]
        
        # Plot Hypervolume
        hvs = [d['HV'] for d in param_data]
        line1 = ax1.plot(values, hvs, 'o-', color=colors[0], linewidth=3, 
                         markersize=8, markerfacecolor='white', 
                         label='Hypervolume')
        ax1.set_xlabel(f'{param} Value')
        ax1.set_ylabel('Hypervolume', color=colors[0])
        ax1.tick_params(axis='y', labelcolor=colors[0])
        ax1.grid(True, alpha=0.3)
        
        # Create secondary axis for Survival Rate
        ax2 = ax1.twinx()
        survivals = [d['Survival'] for d in param_data]
        line2 = ax2.plot(values, survivals, 's-', color=colors[1], linewidth=2,
                         markersize=6, label='Survival Rate')
        ax2.set_ylabel('Survival Rate (%)', color=colors[1])
        ax2.tick_params(axis='y', labelcolor=colors[1])
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                   ncol=2, fontsize=9)
        
        ax1.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_algorithm_comparison(performance_df, save_path=None):
    """
    Plot algorithm performance comparison.
    
    Parameters
    ----------
    performance_df : pd.DataFrame
        DataFrame with columns: Algorithm, Corridor, Normalized_Cost, Normalized_Time, Reliability
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Cost comparison
    sns.barplot(data=performance_df, x='Algorithm', y='Normalized_Cost', 
                hue='Corridor', ax=axes[0], palette='viridis')
    axes[0].set_title('Algorithm Cost Performance', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_ylabel('Average Cost (USD)')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Time comparison
    sns.barplot(data=performance_df, x='Algorithm', y='Normalized_Time', 
                hue='Corridor', ax=axes[1], palette='viridis')
    axes[1].set_title('Algorithm Time Performance', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylabel('Average Time (Hours)')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Reliability comparison
    sns.barplot(data=performance_df, x='Algorithm', y='Reliability', 
                hue='Corridor', ax=axes[2], palette='viridis')
    axes[2].set_title('Algorithm Reliability Performance', fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].set_ylabel('Reliability Score')
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_perturbation_trends(results_mombasa, results_dar, algorithms, save_path=None):
    """
    Plot performance degradation under increasing perturbations.
    
    Parameters
    ----------
    results_mombasa : dict
        Results for Northern Corridor
    results_dar : dict
        Results for Central Corridor
    algorithms : dict
        Dictionary of algorithm names
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    corridors = [('Mombasa', results_mombasa), ('Dar es Salaam', results_dar)]
    
    for idx, (corridor_name, results) in enumerate(corridors):
        ax = axes[idx]
        perturbations = [5, 10, 15, 20, 25]
        
        for algo_name in algorithms.keys():
            df_algo = pd.DataFrame(results[algo_name])
            costs = [df_algo[df_algo['Perturbation'] == p]['AverageCost'].values[0] 
                    for p in perturbations]
            ax.plot(perturbations, costs, 'o-', linewidth=2, markersize=8, 
                   label=algo_name)
        
        ax.set_xlabel('Perturbation Level (%)')
        ax.set_ylabel('Average Cost (USD)')
        ax.set_title(f'{corridor_name} Corridor - Cost Under Perturbations', 
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_dataset_analysis(final_df, save_path=None):
    """
    Plot comprehensive dataset analysis visualizations.
    
    Parameters
    ----------
    final_df : pd.DataFrame
        The East African dataset
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('East African Logistics Dataset Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Cost distribution by corridor
    sns.boxplot(data=final_df, x='Corridor', y='Cost (cijm) (USD/TEU)', 
                ax=axes[0,0], palette='viridis')
    axes[0,0].set_title('A) Cost Distribution by Corridor', fontweight='bold')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Time distribution by corridor
    sns.boxplot(data=final_df, x='Corridor', y='Stochastic Time (tijm,k) (Hours)', 
                ax=axes[0,1], palette='viridis')
    axes[0,1].set_title('B) Time Distribution by Corridor', fontweight='bold')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Reliability by mode
    sns.boxplot(data=final_df, x='Mode', y='Reliability (Relijm)', 
                ax=axes[1,0], palette='viridis')
    axes[1,0].set_title('C) Reliability by Transportation Mode', fontweight='bold')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Seasonal impact
    seasonal_data = final_df.groupby(['Corridor', 'Season'])['Stochastic Time (tijm,k) (Hours)'].mean().reset_index()
    sns.barplot(data=seasonal_data, x='Corridor', y='Stochastic Time (tijm,k) (Hours)', 
                hue='Season', ax=axes[1,1], palette='viridis')
    axes[1,1].set_title('D) Seasonal Impact on Transportation Time', fontweight='bold')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()