#!/usr/bin/env python
# experiments/run_adaptive_decomposition.py
"""
Experiment 1: Adaptive Decomposition Methods Validation
Compares PBI-only, Tchebicheff-only, and Hybrid-Adaptive approaches.
"""

import os
import time
import numpy as np
import pandas as pd
from pymoo.optimize import minimize

from src.utils.config import (
    MAX_EVALUATIONS, POP_SIZE, N_GEN, SEED,
    REF_POINT_ZDT, REF_POINT_DTLZ, REF_POINT_WFG
)
from src.problems.benchmark import get_zdt_problems, get_dtlz_problems, get_wfg_problems
from src.algorithms.variants import get_pbi_algorithm, get_tchebicheff_algorithm
from src.algorithms.en_moead import EnhancedMOEAD
from src.metrics.performance import compute_all_metrics, get_reference_point
from src.visualization.plotting import plot_pareto_fronts
from src.utils.helpers import save_dataframe


def run_experiment(problem_name, problem, results_dir):
    """Run adaptive decomposition experiment on a single problem."""
    print(f"\n{'='*60}")
    print(f"Running Adaptive Decomposition on {problem_name}")
    print(f"{'='*60}")
    
    n_obj = problem.n_obj
    ref_point = get_reference_point(problem_name)
    
    # Get reference directions
    from pymoo.util.ref_dirs import get_reference_directions
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
    
    # Initialize algorithms
    algorithms = {
        "PBI-only": get_pbi_algorithm(ref_dirs),
        "Tchebicheff-only": get_tchebicheff_algorithm(ref_dirs),
        "Hybrid-Adaptive": EnhancedMOEAD(ref_dirs, base_problem=problem)
    }
    
    # Set reference point for EnhancedMOEAD
    algorithms["Hybrid-Adaptive"].reference_point = ref_point
    
    results = {}
    metrics_summary = []
    
    for name, algorithm in algorithms.items():
        print(f"\n  Running {name}...")
        start_time = time.time()
        
        try:
            res = minimize(
                problem,
                algorithm,
                ('n_gen', N_GEN),
                seed=SEED,
                verbose=False
            )
            
            runtime = time.time() - start_time
            
            if res is not None and hasattr(res, 'F') and res.F is not None:
                # Compute metrics
                metrics = compute_all_metrics(res.F, problem, problem_name)
                
                results[name] = res
                
                # Store for summary
                metrics_summary.append({
                    'Problem': problem_name,
                    'Algorithm': name,
                    'HV': metrics['HV'],
                    'IGD': metrics['IGD'],
                    'Spread': metrics['Spread'],
                    'Time_s': runtime
                })
                
                print(f"    HV: {metrics['HV']:.4f}, IGD: {metrics['IGD']:.4f}, "
                      f"Spread: {metrics['Spread']:.4f}, Time: {runtime:.2f}s")
            else:
                print(f"    ❌ No valid results")
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    # Plot Pareto fronts
    plot_pareto_fronts({problem_name: results}, problem_name,
                       save_path=os.path.join(results_dir, f"pareto_{problem_name}.png"))
    
    return metrics_summary


def main():
    """Main execution for adaptive decomposition experiments."""
    print("=" * 80)
    print("ADAPTIVE DECOMPOSITION METHODS VALIDATION")
    print("=" * 80)
    print(f"Parameters: Max Evaluations={MAX_EVALUATIONS}, Population={POP_SIZE}")
    
    # Create results directory
    results_dir = "results/adaptive_decomposition"
    os.makedirs(results_dir, exist_ok=True)
    
    all_metrics = []
    
    # Test ZDT problems
    print("\n" + "=" * 60)
    print("ZDT BENCHMARK SUITE")
    print("=" * 60)
    
    zdt_problems = get_zdt_problems()
    for name, problem in zdt_problems.items():
        metrics = run_experiment(name, problem, results_dir)
        all_metrics.extend(metrics)
    
    # Test DTLZ problems
    print("\n" + "=" * 60)
    print("DTLZ BENCHMARK SUITE")
    print("=" * 60)
    
    dtlz_problems = get_dtlz_problems()
    for name, problem in dtlz_problems.items():
        metrics = run_experiment(name, problem, results_dir)
        all_metrics.extend(metrics)
    
    # Test WFG problems
    print("\n" + "=" * 60)
    print("WFG BENCHMARK SUITE")
    print("=" * 60)
    
    wfg_problems = get_wfg_problems()
    for name, problem in wfg_problems.items():
        metrics = run_experiment(name, problem, results_dir)
        all_metrics.extend(metrics)
    
    # Save summary
    summary_df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(results_dir, "adaptive_decomposition_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n✅ Adaptive decomposition experiments completed!")
    print(f"📊 Summary saved to: {summary_path}")
    
    # Print final ranking
    print("\n" + "=" * 60)
    print("ALGORITHM RANKING (Average HV)")
    print("=" * 60)
    
    ranking = summary_df.groupby('Algorithm')['HV'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    print(ranking.round(4))


if __name__ == "__main__":
    main()