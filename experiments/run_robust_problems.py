#!/usr/bin/env python
# experiments/run_robust_problems.py
"""
Experiment 5: Robust Test Problems Validation
Evaluates En-MOEA/D on GFunction, Ishigami, OakleyOHagan, etc.
"""

import os
import time
import numpy as np
import pandas as pd
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from src.utils.config import MAX_EVALUATIONS, POP_SIZE, N_GEN, SEED, REF_POINT_DTLZ
from src.problems.robust_problems import get_all_robust_problems
from src.algorithms.variants import get_all_algorithms
from src.metrics.performance import compute_all_metrics
from src.visualization.plotting import plot_pareto_fronts
from src.utils.helpers import save_dataframe


def run_robust_problem(problem_name, problem, algorithms, results_dir):
    """Run all algorithms on a single robust problem."""
    print(f"\n{'='*60}")
    print(f"Running on {problem_name}")
    print(f"{'='*60}")
    
    n_obj = problem.n_obj
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
    
    # Get algorithm instances for this problem
    algo_instances = {name: algo(problem.n_obj) for name, algo in algorithms.items()}
    
    results = {}
    metrics_list = []
    
    for name, algorithm in algo_instances.items():
        print(f"\n  Algorithm: {name}")
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
                
                metrics_list.append({
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
    if results:
        plot_pareto_fronts({problem_name: results}, problem_name,
                           save_path=os.path.join(results_dir, f"pareto_{problem_name}.png"))
    
    return metrics_list


def main():
    """Main execution for robust test problems."""
    print("=" * 80)
    print("ROBUST TEST PROBLEMS VALIDATION")
    print("=" * 80)
    print(f"Parameters: Max Evaluations={MAX_EVALUATIONS}, Population={POP_SIZE}")
    
    # Create results directory
    results_dir = "results/robust_problems"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get problems and algorithms
    problems = get_all_robust_problems()
    algorithms = {
        "MOEA/D": lambda n_obj: get_all_algorithms(None, n_obj, with_adaptive=False)["MOEA/D"],
        "MOEA/D-DE": lambda n_obj: get_all_algorithms(None, n_obj, with_adaptive=False)["MOEA/D-DE"],
        "MOEA/D-AWA": lambda n_obj: get_all_algorithms(None, n_obj, with_adaptive=False)["MOEA/D-AWA"],
        "MOEA/D-STM": lambda n_obj: get_all_algorithms(None, n_obj, with_adaptive=False)["MOEA/D-STM"],
        "En-MOEA/D": lambda n_obj: get_all_algorithms(problems[list(problems.keys())[0]], n_obj, with_adaptive=True)["En-MOEA/D"]
    }
    
    all_metrics = []
    
    # Run each problem
    for name, problem in problems.items():
        metrics = run_robust_problem(name, problem, algorithms, results_dir)
        all_metrics.extend(metrics)
    
    # Save summary
    summary_df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(results_dir, "robust_problems_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n✅ Robust problems validation completed!")
    print(f"📊 Summary saved to: {summary_path}")
    
    # Print final ranking
    print("\n" + "=" * 60)
    print("ALGORITHM RANKING (Average HV)")
    print("=" * 60)
    
    ranking = summary_df.groupby('Algorithm')['HV'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    print(ranking.round(4))
    
    # Best algorithm by problem
    print("\n" + "=" * 60)
    print("BEST ALGORITHM PER PROBLEM (by HV)")
    print("=" * 60)
    
    best_per_problem = summary_df.loc[summary_df.groupby('Problem')['HV'].idxmax()]
    print(best_per_problem[['Problem', 'Algorithm', 'HV']].to_string(index=False))


if __name__ == "__main__":
    main()