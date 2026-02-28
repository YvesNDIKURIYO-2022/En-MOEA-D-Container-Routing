#!/usr/bin/env python
# experiments/run_wfg_benchmarks.py
"""
Experiment 6: WFG Benchmark Suites
Evaluates all algorithms on WFG1-WFG9 test problems.
"""

import os
import time
import numpy as np
import pandas as pd
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from src.utils.config import (
    MAX_EVALUATIONS, POP_SIZE, N_GEN, SEED,
    N_VAR_WFG, N_OBJ_WFG, REF_POINT_WFG
)
from src.problems.benchmark import get_wfg_problems
from src.algorithms.variants import get_all_algorithms
from src.metrics.performance import compute_all_metrics, get_reference_point
from src.visualization.plotting import plot_pareto_fronts


def run_wfg_problem(problem_name, problem, algorithms, results_dir):
    """Run all algorithms on a single WFG problem."""
    print(f"\n{'='*60}")
    print(f"Running on {problem_name}")
    print(f"{'='*60}")
    
    n_obj = problem.n_obj
    ref_point = get_reference_point(problem_name)
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
    
    # Get algorithm instances for this problem
    algo_instances = {}
    for name, algo_func in algorithms.items():
        if name == "En-MOEA/D" or name == "Hybrid-Adaptive":
            algo_instances[name] = algo_func(problem)
        else:
            algo_instances[name] = algo_func(n_obj)
    
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
    """Main execution for WFG benchmark suites."""
    print("=" * 80)
    print("WFG BENCHMARK SUITES")
    print("=" * 80)
    print(f"Parameters: n_var={N_VAR_WFG}, n_obj={N_OBJ_WFG}")
    print(f"Max Evaluations={MAX_EVALUATIONS}, Population={POP_SIZE}")
    
    # Create results directory
    results_dir = "results/wfg_benchmarks"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get problems and algorithms
    problems = get_wfg_problems(n_var=N_VAR_WFG, n_obj=N_OBJ_WFG)
    
    # Get algorithm functions
    from src.algorithms.variants import (
        get_moead_algorithm, get_moead_de_algorithm,
        get_moead_awa_algorithm, get_moead_stm_algorithm
    )
    from src.algorithms.en_moead import EnhancedMOEAD
    
    algorithms = {
        "MOEA/D": lambda n_obj: get_moead_algorithm(
            get_reference_directions("das-dennis", n_obj, n_partitions=12)),
        "MOEA/D-DE": lambda n_obj: get_moead_de_algorithm(
            get_reference_directions("das-dennis", n_obj, n_partitions=12)),
        "MOEA/D-AWA": lambda n_obj: get_moead_awa_algorithm(
            get_reference_directions("das-dennis", n_obj, n_partitions=12)),
        "MOEA/D-STM": lambda n_obj: get_moead_stm_algorithm(
            get_reference_directions("das-dennis", n_obj, n_partitions=12)),
        "En-MOEA/D": lambda problem: EnhancedMOEAD(
            get_reference_directions("das-dennis", problem.n_obj, n_partitions=12),
            base_problem=problem
        )
    }
    
    all_metrics = []
    
    # Run each WFG problem
    for name, problem in problems.items():
        metrics = run_wfg_problem(name, problem, algorithms, results_dir)
        all_metrics.extend(metrics)
    
    # Save summary
    summary_df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(results_dir, "wfg_benchmarks_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n✅ WFG benchmark suites completed!")
    print(f"📊 Summary saved to: {summary_path}")
    
    # Statistical summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY BY ALGORITHM")
    print("=" * 60)
    
    pivot_hv = summary_df.pivot_table(values='HV', index='Problem', columns='Algorithm', aggfunc='mean')
    print("\nAverage Hypervolume:")
    print(pivot_hv.round(4))
    
    pivot_igd = summary_df.pivot_table(values='IGD', index='Problem', columns='Algorithm', aggfunc='mean')
    print("\nAverage IGD:")
    print(pivot_igd.round(4))
    
    # Count best performances
    print("\n" + "=" * 60)
    print("BEST ALGORITHM COUNT")
    print("=" * 60)
    
    best_hv = summary_df.loc[summary_df.groupby('Problem')['HV'].idxmax()]
    hv_counts = best_hv['Algorithm'].value_counts()
    print("\nBest HV count:")
    print(hv_counts)
    
    best_igd = summary_df.loc[summary_df.groupby('Problem')['IGD'].idxmin()]
    igd_counts = best_igd['Algorithm'].value_counts()
    print("\nBest IGD count:")
    print(igd_counts)


if __name__ == "__main__":
    main()