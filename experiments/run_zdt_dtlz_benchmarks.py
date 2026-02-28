#!/usr/bin/env python
# experiments/run_zdt_dtlz_benchmarks.py
"""
Experiment 7: ZDT and DTLZ Benchmark Suites
Evaluates all algorithms on ZDT1-ZDT6 and DTLZ1-DTLZ7.
"""

import os
import time
import numpy as np
import pandas as pd
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from src.utils.config import (
    MAX_EVALUATIONS, POP_SIZE, N_GEN, SEED,
    N_VAR_ZDT, N_VAR_DTLZ, N_OBJ_DTLZ
)
from src.problems.benchmark import get_zdt_problems, get_dtlz_problems
from src.algorithms.variants import get_all_algorithms
from src.metrics.performance import compute_all_metrics, get_reference_point
from src.visualization.plotting import plot_pareto_fronts


def run_benchmark_problem(problem_name, problem, algorithms, results_dir):
    """Run all algorithms on a single benchmark problem."""
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
    """Main execution for ZDT and DTLZ benchmark suites."""
    print("=" * 80)
    print("ZDT AND DTLZ BENCHMARK SUITES")
    print("=" * 80)
    print(f"Parameters: Max Evaluations={MAX_EVALUATIONS}, Population={POP_SIZE}")
    
    # Create results directory
    results_dir = "results/zdt_dtlz_benchmarks"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get problems
    zdt_problems = get_zdt_problems()
    dtlz_problems = get_dtlz_problems(n_var=N_VAR_DTLZ, n_obj=N_OBJ_DTLZ)
    all_problems = {**zdt_problems, **dtlz_problems}
    
    # Get algorithm functions
    from src.algorithms.variants import (
        get_moead_algorithm, get_moead_de_algorithm,
        get_moead_awa_algorithm, get_moead_stm_algorithm,
        get_pbi_algorithm, get_tchebicheff_algorithm
    )
    from src.algorithms.en_moead import EnhancedMOEAD
    
    algorithms = {
        "PBI-only": lambda n_obj: get_pbi_algorithm(
            get_reference_directions("das-dennis", n_obj, n_partitions=12)),
        "Tchebicheff-only": lambda n_obj: get_tchebicheff_algorithm(
            get_reference_directions("das-dennis", n_obj, n_partitions=12)),
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
    
    # Run ZDT problems
    print("\n" + "=" * 60)
    print("ZDT BENCHMARK SUITE")
    print("=" * 60)
    
    for name, problem in zdt_problems.items():
        metrics = run_benchmark_problem(name, problem, algorithms, results_dir)
        all_metrics.extend(metrics)
    
    # Run DTLZ problems
    print("\n" + "=" * 60)
    print("DTLZ BENCHMARK SUITE")
    print("=" * 60)
    
    for name, problem in dtlz_problems.items():
        metrics = run_benchmark_problem(name, problem, algorithms, results_dir)
        all_metrics.extend(metrics)
    
    # Save summary
    summary_df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(results_dir, "zdt_dtlz_benchmarks_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n✅ ZDT and DTLZ benchmark suites completed!")
    print(f"📊 Summary saved to: {summary_path}")
    
    # Statistical summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY BY PROBLEM TYPE")
    print("=" * 60)
    
    # Separate ZDT and DTLZ results
    zdt_results = summary_df[summary_df['Problem'].str.contains('ZDT')]
    dtlz_results = summary_df[summary_df['Problem'].str.contains('DTLZ')]
    
    print("\nZDT Problems - Average HV by Algorithm:")
    zdt_pivot = zdt_results.pivot_table(values='HV', index='Problem', columns='Algorithm', aggfunc='mean')
    print(zdt_pivot.round(4))
    
    print("\nDTLZ Problems - Average HV by Algorithm:")
    dtlz_pivot = dtlz_results.pivot_table(values='HV', index='Problem', columns='Algorithm', aggfunc='mean')
    print(dtlz_pivot.round(4))
    
    # Best algorithm counts
    print("\n" + "=" * 60)
    print("BEST ALGORITHM COUNT")
    print("=" * 60)
    
    best_hv = summary_df.loc[summary_df.groupby('Problem')['HV'].idxmax()]
    hv_counts = best_hv['Algorithm'].value_counts()
    print("\nBest HV count across all problems:")
    print(hv_counts)


if __name__ == "__main__":
    main()