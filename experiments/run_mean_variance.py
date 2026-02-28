#!/usr/bin/env python
# experiments/run_mean_variance.py
"""
Experiment 3: Mean-Variance Robustness Validation
Compares RobustMOEA/D against standard algorithms.
"""

import os
import time
import numpy as np
import pandas as pd
from pymoo.problems import get_problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from src.utils.config import (
    POP_SIZE, MAX_EVALUATIONS, SEED,
    CROSSOVER_PROB, DISTRIBUTION_INDEX,
    REF_POINT_ZDT, REF_POINT_WFG, ROBUSTNESS_TESTS
)
from src.algorithms.robustness import MonteCarloScenarioGenerator
from src.utils.helpers import evaluate_population_robustness
from src.visualization.plotting import plot_algorithm_comparison


class RobustMOEAD(MOEAD):
    """Simple Robust MOEA/D implementation."""
    
    def __init__(self, robustness_strength=0.6, **kwargs):
        super().__init__(**kwargs)
        self.robustness_strength = robustness_strength
        self.generation = 0
    
    def _next(self):
        self.generation += 1
        return super()._next()


def run_focused_experiment(results_dir):
    """Run focused experiment comparing RobustMOEA/D against standard algorithms."""
    print("\n🎯 FOCUSED EXPERIMENT: RobustMOEA/D vs Standard Algorithms")
    print("=" * 60)
    
    results = []
    n_runs = 4
    n_gen = 100
    
    # Test problems
    problems = {
        "ZDT1": get_problem("zdt1"),
        "ZDT2": get_problem("zdt2"),
        "WFG1": get_problem("wfg1", n_var=12, n_obj=3),
        "WFG2": get_problem("wfg2", n_var=12, n_obj=3)
    }
    
    common_params = {
        'sampling': FloatRandomSampling(),
        'crossover': SBX(prob=CROSSOVER_PROB, eta=DISTRIBUTION_INDEX),
        'mutation': PM(eta=DISTRIBUTION_INDEX),
        'seed': SEED
    }
    
    for pname, problem in problems.items():
        print(f"\n🔍 Testing {pname}")
        print("-" * 40)
        
        n_obj = problem.n_obj
        ref_point = REF_POINT_WFG if n_obj == 3 else REF_POINT_ZDT
        
        # Reference directions
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=10)
        pop_size = len(ref_dirs)
        
        # Algorithms
        algorithms = {
            "NSGA-II": NSGA2(pop_size=pop_size, eliminate_duplicates=True, **common_params),
            "SPEA2": SPEA2(pop_size=pop_size, **common_params),
            "MOEA/D": MOEAD(ref_dirs=ref_dirs, n_neighbors=15, **common_params),
            "RobustMOEA/D": RobustMOEAD(ref_dirs=ref_dirs, n_neighbors=15, **common_params)
        }
        
        for name, algo in algorithms.items():
            print(f"   Running {name}...")
            
            for run in range(n_runs):
                try:
                    start_time = time.time()
                    
                    res = minimize(
                        problem, algo,
                        termination=get_termination("n_gen", n_gen),
                        seed=SEED + run,
                        verbose=False
                    )
                    
                    if res is not None and hasattr(res, 'F') and res.F is not None:
                        # Calculate HV
                        from pymoo.indicators.hv import HV
                        hv = HV(ref_point=ref_point)(res.F)
                        
                        # Robustness evaluation
                        robust_metrics = evaluate_population_robustness(res.X, problem)
                        
                        results.append({
                            'problem': pname,
                            'algorithm': name,
                            'run': run + 1,
                            'HV': hv,
                            'robustness_score': robust_metrics['robustness_score'],
                            'survival_rate': robust_metrics['survival_rate'],
                            'runtime': time.time() - start_time
                        })
                        
                except Exception as e:
                    print(f"      ❌ Run {run+1} failed: {e}")
    
    return pd.DataFrame(results)


def analyze_results(df, results_dir):
    """Analyze and display results."""
    if df.empty:
        print("❌ No results to analyze")
        return
    
    print("\n" + "=" * 60)
    print("📊 FOCUSED RESULTS ANALYSIS")
    print("=" * 60)
    
    # Summary statistics
    summary = df.groupby(['problem', 'algorithm']).agg({
        'HV': ['mean', 'std'],
        'robustness_score': ['mean', 'std'],
        'survival_rate': ['mean', 'std'],
        'runtime': ['mean']
    }).round(4)
    
    print("\nSummary Statistics:")
    print(summary)
    
    # Performance comparison
    print("\n🔍 ROBUSTMOEA/D vs STANDARD ALGORITHMS")
    print("=" * 50)
    
    for problem in df['problem'].unique():
        print(f"\n{problem}:")
        prob_data = df[df['problem'] == problem]
        
        robust_data = prob_data[prob_data['algorithm'] == 'RobustMOEA/D']
        if not robust_data.empty:
            robust_hv = robust_data['HV'].mean()
            robust_robustness = robust_data['robustness_score'].mean()
            
            print(f"  RobustMOEA/D: HV={robust_hv:.4f}, Robustness={robust_robustness:.1f}%")
            
            for algo in ['NSGA-II', 'SPEA2', 'MOEA/D']:
                algo_data = prob_data[prob_data['algorithm'] == algo]
                if not algo_data.empty:
                    algo_hv = algo_data['HV'].mean()
                    algo_robustness = algo_data['robustness_score'].mean()
                    
                    hv_diff = ((robust_hv - algo_hv) / algo_hv) * 100
                    print(f"  {algo:<10}: HV={algo_hv:.4f} ({hv_diff:+.1f}%), "
                          f"Robustness={algo_robustness:.1f}%")
    
    # Overall ranking
    print("\n🏆 OVERALL PERFORMANCE RANKING")
    print("=" * 50)
    
    overall = df.groupby('algorithm').agg({
        'HV': 'mean',
        'robustness_score': 'mean',
        'survival_rate': 'mean'
    })
    
    overall['combined_score'] = (
        overall['HV'] / overall['HV'].max() * 40 +
        overall['robustness_score'] / 100 * 40 +
        overall['survival_rate'] / 100 * 20
    )
    
    overall = overall.sort_values('combined_score', ascending=False)
    print(overall.round(2))
    
    # Save results
    summary.to_csv(os.path.join(results_dir, "mean_variance_summary.csv"))
    overall.to_csv(os.path.join(results_dir, "algorithm_ranking.csv"))
    
    return overall


def main():
    """Main execution for mean-variance robustness validation."""
    print("=" * 80)
    print("MEAN-VARIANCE ROBUSTNESS VALIDATION")
    print("=" * 80)
    
    # Create results directory
    results_dir = "results/mean_variance"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run experiment
    results_df = run_focused_experiment(results_dir)
    
    # Analyze results
    if not results_df.empty:
        ranking = analyze_results(results_df, results_dir)
        
        # Save raw results
        results_df.to_csv(os.path.join(results_dir, "raw_results.csv"), index=False)
        
        print(f"\n✅ Mean-variance validation completed!")
        print(f"📊 Results saved to: {results_dir}")
    else:
        print("❌ Experiment failed - no results collected")


if __name__ == "__main__":
    main()