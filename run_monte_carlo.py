#!/usr/bin/env python
# experiments/run_monte_carlo.py
"""
Experiment 4: Monte Carlo Simulation Validation
Compares MOEA/D vs MOEA/D-MCSS under uncertainty.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
import pandas as pd
from scipy.stats import ttest_rel

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.problem import Problem
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.indicators.hv import HV
from pymoo.util.ref_dirs import get_reference_directions

from src.utils.config import (
    POP_SIZE, MAX_EVALUATIONS, SEED, NUM_SCENARIOS,
    CROSSOVER_PROB, DISTRIBUTION_INDEX,
    REF_POINT_ZDT, REF_POINT_DTLZ
)


class RobustDTLZ2(Problem):
    """DTLZ2 with uncertainty injection."""
    
    def __init__(self, n_var=10, n_obj=3, perturbation=0.1, seed=SEED):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0, xu=1)
        self.perturbation = perturbation
        self._dtlz2 = get_problem("dtlz2", n_var=n_var, n_obj=n_obj)
        self.rng = np.random.default_rng(seed)

    def _evaluate(self, X, out, *args, **kwargs):
        out_temp = {}
        self._dtlz2._evaluate(X, out_temp, *args, **kwargs)
        mode_factor = np.where(X[:, 0] > 0.5, 1.2, 0.8)
        noise = mode_factor[:, None] * self.rng.standard_normal((X.shape[0], self.n_obj))
        out["F"] = out_temp["F"] * (1 + self.perturbation * noise) + 0.1 * self.rng.standard_normal((X.shape[0], self.n_obj))


class RobustZDT1(Problem):
    """ZDT1 with uncertainty injection."""
    
    def __init__(self, n_var=10, perturbation=0.1, seed=SEED):
        super().__init__(n_var=n_var, n_obj=2, xl=0, xu=1)
        self.perturbation = perturbation
        self._zdt1 = get_problem("zdt1", n_var=n_var)
        self.rng = np.random.default_rng(seed)

    def _evaluate(self, X, out, *args, **kwargs):
        out_temp = {}
        self._zdt1._evaluate(X, out_temp, *args, **kwargs)
        mode_factor = np.where(X[:, 0] > 0.5, 1.2, 0.8)
        noise = mode_factor[:, None] * self.rng.standard_normal((X.shape[0], 2))
        out["F"] = out_temp["F"] * (1 + self.perturbation * noise) + 0.1 * self.rng.standard_normal((X.shape[0], 2))


def evaluate_with_uncertainty(X, problem, n_samples):
    """Evaluate with Monte Carlo sampling."""
    F_samples = []
    for _ in range(n_samples):
        out = {}
        problem._evaluate(X, out)
        F_samples.append(out["F"])
    return np.mean(F_samples, axis=0), np.var(F_samples, axis=0)


class MOEAD_MCSS(MOEAD):
    """MOEA/D with Monte Carlo scenario sampling."""
    
    def __init__(self, n_samples=100, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.rng = np.random.default_rng(SEED)

    def _evaluate(self, X, out, *args, **kwargs):
        F_mean, F_var = evaluate_with_uncertainty(X, self.problem, self.n_samples)
        robustness = 1 + 0.2 * np.tanh(F_var / 0.2)
        out["F"] = F_mean * robustness + 0.05 * self.rng.standard_normal(F_mean.shape)


def run_experiment(config):
    """Run Monte Carlo validation experiment."""
    
    print("\n=== Monte Carlo Simulation Validation ===")
    print(f"Problem Type: {config['problem_type']}")
    print(f"Number of Variables: {config['n_var']}")
    print(f"Population Size: {config['population_size']}")
    print(f"Maximum Evaluations: {config['max_evaluations']}")
    print(f"Number of Scenarios: {config['n_samples']}")
    print("=" * 60)

    # Create reference directions
    n_obj = 2 if config['problem_type'] == 'zdt' else 3
    ref_dirs = get_reference_directions("uniform", n_obj, n_partitions=config['population_size']-1)
    
    # Initialize problem
    if config['problem_type'] == 'zdt':
        problem = RobustZDT1(config['n_var'], config['perturbation'])
        ref_point = np.array(REF_POINT_ZDT)
    else:
        problem = RobustDTLZ2(config['n_var'], n_obj, config['perturbation'])
        ref_point = np.array(REF_POINT_DTLZ)

    # Algorithms
    algorithms = {
        "MOEA/D": MOEAD(
            ref_dirs=ref_dirs,
            sampling=LHS(),
            crossover=SBX(prob=config['crossover_prob'], eta=config['distribution_index']),
            mutation=PM(eta=config['distribution_index'])
        ),
        "MOEA/D-MCSS": MOEAD_MCSS(
            ref_dirs=ref_dirs,
            n_samples=config['n_samples'],
            sampling=LHS(),
            crossover=SBX(prob=config['crossover_prob'], eta=config['distribution_index']),
            mutation=PM(eta=config['distribution_index'])
        )
    }

    metrics = {name: {m: [] for m in ["HV", "Mean", "Robustness", "Time", "Stability"]} 
               for name in algorithms}
    
    n_gen = config['max_evaluations'] // config['population_size']

    for run in range(config['n_runs']):
        seed = SEED + run
        print(f"\nRun {run + 1}/{config['n_runs']} (Seed: {seed})")

        for name, algorithm in algorithms.items():
            start_time = time.time()
            res = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
            runtime = time.time() - start_time

            # Evaluate with multiple samples
            hv_vals = []
            F_test = []
            
            for _ in range(config['eval_samples']):
                out = {}
                problem._evaluate(res.X, out)
                F_test.append(out["F"])
                hv_vals.append(HV(ref_point=ref_point)(np.atleast_2d(out["F"])))

            F_test = np.array(F_test)

            metrics[name]["HV"].append(HV(ref_point=ref_point)(res.F))
            metrics[name]["Mean"].append(np.mean(F_test))
            metrics[name]["Robustness"].append(np.mean(np.var(F_test, axis=1)))
            metrics[name]["Time"].append(runtime)
            metrics[name]["Stability"].append(np.std(hv_vals))

            print(f"{name:13s} | HV: {metrics[name]['HV'][-1]:.4f} ±{metrics[name]['Stability'][-1]:.4f} | "
                  f"Time: {runtime:.2f}s")

    return metrics, problem


def plot_results(metrics, config, results_dir):
    """Plot comparison results."""
    plt.figure(figsize=(15, 5))
    
    # Bar chart comparison
    plt.subplot(1, 3, 1)
    metric_names = ["HV", "Robustness", "Stability"]
    x = np.arange(len(metric_names))
    width = 0.35
    
    for i, name in enumerate(["MOEA/D", "MOEA/D-MCSS"]):
        means = [np.mean(metrics[name][m]) for m in metric_names]
        stds = [np.std(metrics[name][m]) for m in metric_names]
        plt.bar(x + i * width, means, width, yerr=stds, capsize=5, label=name)
    
    plt.ylabel("Metric Value")
    plt.title(f"Performance Comparison - {config['problem_type'].upper()}")
    plt.xticks(x + width / 2, metric_names)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # T-test results
    plt.subplot(1, 3, 2)
    hv1 = metrics["MOEA/D"]["HV"]
    hv2 = metrics["MOEA/D-MCSS"]["HV"]
    t_stat, p_val = ttest_rel(hv1, hv2)
    
    plt.bar(['MOEA/D', 'MOEA/D-MCSS'], [np.mean(hv1), np.mean(hv2)], 
            yerr=[np.std(hv1), np.std(hv2)], capsize=5)
    plt.ylabel("Hypervolume")
    plt.title(f"T-test: p={p_val:.4f}\n{'Significant' if p_val < 0.05 else 'Not Significant'}")
    plt.grid(alpha=0.3)
    
    # Runtime comparison
    plt.subplot(1, 3, 3)
    runtime1 = metrics["MOEA/D"]["Time"]
    runtime2 = metrics["MOEA/D-MCSS"]["Time"]
    plt.bar(['MOEA/D', 'MOEA/D-MCSS'], [np.mean(runtime1), np.mean(runtime2)],
            yerr=[np.std(runtime1), np.std(runtime2)], capsize=5)
    plt.ylabel("Runtime (seconds)")
    plt.title("Computational Cost Comparison")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"monte_carlo_comparison_{config['problem_type']}.png"), dpi=300)
    plt.show()


def main():
    """Main execution for Monte Carlo validation."""
    print("=" * 80)
    print("MONTE CARLO SIMULATION VALIDATION")
    print("=" * 80)
    
    # Configuration
    config = {
        "n_var": 10,
        "population_size": POP_SIZE,
        "max_evaluations": MAX_EVALUATIONS,
        "n_samples": NUM_SCENARIOS,
        "n_runs": 10,
        "eval_samples": 100,
        "perturbation": 0.1,
        "crossover_prob": CROSSOVER_PROB,
        "distribution_index": DISTRIBUTION_INDEX,
    }
    
    # Create results directory
    results_dir = "results/monte_carlo"
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    
    # Run for ZDT
    config['problem_type'] = 'zdt'
    metrics_zdt, problem_zdt = run_experiment(config)
    all_results['zdt'] = metrics_zdt
    plot_results(metrics_zdt, config, results_dir)
    
    # Run for DTLZ
    config['problem_type'] = 'dtlz'
    metrics_dtlz, problem_dtlz = run_experiment(config)
    all_results['dtlz'] = metrics_dtlz
    plot_results(metrics_dtlz, config, results_dir)
    
    # Save results
    for ptype, metrics in all_results.items():
        results_dict = {}
        for algo_name, algo_metrics in metrics.items():
            results_dict[algo_name] = {
                metric: (np.mean(values), np.std(values))
                for metric, values in algo_metrics.items()
            }
        
        with open(os.path.join(results_dir, f"results_{ptype}.json"), 'w') as f:
            json.dump(results_dict, f, indent=4)
    
    print(f"\n✅ Monte Carlo validation completed!")
    print(f"📊 Results saved to: {results_dir}")


if __name__ == "__main__":
    main()