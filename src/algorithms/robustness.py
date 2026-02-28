# src/algorithms/robustness.py
"""
Robustness components: Monte Carlo scenario generation and unified robust objective.
"""

import numpy as np
from scipy.stats import truncnorm
from src.utils.config import NUM_SCENARIOS, SEED, OMEGA, GAMMA, GAMMA2, GAMMA3


class MonteCarloScenarioGenerator:
    """Generates Monte Carlo scenarios for uncertainty modeling with optimal parameters."""
    
    def __init__(self, num_scenarios=NUM_SCENARIOS, seed=SEED):
        self.num_scenarios = num_scenarios
        self.rng = np.random.RandomState(seed)
        self.scenarios = {}
    
    def generate_scenarios(self, problem_name, n_vars, n_objs):
        """Generate scenarios for uncertain parameters."""
        key = f"{problem_name}_{n_vars}_{n_objs}"
        if key not in self.scenarios:
            scenarios = []
            for _ in range(self.num_scenarios):
                scenario = {
                    'var_perturb': truncnorm.rvs(-2, 2, loc=1.0, scale=0.1, 
                                                random_state=self.rng, size=n_vars),
                    'obj_perturb': truncnorm.rvs(-2, 2, loc=1.0, scale=0.1,
                                                random_state=self.rng, size=n_objs)
                }
                scenarios.append(scenario)
            self.scenarios[key] = scenarios
        return self.scenarios[key]
    
    def reset(self, seed=None):
        """Reset generator with optional new seed."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.scenarios = {}


class UnifiedRobustObjective:
    """
    Implements the unified robust objective formulation with OPTIMAL weights.
    
    Combines:
    - Expected performance (mean)
    - Extreme event protection (worst-case)
    - Performance stability (variance)
    """
    
    def __init__(self, base_problem, scenario_generator):
        self.base_problem = base_problem
        self.scenario_generator = scenario_generator
        self.n_vars = base_problem.n_var
        self.n_objs = base_problem.n_obj
        self.scenarios = None
        self.problem_name = type(base_problem).__name__
    
    def evaluate_robust_objectives(self, X):
        """
        Evaluate unified robust objectives for a population with optimal weights.
        
        Parameters
        ----------
        X : array-like
            Population of solutions (shape: n_pop × n_vars)
        
        Returns
        -------
        F_robust : array
            Robust objective values (shape: n_pop × n_objs)
        """
        if self.scenarios is None:
            self.scenarios = self.scenario_generator.generate_scenarios(
                self.problem_name, self.n_vars, self.n_objs)
        
        population_size = len(X)
        robust_objectives = np.zeros((population_size, self.n_objs))
        
        for i, individual in enumerate(X):
            # Evaluate across all scenarios
            scenario_values = []
            for scenario in self.scenarios:
                # Apply perturbations to create uncertain evaluation
                perturbed_X = individual * scenario['var_perturb']
                F = self.base_problem.evaluate(perturbed_X.reshape(1, -1))
                perturbed_F = F.flatten() * scenario['obj_perturb']
                scenario_values.append(perturbed_F)
            
            scenario_matrix = np.array(scenario_values)
            
            # Calculate robust objectives with OPTIMAL WEIGHTS
            for obj_idx in range(self.n_objs):
                obj_values = scenario_matrix[:, obj_idx]
                
                if obj_idx == 0:  # Cost-like objective (minimize)
                    expected_val = np.mean(obj_values)
                    worst_case = np.max(obj_values)
                    variance = np.var(obj_values)
                    robust_objectives[i, obj_idx] = (OMEGA * expected_val + 
                                                    (1 - OMEGA) * worst_case + 
                                                    GAMMA * variance)
                
                elif obj_idx == 1:  # Time-like objective (minimize)
                    expected_val = np.mean(obj_values)
                    variance = np.var(obj_values)
                    robust_objectives[i, obj_idx] = expected_val + GAMMA2 * variance
                
                else:  # Reliability-like objective (maximize, converted to minimize)
                    expected_val = np.mean(obj_values)
                    variance = np.var(obj_values)
                    # Convert maximization to minimization by taking negative
                    robust_objectives[i, obj_idx] = -expected_val + GAMMA3 * variance
        
        return robust_objectives