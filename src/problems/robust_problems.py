# src/problems/robust_problems.py
"""
Robust test problems: GFunction, Ishigami, OakleyOHagan, Borehole, etc.
"""

import numpy as np
from pymoo.core.problem import Problem
from src.utils.config import NUM_SCENARIOS, SEED
from src.algorithms.robustness import mean_variance_robustness


class RobustProblemWrapper(Problem):
    """
    Base wrapper class that adds Monte Carlo robustness to any problem.
    """
    
    def __init__(self, base_problem, n_scenarios=NUM_SCENARIOS, seed=SEED):
        super().__init__(n_var=base_problem.n_var, 
                        n_obj=base_problem.n_obj,
                        xl=base_problem.xl, 
                        xu=base_problem.xu)
        self.base_problem = base_problem
        self.n_scenarios = n_scenarios
        self.rng = np.random.RandomState(seed)
        self.scenarios = self.generate_scenarios()
    
    def generate_scenarios(self):
        """Generate Monte Carlo scenarios for uncertainty."""
        return self.rng.uniform(0, 1, (self.n_scenarios, self.n_var))
    
    def _evaluate(self, X, out, *args, **kwargs):
        robust_objs = []
        for x in X:
            scenario_results = [self.objective_functions(x, scenario) 
                               for scenario in self.scenarios]
            scenario_results = np.array(scenario_results)
            # Use the imported mean_variance_robustness
            from src.utils.helpers import mean_variance_robustness
            robust_obj = mean_variance_robustness(scenario_results)
            robust_objs.append(robust_obj)
        out["F"] = np.array(robust_objs)
    
    def objective_functions(self, x, scenario):
        """To be overridden by subclasses."""
        raise NotImplementedError
    
    def pareto_front(self, **kwargs):
        """Generate placeholder Pareto front."""
        return self.rng.rand(100, self.n_obj)


class BoreholeFunction(RobustProblemWrapper):
    """Borehole function robust problem."""
    
    def __init__(self, n_scenarios=NUM_SCENARIOS, seed=SEED):
        super().__init__(None, n_scenarios, seed)
        self.n_var = 2
        self.n_obj = 3
        self.xl = np.array([0.05, 100])
        self.xu = np.array([0.15, 50000])
    
    def objective_functions(self, x, scenario):
        rw, Tu = x + scenario[:2]
        f1 = 2 * np.pi * Tu * (rw**2)
        f2 = 2 * np.pi * (Tu**2) * rw
        f3 = 2 * np.pi * Tu * rw
        return np.array([f1, f2, f3])


class IshigamiFunction(RobustProblemWrapper):
    """Ishigami function robust problem."""
    
    def __init__(self, n_scenarios=NUM_SCENARIOS, seed=SEED):
        super().__init__(None, n_scenarios, seed)
        self.n_var = 3
        self.n_obj = 3
        self.xl = -np.pi
        self.xu = np.pi
    
    def objective_functions(self, x, scenario):
        a, b, c = x + scenario[:3]
        f1 = np.sin(a) + 7 * np.sin(b)**2
        f2 = 0.1 * (c**4) * np.sin(a)
        f3 = np.sin(a + b + c)
        return np.array([f1, f2, f3])


class GFunction(RobustProblemWrapper):
    """G-Function robust problem."""
    
    def __init__(self, n_scenarios=NUM_SCENARIOS, seed=SEED):
        super().__init__(None, n_scenarios, seed)
        self.n_var = 9
        self.n_obj = 3
        self.xl = 0
        self.xu = 1
        self.a = np.array([0, 1, 4, 9, 16, 25, 36, 49, 64])
    
    def objective_functions(self, x, scenario):
        g = np.sum((x + scenario[:9] - self.a)**2)
        f1 = g * (1 + np.sin(3 * np.pi * x[0]))
        f2 = g * (1 + np.sin(3 * np.pi * x[1]))
        f3 = g * (1 + np.sin(3 * np.pi * x[2]))
        return np.array([f1, f2, f3])


class OakleyOHagan1DFunction(RobustProblemWrapper):
    """Oakley & O'Hagan 1D robust problem."""
    
    def __init__(self, n_scenarios=NUM_SCENARIOS, seed=SEED):
        super().__init__(None, n_scenarios, seed)
        self.n_var = 1
        self.n_obj = 3
        self.xl = 0
        self.xu = 1
    
    def objective_functions(self, x, scenario):
        f1 = np.sin(2 * np.pi * (x + scenario[0]))
        f2 = np.cos(2 * np.pi * (x + scenario[0]))
        f3 = f1 * f2
        return np.array([f1, f2, f3])


class OakleyOHagan2DFunction(RobustProblemWrapper):
    """Oakley & O'Hagan 2D robust problem."""
    
    def __init__(self, n_scenarios=NUM_SCENARIOS, seed=SEED):
        super().__init__(None, n_scenarios, seed)
        self.n_var = 2
        self.n_obj = 3
        self.xl = 0
        self.xu = 1
    
    def objective_functions(self, x, scenario):
        f1 = np.sin(2 * np.pi * (x[0] + scenario[0])) * np.cos(2 * np.pi * (x[1] + scenario[1]))
        f2 = np.cos(2 * np.pi * (x[0] + scenario[0])) * np.sin(2 * np.pi * (x[1] + scenario[1]))
        f3 = np.sin(2 * np.pi * (x[0] + scenario[0])) * np.sin(2 * np.pi * (x[1] + scenario[1]))
        return np.array([f1, f2, f3])


def get_all_robust_problems():
    """Get all robust test problems."""
    return {
        "Borehole": BoreholeFunction(),
        "Ishigami": IshigamiFunction(),
        "GFunction": GFunction(),
        "OakleyOHagan1D": OakleyOHagan1DFunction(),
        "OakleyOHagan2D": OakleyOHagan2DFunction(),
    }