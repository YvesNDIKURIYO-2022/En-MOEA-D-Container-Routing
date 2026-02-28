# src/algorithms/en_moead.py
"""
Enhanced MOEA/D with unified robust objectives and adaptive decomposition.
"""

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.indicators.hv import Hypervolume

from src.utils.config import (
    CROSSOVER_PROB, DISTRIBUTION_INDEX, N_GEN
)
from src.algorithms.decomposition import AdaptiveHybridDecomposition
from src.algorithms.robustness import MonteCarloScenarioGenerator, UnifiedRobustObjective


class EnhancedMOEAD(MOEAD):
    """
    Enhanced MOEA/D with unified robust objectives and adaptive decomposition.
    
    This algorithm integrates three key innovations:
    1. Adaptive decomposition switching between PBI and Chebyshev
    2. Monte Carlo scenario analysis for uncertainty quantification
    3. Mean-variance risk control strategy
    """
    
    def __init__(self, ref_dirs, base_problem, 
                 n_neighbors=20,
                 prob_neighbor_mating=0.9,
                 **kwargs):
        """
        Parameters
        ----------
        ref_dirs : array-like
            Reference directions for decomposition
        base_problem : pymoo.Problem
            Base optimization problem
        n_neighbors : int
            Number of neighbors for mating
        prob_neighbor_mating : float
            Probability of mating with neighbors
        """
        if base_problem is None:
            raise ValueError("base_problem must be provided for EnhancedMOEAD")
        
        # Store base problem
        self.base_problem = base_problem
        
        # Initialize components
        self.scenario_generator = MonteCarloScenarioGenerator()
        self.robust_evaluator = UnifiedRobustObjective(
            base_problem, self.scenario_generator)
        self.adaptive_decomposition = AdaptiveHybridDecomposition()
        
        # Initialize MOEA/D with adaptive decomposition
        super().__init__(
            ref_dirs=ref_dirs,
            decomposition=self.adaptive_decomposition,
            n_neighbors=n_neighbors,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=CROSSOVER_PROB, eta=DISTRIBUTION_INDEX),
            mutation=PM(eta=DISTRIBUTION_INDEX),
            prob_neighbor_mating=prob_neighbor_mating,
            **kwargs
        )
        
        self.hv_calculator = None
        self.reference_point = None
        self.initial_neighbors = n_neighbors
        self.final_neighbors = max(10, n_neighbors // 2)
    
    def _evaluate(self, X, out, *args, **kwargs):
        """Override evaluation to use robust objectives with optimal weights."""
        F_robust = self.robust_evaluator.evaluate_robust_objectives(X)
        out["F"] = F_robust
    
    def _next(self):
        """Override next generation with adaptive mechanisms."""
        # Update convergence metric for decomposition transition
        if hasattr(self, 'pop') and len(self.pop) > 0 and self.reference_point is not None:
            current_gen = self.n_gen
            F = self.pop.get("F")
            if F is not None and len(F) > 0:
                hv = Hypervolume(ref_point=self.reference_point).do(F)
                self.adaptive_decomposition.update_convergence_metric(
                    hv, current_gen, N_GEN)
        
        # Dynamically adjust neighborhood size
        progress = self.n_gen / N_GEN
        self.n_neighbors = int(self.initial_neighbors - 
                               (self.initial_neighbors - self.final_neighbors) * progress)
        
        return super()._next()
    
    def reset(self):
        """Reset algorithm state (useful for new runs)."""
        self.adaptive_decomposition.reset()
        self.scenario_generator.reset()
        self.n_gen = 0