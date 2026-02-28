# src/algorithms/decomposition.py
"""
Adaptive decomposition that switches between PBI and Chebyshev.
"""

from pymoo.decomposition.pbi import PBI
from pymoo.decomposition.tchebicheff import Tchebicheff
from src.utils.config import PBI_THETA, TRANSITION_THRESHOLD, WINDOW_SIZE


class AdaptiveHybridDecomposition:
    """
    Implements adaptive transition between PBI and Chebyshev decomposition.
    
    Uses PBI in early stages (exploration) and switches to Chebyshev
    in later stages (convergence) based on HV improvement stagnation.
    """
    
    def __init__(self, theta=PBI_THETA, transition_threshold=TRANSITION_THRESHOLD, 
                 window_size=WINDOW_SIZE, transition_point=0.5):
        """
        Parameters
        ----------
        theta : float
            Penalty parameter for PBI
        transition_threshold : float
            Minimum relative improvement to stay in PBI mode
        window_size : int
            Number of generations to look back for improvement
        transition_point : float
            Fraction of generations after which transition is allowed (default: 0.5)
        """
        self.pbi = PBI(theta=theta)
        self.tcheb = Tchebicheff()
        self.window_size = window_size
        self.transition_threshold = transition_threshold
        self.transition_point = transition_point
        self.hv_history = []
        self.using_pbi = True
        self.current_gen = 0
        self.total_gen = None
    
    def update_convergence_metric(self, hv_value, current_gen, total_gen):
        """Update convergence metric and check for transition condition."""
        self.current_gen = current_gen
        self.total_gen = total_gen
        self.hv_history.append(hv_value)
        
        # Only consider transition after window_size generations
        if len(self.hv_history) < self.window_size + 1:
            return
        
        # Calculate relative improvement
        current_hv = self.hv_history[-1]
        previous_hv = self.hv_history[-self.window_size-1]
        relative_improvement = (current_hv - previous_hv) / previous_hv if previous_hv > 0 else 0
        
        # Check transition conditions
        progress_ratio = current_gen / total_gen
        
        # Transition based on either stagnation or reaching transition point
        if (relative_improvement < self.transition_threshold and 
            progress_ratio >= self.transition_point) and self.using_pbi:
            self.using_pbi = False
    
    def do(self, F, weights, **kwargs):
        """Apply current decomposition method."""
        if self.using_pbi:
            return self.pbi.do(F, weights, **kwargs)
        else:
            return self.tcheb.do(F, weights, **kwargs)
    
    def reset(self):
        """Reset decomposition state (useful for new runs)."""
        self.hv_history = []
        self.using_pbi = True
        self.current_gen = 0