# src/metrics/performance.py
"""
Performance metrics: Hypervolume, IGD, Spread, and related calculations.
"""

import numpy as np
from pymoo.indicators.hv import HV as PymooHV
from pymoo.indicators.igd import IGD as PymooIGD
from src.utils.config import REF_POINT_ZDT, REF_POINT_DTLZ, REF_POINT_WFG


class HV:
    """Hypervolume indicator wrapper."""
    
    def __init__(self, ref_point):
        self.ref_point = ref_point
        self.indicator = PymooHV(ref_point=ref_point)
    
    def __call__(self, F):
        if F is None or len(F) == 0:
            return np.nan
        try:
            return self.indicator.do(F)
        except:
            return np.nan


class IGD:
    """Inverted Generational Distance indicator wrapper."""
    
    def __init__(self, pf):
        self.pf = pf
        if pf is not None and len(pf) > 0:
            self.indicator = PymooIGD(pf=pf)
        else:
            self.indicator = None
    
    def __call__(self, F):
        if self.indicator is None or F is None or len(F) == 0:
            return np.nan
        try:
            return self.indicator.do(F)
        except:
            return np.nan


def calculate_spread(F):
    """
    Calculate the Spread (Δ) metric for solution diversity.
    
    Parameters
    ----------
    F : array-like
        Pareto front solutions (shape: n_solutions × n_obj)
    
    Returns
    -------
    float
        Spread metric (lower is better, indicates better distribution)
    """
    if len(F) < 2:
        return np.nan
    
    try:
        F = F[np.argsort(F[:, 0])]
        dists = np.linalg.norm(np.diff(F, axis=0), axis=1)
        mean_d = np.mean(dists)
        
        if mean_d < 1e-10:
            return np.nan
        
        d_f = np.linalg.norm(F[0] - F[-1])
        sum_diff = np.sum(np.abs(dists - mean_d))
        delta = (sum_diff + d_f) / (mean_d * len(F) + d_f)
        return delta
    except:
        return np.nan


def calculate_mad(F):
    """
    Calculate Median Absolute Deviation for solution stability.
    
    Parameters
    ----------
    F : array-like
        Pareto front solutions (shape: n_solutions × n_obj)
    
    Returns
    -------
    float or array
        MAD value(s)
    """
    if F is None or len(F) == 0:
        return np.nan
    
    median = np.median(F, axis=0)
    mad = np.median(np.abs(F - median), axis=0)
    return mad


def normalize_front(F, pf):
    """
    Normalize objectives using true Pareto front bounds.
    
    Parameters
    ----------
    F : array-like
        Solutions to normalize
    pf : array-like
        True Pareto front
    
    Returns
    -------
    array
        Normalized solutions
    """
    if pf is None or len(pf) == 0:
        return F
    
    try:
        ideal = np.min(pf, axis=0)
        nadir = np.max(pf, axis=0)
        range_vals = nadir - ideal
        range_vals[range_vals < 1e-10] = 1
        return (F - ideal) / range_vals
    except:
        return F


def get_reference_point(problem_name):
    """
    Get appropriate reference point based on problem type.
    
    Parameters
    ----------
    problem_name : str
        Name of the problem (e.g., 'ZDT1', 'DTLZ2', 'WFG4')
    
    Returns
    -------
    array
        Reference point
    """
    if "ZDT" in problem_name:
        return REF_POINT_ZDT
    elif "WFG" in problem_name:
        return REF_POINT_WFG
    else:
        return REF_POINT_DTLZ


def compute_all_metrics(F, problem, problem_name, pf=None):
    """
    Compute all performance metrics for a given Pareto front.
    
    Parameters
    ----------
    F : array-like
        Obtained Pareto front
    problem : pymoo.Problem
        The optimization problem
    problem_name : str
        Name of the problem
    pf : array-like, optional
        True Pareto front
    
    Returns
    -------
    dict
        Dictionary of metric_name -> value
    """
    ref_point = get_reference_point(problem_name)
    
    if pf is None:
        try:
            pf = problem.pareto_front()
        except:
            pf = None
    
    # Normalize if possible
    norm_F = normalize_front(F, pf) if pf is not None else F
    
    # Calculate metrics
    hv = HV(ref_point)(norm_F) if norm_F is not None else np.nan
    igd = IGD(pf)(norm_F) if pf is not None else np.nan
    spread = calculate_spread(norm_F) if norm_F is not None else np.nan
    mad = calculate_mad(norm_F) if norm_F is not None else np.nan
    
    return {
        'HV': hv if not np.isnan(hv) else 0,
        'IGD': igd if not np.isnan(igd) else float('inf'),
        'Spread': spread if not np.isnan(spread) else float('inf'),
        'MAD': mad
    }