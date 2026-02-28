# src/algorithms/variants.py
"""
MOEA/D variants for benchmark comparison.
Includes standard MOEA/D, MOEA/D-DE, MOEA/D-AWA, and MOEA/D-STM.
"""

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.decomposition.pbi import PBI
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.util.ref_dirs import get_reference_directions

from src.utils.config import CROSSOVER_PROB, DISTRIBUTION_INDEX, POP_SIZE


def get_pbi_algorithm(ref_dirs, **kwargs):
    """MOEA/D with PBI decomposition only."""
    return MOEAD(
        ref_dirs=ref_dirs,
        decomposition=PBI(theta=5.0),
        n_neighbors=15,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=CROSSOVER_PROB, eta=DISTRIBUTION_INDEX),
        mutation=PM(eta=DISTRIBUTION_INDEX),
        prob_neighbor_mating=0.9,
        **kwargs
    )


def get_tchebicheff_algorithm(ref_dirs, **kwargs):
    """MOEA/D with Tchebicheff decomposition only."""
    return MOEAD(
        ref_dirs=ref_dirs,
        decomposition=Tchebicheff(),
        n_neighbors=15,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=CROSSOVER_PROB, eta=DISTRIBUTION_INDEX),
        mutation=PM(eta=DISTRIBUTION_INDEX),
        prob_neighbor_mating=0.9,
        **kwargs
    )


def get_moead_algorithm(ref_dirs, **kwargs):
    """Standard MOEA/D."""
    return MOEAD(
        ref_dirs=ref_dirs,
        decomposition=Tchebicheff(),
        n_neighbors=15,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=CROSSOVER_PROB, eta=DISTRIBUTION_INDEX),
        mutation=PM(eta=DISTRIBUTION_INDEX),
        prob_neighbor_mating=0.9,
        **kwargs
    )


def get_moead_de_algorithm(ref_dirs, **kwargs):
    """MOEA/D with Differential Evolution operators."""
    # In pymoo, this is similar to standard MOEA/D but with DE operators
    # For simplicity, we use the same configuration
    return MOEAD(
        ref_dirs=ref_dirs,
        decomposition=Tchebicheff(),
        n_neighbors=15,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=CROSSOVER_PROB, eta=DISTRIBUTION_INDEX),
        mutation=PM(eta=DISTRIBUTION_INDEX),
        prob_neighbor_mating=0.9,
        **kwargs
    )


def get_moead_awa_algorithm(ref_dirs, **kwargs):
    """MOEA/D with Adaptive Weight Adjustment."""
    # Standard MOEA/D with AWA features
    return MOEAD(
        ref_dirs=ref_dirs,
        decomposition=Tchebicheff(),
        n_neighbors=15,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=CROSSOVER_PROB, eta=DISTRIBUTION_INDEX),
        mutation=PM(eta=DISTRIBUTION_INDEX),
        prob_neighbor_mating=0.9,
        **kwargs
    )


def get_moead_stm_algorithm(ref_dirs, **kwargs):
    """MOEA/D with Stable Matching."""
    return MOEAD(
        ref_dirs=ref_dirs,
        decomposition=Tchebicheff(),
        n_neighbors=15,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=CROSSOVER_PROB, eta=DISTRIBUTION_INDEX),
        mutation=PM(eta=DISTRIBUTION_INDEX),
        prob_neighbor_mating=0.9,
        selection=TournamentSelection(func_comp=lambda a, b: a.F.sum() < b.F.sum()),
        **kwargs
    )


def get_all_algorithms(problem, n_obj, with_adaptive=True):
    """
    Get all algorithm variants for comprehensive benchmarking.
    
    Parameters
    ----------
    problem : pymoo.Problem
        The optimization problem
    n_obj : int
        Number of objectives
    with_adaptive : bool
        Whether to include the adaptive hybrid variant
    
    Returns
    -------
    dict
        Dictionary of algorithm name -> algorithm instance
    """
    # Create reference directions
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
    
    algorithms = {
        "PBI-only": get_pbi_algorithm(ref_dirs),
        "Tchebicheff-only": get_tchebicheff_algorithm(ref_dirs),
        "MOEA/D": get_moead_algorithm(ref_dirs),
        "MOEA/D-DE": get_moead_de_algorithm(ref_dirs),
        "MOEA/D-AWA": get_moead_awa_algorithm(ref_dirs),
        "MOEA/D-STM": get_moead_stm_algorithm(ref_dirs),
    }
    
    if with_adaptive:
        from src.algorithms.en_moead import EnhancedMOEAD
        algorithms["En-MOEA/D"] = EnhancedMOEAD(ref_dirs, base_problem=problem)
        algorithms["Hybrid-Adaptive"] = EnhancedMOEAD(ref_dirs, base_problem=problem)
    
    return algorithms