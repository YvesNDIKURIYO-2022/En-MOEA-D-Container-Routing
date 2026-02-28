# src/utils/config.py
"""
Centralized configuration for all experiments.
Contains optimal parameters derived from sensitivity analysis.
"""

import numpy as np

# =============================================
# COMPUTATIONAL PARAMETERS
# =============================================
MAX_EVALUATIONS = 25000
POP_SIZE = 100
N_GEN = MAX_EVALUATIONS // POP_SIZE
SEED = 42

# =============================================
# OPTIMAL ROBUSTNESS PARAMETERS (from sensitivity analysis)
# =============================================
NUM_SCENARIOS = 100
ALPHA = 0.7      # Mean weight (α)
BETA = 0.9       # Variance weight (β)
OMEGA = 0.9      # Balance parameter (ω)
GAMMA = 0.2      # Variance weight in cost objective (γ)
GAMMA2 = 0.2     # Variance weight in time objective (γ₂)
GAMMA3 = 0.2     # Variance weight in reliability objective (γ₃)
TAU = 0.5        # Convergence window (τ) - 50% of generations

# =============================================
# REFERENCE POINTS
# =============================================
REF_POINT_ZDT = np.array([2.0, 2.0])
REF_POINT_DTLZ = np.array([2.0, 2.0, 2.0])
REF_POINT_WFG = np.array([2.0, 2.0, 2.0])

# =============================================
# GENETIC OPERATORS
# =============================================
CROSSOVER_PROB = 0.9
DISTRIBUTION_INDEX = 20
MUTATION_ETA = 20

# =============================================
# ADAPTIVE DECOMPOSITION PARAMETERS
# =============================================
PBI_THETA = 5.0
TRANSITION_THRESHOLD = 0.001
WINDOW_SIZE = 5

# =============================================
# ROBUSTNESS ANALYSIS PARAMETERS
# =============================================
PERTURBATIONS = [5, 10, 15, 20, 25]  # Percentage perturbations
ROBUSTNESS_TESTS = 8  # Number of robustness tests per evaluation

# =============================================
# PROBLEM DIMENSIONS
# =============================================
N_VAR_ZDT = 10
N_VAR_DTLZ = 10
N_VAR_WFG = 12
N_OBJ_DTLZ = 3
N_OBJ_WFG = 3

# =============================================
# ALGORITHM NAMES
# =============================================
ALGORITHMS = {
    "MOEA/D": "Standard MOEA/D with Tchebicheff",
    "MOEA/D-DE": "MOEA/D with Differential Evolution",
    "MOEA/D-AWA": "MOEA/D with Adaptive Weight Adjustment",
    "MOEA/D-STM": "MOEA/D with Stable Matching",
    "En-MOEA/D": "Enhanced MOEA/D (proposed)",
    "PBI-only": "MOEA/D with PBI decomposition",
    "Tchebicheff-only": "MOEA/D with Tchebicheff decomposition",
    "Hybrid-Adaptive": "MOEA/D with adaptive decomposition"
}