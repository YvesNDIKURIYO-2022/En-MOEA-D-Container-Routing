# src/problems/benchmark.py
"""
Standard benchmark problems: ZDT, DTLZ, WFG suites.
"""

from pymoo.problems import get_problem
from src.utils.config import N_VAR_ZDT, N_VAR_DTLZ, N_VAR_WFG, N_OBJ_DTLZ, N_OBJ_WFG


def get_zdt_problems():
    """Get all ZDT benchmark problems."""
    return {
        "ZDT1": get_problem("zdt1"),
        "ZDT2": get_problem("zdt2"),
        "ZDT3": get_problem("zdt3"),
        "ZDT4": get_problem("zdt4"),
        "ZDT5": get_problem("zdt5"),
        "ZDT6": get_problem("zdt6"),
    }


def get_dtlz_problems(n_var=N_VAR_DTLZ, n_obj=N_OBJ_DTLZ):
    """Get all DTLZ benchmark problems."""
    return {
        "DTLZ1": get_problem("dtlz1", n_var=n_var, n_obj=n_obj),
        "DTLZ2": get_problem("dtlz2", n_var=n_var, n_obj=n_obj),
        "DTLZ3": get_problem("dtlz3", n_var=n_var, n_obj=n_obj),
        "DTLZ4": get_problem("dtlz4", n_var=n_var, n_obj=n_obj),
        "DTLZ5": get_problem("dtlz5", n_var=n_var, n_obj=n_obj),
        "DTLZ6": get_problem("dtlz6", n_var=n_var, n_obj=n_obj),
        "DTLZ7": get_problem("dtlz7", n_var=n_var, n_obj=n_obj),
    }


def get_wfg_problems(n_var=N_VAR_WFG, n_obj=N_OBJ_WFG):
    """Get selected WFG benchmark problems."""
    return {
        "WFG1": get_problem("wfg1", n_var=n_var, n_obj=n_obj),
        "WFG2": get_problem("wfg2", n_var=n_var, n_obj=n_obj),
        "WFG3": get_problem("wfg3", n_var=n_var, n_obj=n_obj),
        "WFG4": get_problem("wfg4", n_var=n_var, n_obj=n_obj),
        "WFG5": get_problem("wfg5", n_var=n_var, n_obj=n_obj),
        "WFG6": get_problem("wfg6", n_var=n_var, n_obj=n_obj),
        "WFG7": get_problem("wfg7", n_var=n_var, n_obj=n_obj),
        "WFG8": get_problem("wfg8", n_var=n_var, n_obj=n_obj),
        "WFG9": get_problem("wfg9", n_var=n_var, n_obj=n_obj),
    }


def get_all_benchmark_problems():
    """Get all benchmark problems."""
    problems = {}
    problems.update(get_zdt_problems())
    problems.update(get_dtlz_problems())
    problems.update(get_wfg_problems())
    return problems