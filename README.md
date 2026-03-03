# En-MOEA/D: Enhanced Multi-Objective Evolutionary Algorithm based on Decomposition for Container Routing under Uncertainty

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
**Preprint:** [DOI: 10.2139/ssrn.5399005](https://dx.doi.org/10.2139/ssrn.5399005) 

**Official implementation of the Enhanced MOEA/D (En-MOEA/D) algorithm** for multi-objective robust optimization of container routing problems under uncertainty. This repository contains all the code necessary to reproduce the experiments and results presented in the paper:

> *"A Multi-objective Robust Optimization based on Evolutionary Algorithm for container routing problems under risks and uncertainties"* (Applied Soft Computing, 2026)

**Authors:** Yves Ndikuriyo, Yinggui Zhang, Dung Davou Fom  
**Corresponding author:** Yves Ndikuriyo ([yvesndikuriyo@csu.edu.cn](mailto:yvesndikuriyo@csu.edu.cn))  
**Institution:** School of Traffic and Transportation Engineering, Central South University, Changsha, China

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Innovations](#-key-innovations)
- [Performance Highlights](#-performance-highlights)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Running Experiments](#-running-experiments)
- [Configuration](#-configuration)
- [Output and Visualization](#-output-and-visualization)
- [Authors](#-authors)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 📋 Overview

En-MOEA/D is a novel framework that bridges the critical gap between multi-objective optimization and robust decision-making for container routing under uncertainty. The algorithm integrates three key innovations to explicitly balance expected performance, worst-case resilience, and operational stability.

The framework has been extensively validated on:
- **ZDT, DTLZ, and WFG benchmark suites** - Demonstrating superior convergence and diversity
- **Robust test problems** - GFunction, Ishigami, OakleyOHagan, and more
- **Real-world East African logistics network** - Northern and Central Corridors connecting landlocked countries to maritime ports

---

## 🔬 Key Innovations

### 1. Adaptive Decomposition Mechanism
Dynamically transitions between PBI (Penalty Boundary Intersection) and Chebyshev scalarization based on search progress:
- **Early stages**: PBI maximizes solution diversity by penalizing deviations from ideal weights
- **Later stages**: Chebyshev prioritizes convergence by minimizing the largest weighted deviation
- **Adaptive transition**: Monitors Hypervolume improvement to dynamically switch strategies at the optimal point (50% of generations)

### 2. High-Fidelity Uncertainty Quantification
Combines Monte Carlo simulation for probabilistic scenario analysis with worst-case robust optimization:
- **100 Monte Carlo scenarios** capturing both frequent operational fluctuations and rare severe disruptions
- **Truncated normal distributions** for realistic parameter variations within bounded intervals
- **Fixed random seed** (42) ensuring complete experimental reproducibility

### 3. Mean-Variance Risk Control
Embeds a financially-inspired, tunable risk-preference model directly within objective functions:

**Robust Cost** ($\widetilde{Z}_1$):
$$\widetilde{Z}_1(x) = \omega \cdot \mathbb{E}[Z_1^{stoch}(x)] + (1 - \omega) \cdot Z_1^{worst}(x) + \gamma \cdot Var[Z_1^{stoch}(x)]$$

**Robust Time** ($\widetilde{Z}_2$):
$$\widetilde{Z}_2(x) = \mathbb{E}[Z_2(x)] + \gamma_2 \cdot Var[Z_2(x)]$$

**Robust Reliability** ($\widetilde{Z}_3$):
$$\widetilde{Z}_3(x) = \mathbb{E}[Z_3(x)] - \gamma_3 \cdot Var[Z_3(x)]$$

---

## 📊 Performance Highlights

### Benchmark Results
| Metric | Improvement |
|--------|-------------|
| **Hypervolume** | **+45.2%** (β: 0.3 → 0.9) |
| **Total Performance** | **~54%** (optimized parameter set) |
| **ω Extreme Degradation** | **20.2%** (confirms balanced weighting essential) |
| **τ Sensitivity** | **1.48%** (robust to convergence window variations) |

### Real-World Case Study (East African Community)
| Metric | Northern Corridor | Central Corridor | Advantage |
|--------|-------------------|------------------|-----------|
| Average Cost/TEU | $5,214.61 | $5,078.98 | **+2.6% Central** |
| Average Reliability | 0.65 | 0.68 | **+4.6% Central** |
| Under 20% Perturbation | +13.0% cost increase | +13.7% cost increase | Comparable resilience |
| Operational Reliability | 76% maintained under 15-25% disruptions | | |

### Algorithm Comparison (Burundi Routes)
| Algorithm | Northern Corridor | Central Corridor |
|-----------|-------------------|------------------|
| **En-MOEA/D** | **$6,116** \| **502.8h** \| 0.40 | **$5,635** \| **478.1h** \| 0.40 |
| MOEA/D | $6,230 \| 524.4h \| 0.38 | $5,738 \| 500.8h \| 0.38 |
| MOEA/D-DE | $6,119 \| 509.1h \| 0.39 | $5,667 \| 486.5h \| 0.39 |
| MOEA/D-AWA | $6,271 \| 538.9h \| 0.37 | $5,813 \| 511.7h \| 0.37 |
| MOEA/D-STM | $6,309 \| 551.2h \| 0.36 | $5,866 \| 524.3h \| 0.36 |

---

## 📁 Repository Structure

```
En-MOEA-D-Container-Routing/
├── src/
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── en_moead.py                # Main EnhancedMOEAD algorithm
│   │   ├── decomposition.py           # Adaptive PBI/Chebyshev switching
│   │   ├── robustness.py              # Monte Carlo & mean-variance components
│   │   └── variants.py                # MOEA/D variants for comparison
│   ├── problems/
│   │   ├── __init__.py
│   │   ├── benchmark.py               # ZDT, DTLZ, WFG problem definitions
│   │   ├── robust_problems.py         # GFunction, Ishigami, OakleyOHagan
│   │   └── east_africa.py             # East African case study data generation
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── performance.py             # HV, IGD, Spread calculations
│   │   └── statistics.py              # t-tests, ANOVA, Cohen's d
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plotting.py                # Pareto fronts, sensitivity plots, comparisons
│   └── utils/
│       ├── __init__.py
│       ├── config.py                  # Centralized configuration with optimal parameters
│       └── helpers.py                 # Data validation, I/O, memory optimization
├── experiments/
│   ├── run_adaptive_decomposition.py  # Compares PBI-only, Tchebicheff-only, Hybrid-Adaptive
│   ├── run_sensitivity_analysis.py    # Validates optimal parameters α, β, ω, γ, τ
│   ├── run_mean_variance.py           # RobustMOEA/D vs NSGA-II, SPEA2, MOEA/D
│   ├── run_monte_carlo.py             # MOEA/D vs MOEA/D-MCSS under uncertainty
│   ├── run_robust_problems.py         # GFunction, Ishigami, OakleyOHagan benchmarks
│   ├── run_wfg_benchmarks.py          # WFG1-WFG9 test problems
│   ├── run_zdt_dtlz_benchmarks.py     # ZDT1-ZDT6 and DTLZ1-DTLZ7
│   └── run_case_study.py              # East African logistics case study
├── data/
│   └── east_africa/                   # Generated case study data
├── results/                           # Output directory (created automatically)
├── notebooks/                         # Jupyter notebooks for interactive analysis
├── tests/                             # Unit tests
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── LICENSE                            # MIT License
└── README.md                          # This file
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.13 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/YvesNDIKURIYO-2022/En-MOEA-D-Container-Routing.git
cd En-MOEA-D-Container-Routing

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pymoo>=0.6.0
deap>=1.4.0
openpyxl>=3.1.0
jupyter>=1.0.0
statsmodels>=0.14.0
tqdm>=4.65.0
```

---

## 🧪 Running Experiments

Each experiment corresponds to a validation section in the paper. Run them in any order:

### 1. Adaptive Decomposition Validation
```bash
python experiments/run_adaptive_decomposition.py
```
Tests dynamic switching between PBI and Chebyshev on ZDT, DTLZ, and WFG benchmarks. Outputs Pareto front comparisons and performance metrics.

### 2. Parameter Sensitivity Analysis
```bash
python experiments/run_sensitivity_analysis.py
```
Validates optimal parameters α, β, ω, γ, τ, and MC scenarios. Includes ablation study for ω extremes (0.0 and 1.0).

### 3. Mean-Variance Robustness Validation
```bash
python experiments/run_mean_variance.py
```
Compares RobustMOEA/D against NSGA-II, SPEA2, and standard MOEA/D on ZDT and WFG problems.

### 4. Monte Carlo Simulation Validation
```bash
python experiments/run_monte_carlo.py
```
Compares MOEA/D vs MOEA/D-MCSS under uncertainty on ZDT and DTLZ problems with perturbation.

### 5. Robust Test Problems
```bash
python experiments/run_robust_problems.py
```
Evaluates all algorithms on GFunction, Ishigami, OakleyOHagan, Borehole, and other robust test problems.

### 6. WFG Benchmark Suites
```bash
python experiments/run_wfg_benchmarks.py
```
Runs WFG1-WFG9 test problems with all algorithm variants.

### 7. ZDT and DTLZ Benchmark Suites
```bash
python experiments/run_zdt_dtlz_benchmarks.py
```
Runs ZDT1-ZDT6 and DTLZ1-DTLZ7 with comprehensive algorithm comparison.

### 8. East African Case Study
```bash
python experiments/run_case_study.py
```
Real-world validation on Northern and Central Corridors. Generates:
- Full East African logistics dataset (760 scenarios)
- Robustness analysis under 5-25% perturbations
- Algorithm performance comparison
- Manuscript-ready statistics

---

## ⚙️ Configuration

All parameters are centralized in `src/utils/config.py`. The optimal values derived from sensitivity analysis are:

### Optimal Parameter Set
| Parameter | Description | Optimal Value |
|-----------|-------------|---------------|
| `ALPHA` (α) | Mean weight | **0.7** |
| `BETA` (β) | Variance weight | **0.9** |
| `OMEGA` (ω) | Balance parameter | **0.9** |
| `GAMMA` (γ) | Variance weight in cost | **0.2** |
| `GAMMA2` (γ₂) | Variance weight in time | **0.2** |
| `GAMMA3` (γ₃) | Variance weight in reliability | **0.2** |
| `TAU` (τ) | Convergence window | **0.5** (50%) |
| `NUM_SCENARIOS` | Monte Carlo samples | **100** |

### Computational Parameters
| Parameter | Value |
|-----------|-------|
| `MAX_EVALUATIONS` | 25,000 |
| `POP_SIZE` | 100 |
| `N_GEN` | 250 |
| `SEED` | 42 |
| `CROSSOVER_PROB` | 0.9 |
| `DISTRIBUTION_INDEX` | 20 |
| `PERTURBATIONS` | [5, 10, 15, 20, 25] |

---

## 📈 Output and Visualization

Each experiment generates:

### Console Output
- Real-time progress with generation-by-generation metrics
- Final performance metrics (HV, IGD, Spread)
- Statistical significance tests (t-tests, ANOVA)
- Algorithm rankings

### Saved Files in `results/` Directory
| Experiment | Output Files |
|------------|--------------|
| Adaptive Decomposition | `adaptive_decomposition_summary.csv`, `pareto_*.png` |
| Sensitivity Analysis | `sensitivity_results.csv`, `sensitivity_analysis.png` |
| Mean-Variance | `mean_variance_summary.csv`, `algorithm_ranking.csv` |
| Monte Carlo | `monte_carlo_comparison_*.png`, `results_*.json` |
| Robust Problems | `robust_problems_summary.csv`, `pareto_*.png` |
| WFG Benchmarks | `wfg_benchmarks_summary.csv`, `pareto_*.png` |
| ZDT/DTLZ Benchmarks | `zdt_dtlz_benchmarks_summary.csv`, `pareto_*.png` |
| Case Study | `EAC_Dataset.xlsx`, `Robustness_*.xlsx`, `*.png`, `manuscript_statistics.json` |

### Visualization Types
- **Pareto fronts** (2D and 3D) for all problems
- **Sensitivity analysis plots** for each parameter
- **Algorithm comparison bar charts** (cost, time, reliability)
- **Perturbation trend analysis** showing degradation under uncertainty
- **Dataset analysis** (cost distribution, seasonal impacts, reliability by mode)

---

## 👥 Authors

- **Yves Ndikuriyo** - Lead Researcher & Algorithm Development
  - 📧 yvesndikuriyo@csu.edu.cn
  - 🔗 [ORCID: 0009-0006-9324-7265](https://orcid.org/0009-0006-9324-7265)
  - Ph.D. student, School of Traffic and Transportation Engineering, Central South University
  - Research interests: logistics optimization, vehicle routing, robust decision-making

- **Yinggui Zhang** - Research Supervision & Methodology
  - 📧 ygzhang@csu.edu.cn
  - 🔗 [ORCID: 0000-0002-5790-0638](https://orcid.org/0000-0002-5790-0638)
  - Professor, School of Traffic and Transportation Engineering, Central South University
  - Research interests: multi-objective optimization, computational intelligence, railway optimization

- **Dung Davou Fom** - Experimental Analysis & Validation
  - 📧 fompatrickfom@gmail.com
  - 🔗 [ORCID: 0009-0001-8688-813X](https://orcid.org/0009-0001-8688-813X)
  - Ph.D. student, School of Traffic and Transportation Engineering, Central South University
  - Research interests: transportation engineering, risk analysis, multi-objective optimization

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{ndikuriyo2025multi,
  title={A Multi-Objective Robust Optimization Based on Evolutionary Algorithm for Container Routing Problems Under Risks and Uncertainties},
  author={Ndikuriyo, Yves and Zhang, Yinggui and Fom, Dung Davou},
  journal={SSRN Electronic Journal},
  year={2025},
  doi={10.2139/ssrn.5399005},
  url={https://ssrn.com/abstract=5399005}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for suggestions and bug reports.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings for new functions and classes
- Include unit tests for new functionality
- Update README documentation as needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Ndikuriyo Yves, Yinggui Zhang, Dung Davou Fom

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including, without limitation, the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- The MOEA/D framework by Zhang and Li [4]
- Pymoo library for multi-objective optimization tools [30]
- DEAP evolutionary computation framework [31]
- School of Traffic and Transportation Engineering, Central South University
- National Natural Science Foundation of China (Grant No. 71971220)
- Natural Science Foundation of Hunan Province (Grant Nos. 2023JJ30710, 2022JJ31020)
- East African Community for case study data [42-46]
- Northern Corridor Transit and Transport Coordination Authority (NCTTCA)
- Central Corridor Transit Transport Facilitation Agency (CCTTFA)

---

## 📞 Contact

For questions, issues, or collaboration opportunities:

- **Lead Researcher:** Yves Ndikuriyo - [yvesndikuriyo@csu.edu.cn](mailto:yvesndikuriyo@csu.edu.cn)
- **GitHub Issues:** [https://github.com/YvesNDIKURIYO-2022/En-MOEA-D-Container-Routing/issues](https://github.com/YvesNDIKURIYO-2022/En-MOEA-D-Container-Routing/issues)
- **Project Repository:** [https://github.com/YvesNDIKURIYO-2022/En-MOEA-D-Container-Routing](https://github.com/YvesNDIKURIYO-2022/En-MOEA-D-Container-Routing)

---

## 🔍 References

[4] Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm based on decomposition. *IEEE Transactions on Evolutionary Computation*, 11(6), 712-731.

[30] Blank, J., & Deb, K. (2020). Pymoo: Multi-Objective Optimization in Python. *IEEE Access*, 8, 89497-89509.

[31] De Rainville, F. M., et al. (2012). DEAP: A Python framework for Evolutionary Algorithms. *GECCO'12 Companion*, 85-92.

[42] East African Community. (2023). Statistics Bulletin - EAC Data Portal.

[43] World Bank. (2023). Logistics Performance Index (LPI).

[44] Northern Corridor Transit and Transport Coordination Authority. (2024). Northern Corridor Green Freight Strategy 2030.

[45] Central Corridor Transit Transport Facilitation Agency. (2024). Performance Monitoring Reports.

[46] IGAD Climate Prediction and Applications Centre. (2024). Seasonal Forecasts.

---

**Repository:** [https://github.com/YvesNDIKURIYO-2022/En-MOEA-D-Container-Routing](https://github.com/YvesNDIKURIYO-2022/En-MOEA-D-Container-Routing)  
**Last updated:** March 2026  
**Version:** 1.0.0
