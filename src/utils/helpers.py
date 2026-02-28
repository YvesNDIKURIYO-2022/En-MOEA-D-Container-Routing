# src/utils/helpers.py
"""
Helper utility functions for data handling, validation, and I/O operations.
"""

import pandas as pd
import numpy as np
import os
import json
from src.utils.config import ALPHA, BETA


# =============================================
# Data Validation and Optimization
# =============================================

def validate_dataset(df):
    """Validate the generated dataset for consistency."""
    print(f"\n🔍 VALIDATING DATASET...")
    
    required_columns = [
        'Destination', 'Key Import/Export Companies', 'Mode', 
        'Cost (cijm) (USD/TEU)', 'Reliability (Relijm)',
        'Stochastic Time (tijm,k) (Hours)', 'TotalDeliveredCost_USD'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"❌ Missing required columns: {missing_columns}")
    
    assert df['Reliability (Relijm)'].between(0.1, 0.99).all(), "❌ Reliability out of bounds"
    assert (df['Cost (cijm) (USD/TEU)'] > 0).all(), "❌ Cost must be positive"
    assert (df['Stochastic Time (tijm,k) (Hours)'] > 0).all(), "❌ Time must be positive"
    assert (df['TotalDeliveredCost_USD'] > 0).all(), "❌ Total cost must be positive"
    
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"⚠️  Found {duplicate_count} duplicate records")
    
    print(f"✅ Dataset validation passed:")
    print(f"   - All required columns present")
    print(f"   - Value ranges within expected bounds")
    print(f"   - No critical data quality issues")
    
    return True


def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage."""
    print(f"\n💾 OPTIMIZING DATAFRAME MEMORY...")
    
    original_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
    
    # Downcast numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert object columns to category where appropriate
    for col in df.select_dtypes(include=['object']):
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    
    optimized_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
    memory_saved = original_memory - optimized_memory
    
    print(f"✅ Memory optimization completed:")
    print(f"   Original: {original_memory:.2f} MB")
    print(f"   Optimized: {optimized_memory:.2f} MB")
    print(f"   Saved: {memory_saved:.2f} MB ({memory_saved/original_memory*100:.1f}%)")
    
    return df


# =============================================
# File I/O Operations
# =============================================

def save_dataframe(df, filepath, openpyxl_available=True):
    """Save DataFrame with fallback to CSV if Excel is not available."""
    try:
        if openpyxl_available and filepath.endswith('.xlsx'):
            df.to_excel(filepath, index=False)
            print(f"✅ Data saved to Excel: {filepath}")
        else:
            csv_path = filepath.replace('.xlsx', '.csv') if filepath.endswith('.xlsx') else filepath
            df.to_csv(csv_path, index=False)
            print(f"✅ Data saved to CSV: {csv_path}")
    except Exception as e:
        print(f"❌ Error saving data: {e}")
        csv_path = filepath.replace('.xlsx', '.csv') if filepath.endswith('.xlsx') else filepath
        df.to_csv(csv_path, index=False)
        print(f"✅ Data saved to CSV (fallback): {csv_path}")


def save_combined_results(results, filename, base_path, openpyxl_available=True):
    """Save combined results from multiple algorithms."""
    final_df = pd.concat([pd.DataFrame(algo_results) for algo_results in results.values()], 
                        ignore_index=True)
    excel_path = os.path.join(base_path, f"{filename}.xlsx")
    save_dataframe(final_df, excel_path, openpyxl_available)
    return final_df


def save_manuscript_statistics(stats, filepath):
    """Save manuscript statistics as JSON with proper type conversion."""
    
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert_types(stats), f, indent=2)
    
    print(f"✅ Statistics saved to: {filepath}")


# =============================================
# Robustness Metrics
# =============================================

def mean_variance_robustness(objective_values):
    """Compute a robust objective as a weighted sum of mean and variance."""
    mean = np.mean(objective_values, axis=0)
    variance = np.var(objective_values, axis=0)
    return ALPHA * mean + BETA * variance


def evaluate_solution_robustness(x, problem, n_tests=8):
    """Evaluate how robust a solution is to perturbations."""
    if x is None or len(x.shape) < 2:
        return 0.0
    
    try:
        performance_changes = []
        
        for test in range(n_tests):
            # Create different types of perturbations
            if test % 3 == 0:
                perturbation = np.random.normal(0, 0.05, x.shape)
            elif test % 3 == 1:
                perturbation = np.random.normal(0, 0.1, x.shape)
            else:
                perturbation = np.random.uniform(-0.08, 0.08, x.shape)
            
            # Apply perturbation with probability
            mask = np.random.random(x.shape) < 0.7
            perturbation[~mask] = 0
            
            perturbed_x = np.clip(x + perturbation, 0, 1)
            
            # Evaluate perturbed performance
            temp_out = {}
            problem._evaluate(perturbed_x, temp_out)
            if 'F' in temp_out:
                perturbed_perf = np.linalg.norm(temp_out['F'])
                
                temp_out2 = {}
                problem._evaluate(x, temp_out2)
                if 'F' in temp_out2:
                    original_perf = np.linalg.norm(temp_out2['F'])
                    
                    if original_perf > 1e-10:
                        variation = abs(perturbed_perf - original_perf) / original_perf
                        performance_changes.append(variation)
        
        if performance_changes:
            avg_change = np.mean(performance_changes)
            robustness = 1.0 / (1.0 + avg_change * 10)
            return min(1.0, robustness)
            
    except Exception as e:
        pass
    
    return 0.0


def evaluate_population_robustness(X, problem, n_tests=8):
    """Evaluate robustness of an entire population."""
    if X is None or X.shape[0] == 0:
        return {
            'survival_rate': 0, 
            'performance_stability': 0, 
            'robustness_score': 0
        }
    
    try:
        F_orig = problem.evaluate(X, return_values_of=["F"])
        
        survival_count = 0
        performance_changes = []
        robustness_scores = []
        
        for test in range(n_tests):
            # Create perturbation
            if test % 2 == 0:
                perturbation = np.random.normal(0, 0.08, X.shape)
            else:
                perturbation = np.random.uniform(-0.1, 0.1, X.shape)
            
            perturbed_X = np.clip(X + perturbation, 0, 1)
            
            try:
                F_pert = problem.evaluate(perturbed_X, return_values_of=["F"])
                
                # Survival check (no extreme degradation)
                survival_threshold = 3.0
                survived = np.all(F_pert < F_orig * survival_threshold, axis=1)
                survival_count += np.sum(survived)
                
                # Performance stability for survived solutions
                for i in range(len(survived)):
                    if survived[i]:
                        orig_norm = np.linalg.norm(F_orig[i])
                        pert_norm = np.linalg.norm(F_pert[i])
                        if orig_norm > 1e-10:
                            change = abs(pert_norm - orig_norm) / orig_norm
                            performance_changes.append(change)
                            robustness = 1.0 / (1.0 + change * 8)
                            robustness_scores.append(robustness)
            
            except:
                continue
        
        total_tests = X.shape[0] * n_tests
        survival_rate = 100 * survival_count / total_tests if total_tests > 0 else 0
        avg_stability = (1.0 - np.mean(performance_changes)) * 100 if performance_changes else 0
        avg_robustness = np.mean(robustness_scores) * 100 if robustness_scores else 0
        
        return {
            'survival_rate': survival_rate,
            'performance_stability': avg_stability,
            'robustness_score': avg_robustness
        }
        
    except Exception as e:
        return {
            'survival_rate': 0, 
            'performance_stability': 0, 
            'robustness_score': 0
        }