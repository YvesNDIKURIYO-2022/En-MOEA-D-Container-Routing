# src/problems/east_africa.py
"""
East African case study problem definition and data generation.
"""

import pandas as pd
import numpy as np
import random
import os
from src.utils.config import SEED, PERTURBATIONS

# Set random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)

# Corridor data from the original script
corridor_data = {
    'Central Corridor': {
        'Origin_Port': 'Dar es Salaam (Tanzania)',
        'Destinations': {
            'Burundi (Bujumbura)': {
                'Road': {
                    'companies': ['Interfreight Burundi', 'Bujumbura Logistics', 'Bolloré Africa Logistics'],
                    'cost_range': (1850, 2250),
                    'worst_case_cost_range': (2400, 2900),
                    'time_range': (240, 288),
                    'worst_case_time_range': (360, 408),
                    'reliability_range': (0.65, 0.75),
                    'border_delay_range': (8, 12),
                    'clearance_time_range': (8, 15),
                    'monthly_teu_range': (250, 400)
                },
                'Multimodal': {
                    'companies': ['Interfreight Burundi', 'Tanzania Railways Ltd'],
                    'cost_range': (1600, 1950),
                    'worst_case_cost_range': (2100, 2500),
                    'time_range': (264, 312),
                    'worst_case_time_range': (384, 384),
                    'reliability_range': (0.70, 0.80),
                    'border_delay_range': (6, 10),
                    'clearance_time_range': (6, 12),
                    'monthly_teu_range': (150, 300)
                }
            },
            'Rwanda (Kigali)': {
                'Road': {
                    'companies': ['Sheer Logistics', 'Rwanda Freight Forwarders', 'Mitsui O.S.K. Lines'],
                    'cost_range': (1750, 2150),
                    'worst_case_cost_range': (2300, 2800),
                    'time_range': (216, 264),
                    'worst_case_time_range': (336, 384),
                    'reliability_range': (0.70, 0.80),
                    'border_delay_range': (7, 11),
                    'clearance_time_range': (7, 14),
                    'monthly_teu_range': (400, 650)
                },
                'Multimodal': {
                    'companies': ['Sheer Logistics', 'TRL Logistics'],
                    'cost_range': (1500, 1850),
                    'worst_case_cost_range': (2000, 2400),
                    'time_range': (280, 330),
                    'worst_case_time_range': (400, 450),
                    'reliability_range': (0.72, 0.82),
                    'border_delay_range': (6, 9),
                    'clearance_time_range': (6, 11),
                    'monthly_teu_range': (200, 350)
                }
            },
            'Uganda (Kampala)': {
                'Road': {
                    'companies': ['One4All Logistics', 'Uganda Freight Services', 'Maersk East Africa'],
                    'cost_range': (1650, 2050),
                    'worst_case_cost_range': (2200, 2700),
                    'time_range': (192, 240),
                    'worst_case_time_range': (312, 360),
                    'reliability_range': (0.75, 0.85),
                    'border_delay_range': (6, 10),
                    'clearance_time_range': (6, 12),
                    'monthly_teu_range': (550, 800)
                }
            }
        }
    },
    'Northern Corridor': {
        'Origin_Port': 'Mombasa (Kenya)',
        'Destinations': {
            'Burundi (Bujumbura)': {
                'Road': {
                    'companies': ['Bujumbura Logistics', 'Bolloré Logistics', 'Kenya Freight Forwarders'],
                    'cost_range': (2050, 2450),
                    'worst_case_cost_range': (2700, 3200),
                    'time_range': (264, 312),
                    'worst_case_time_range': (384, 432),
                    'reliability_range': (0.68, 0.78),
                    'border_delay_range': (9, 13),
                    'clearance_time_range': (8, 15),
                    'monthly_teu_range': (180, 320)
                }
            },
            'Rwanda (Kigali)': {
                'Road': {
                    'companies': ['Sheer Logistics', 'Kenya Freight Forwarders', 'Mediterranean Shipping Co.'],
                    'cost_range': (1850, 2250),
                    'worst_case_cost_range': (2400, 2900),
                    'time_range': (192, 240),
                    'worst_case_time_range': (312, 360),
                    'reliability_range': (0.78, 0.88),
                    'border_delay_range': (5, 9),
                    'clearance_time_range': (5, 11),
                    'monthly_teu_range': (350, 600)
                },
                'Multimodal': {
                    'companies': ['Sheer Logistics', 'Rift Valley Railways'],
                    'cost_range': (1600, 1950),
                    'worst_case_cost_range': (2100, 2500),
                    'time_range': (220, 270),
                    'worst_case_time_range': (330, 380),
                    'reliability_range': (0.80, 0.90),
                    'border_delay_range': (4, 7),
                    'clearance_time_range': (4, 9),
                    'monthly_teu_range': (180, 320)
                }
            },
            'Uganda (Kampala)': {
                'Road': {
                    'companies': ['One4All Logistics', 'Spedag Interfreight', 'Hapag-Lloyd East Africa'],
                    'cost_range': (1250, 1650),
                    'worst_case_cost_range': (1650, 2150),
                    'time_range': (144, 192),
                    'worst_case_time_range': (264, 312),
                    'reliability_range': (0.82, 0.92),
                    'border_delay_range': (4, 8),
                    'clearance_time_range': (4, 10),
                    'monthly_teu_range': (600, 900)
                },
                'Multimodal': {
                    'companies': ['One4All Logistics', 'Kenya Railways'],
                    'cost_range': (1100, 1450),
                    'worst_case_cost_range': (1450, 1900),
                    'time_range': (168, 216),
                    'worst_case_time_range': (288, 336),
                    'reliability_range': (0.80, 0.90),
                    'border_delay_range': (4, 7),
                    'clearance_time_range': (4, 9),
                    'monthly_teu_range': (300, 500)
                }
            }
        }
    }
}

# Additional companies
additional_companies = [
    'Mitsui O.S.K. Lines', 'Maersk East Africa', 'CMA CGM East Africa',
    'Mediterranean Shipping Co.', 'Dubai Ports World', 'China Ocean Shipping Co.',
    'Hapag-Lloyd East Africa', 'Evergreen Marine Corp.', 'P&O Nedlloyd East Africa',
    'Safmarine', 'Zim Integrated Shipping', 'Pacific International Lines',
    'Wan Hai Lines', 'Yang Ming Marine Transport', 'Hyundai Merchant Marine'
]

# Seasonal variations
seasons = {
    'Dry Season': {'time_multiplier': 0.9, 'reliability_boost': 0.05, 'border_delay_multiplier': 0.8},
    'Rainy Season': {'time_multiplier': 1.2, 'reliability_boost': -0.08, 'border_delay_multiplier': 1.3}
}

# Uncertainty scenarios
uncertainty_scenarios = {
    'Normal Conditions': {'cost_multiplier': 1.0, 'time_multiplier': 1.0, 'reliability_multiplier': 1.0},
    'Mild Disruption': {'cost_multiplier': 1.15, 'time_multiplier': 1.2, 'reliability_multiplier': 0.9},
    'Severe Disruption': {'cost_multiplier': 1.3, 'time_multiplier': 1.5, 'reliability_multiplier': 0.7},
    'Border Crisis': {'cost_multiplier': 1.4, 'time_multiplier': 2.0, 'reliability_multiplier': 0.6},
    'Infrastructure Failure': {'cost_multiplier': 1.25, 'time_multiplier': 1.8, 'reliability_multiplier': 0.5}
}


class EastAfricaDataGenerator:
    """Generate East African case study data."""
    
    def __init__(self, max_international_companies=5):
        self.max_international_companies = max_international_companies
    
    def calculate_maximum_coverage(self):
        """Calculate the maximum possible scenarios."""
        total_scenarios = 0
        route_breakdown = {}
        
        print("🎯 CALCULATING ACTUAL MAXIMUM COVERAGE")
        print("=" * 60)
        
        for corridor, corridor_info in corridor_data.items():
            route_breakdown[corridor] = {}
            corridor_total = 0
            
            print(f"\n{corridor}:")
            print("-" * 40)
            
            for destination, destination_info in corridor_info['Destinations'].items():
                route_breakdown[corridor][destination] = {}
                destination_total = 0
                
                for mode, mode_info in destination_info.items():
                    local_companies = len(mode_info['companies'])
                    intl_companies = self.max_international_companies
                    total_companies = local_companies + intl_companies
                    
                    scenarios_per_route = total_companies * len(seasons) * len(uncertainty_scenarios)
                    total_scenarios += scenarios_per_route
                    corridor_total += scenarios_per_route
                    destination_total += scenarios_per_route
                    
                    route_breakdown[corridor][destination][mode] = {
                        'local_companies': local_companies,
                        'international_companies': intl_companies,
                        'total_scenarios': scenarios_per_route
                    }
                    
                    print(f"  {destination} - {mode}: {local_companies} local + {intl_companies} international = {scenarios_per_route} scenarios")
                
                print(f"  {destination} TOTAL: {destination_total} scenarios")
            
            print(f"{corridor} TOTAL: {corridor_total} scenarios")
        
        print(f"\n" + "=" * 60)
        print(f"🎯 ACTUAL MAXIMUM COVERAGE: {total_scenarios} scenarios")
        
        return total_scenarios, route_breakdown
    
    def get_realistic_costs(self, destination, mode, corridor, scenario='Normal Conditions'):
        """Generate realistic cost components based on East African market data."""
        scenario_multiplier = uncertainty_scenarios[scenario]['cost_multiplier']
        
        # Base FOB costs (varies by trade lane)
        if 'China' in destination or 'Asia' in destination:
            fob_cost = random.uniform(1200, 1800) * scenario_multiplier
        elif 'Europe' in destination or 'Middle East' in destination:
            fob_cost = random.uniform(1000, 1500) * scenario_multiplier
        else:
            fob_cost = random.uniform(800, 1300) * scenario_multiplier
        
        # CIF cost (FOB + insurance + freight)
        insurance_freight = random.uniform(400, 700) * scenario_multiplier
        cif_cost = fob_cost + insurance_freight
        
        # Port handling charges
        if corridor == 'Northern Corridor':
            port_handling = random.uniform(180, 280) * scenario_multiplier
        else:
            port_handling = random.uniform(160, 250) * scenario_multiplier
        
        # Clearance costs
        if destination in ['South Sudan (Juba)', 'DRC (Uvira/Goma)', 'Eastern DRC (Goma)']:
            clearance_cost = random.uniform(350, 600) * scenario_multiplier
        else:
            clearance_cost = random.uniform(250, 450) * scenario_multiplier
        
        return (round(fob_cost, 2), round(cif_cost, 2), 
                round(port_handling, 2), round(clearance_cost, 2))
    
    def create_scenario_record(self, destination, company, mode, corridor, corridor_info,
                              season_name, season_params, scenario_name, scenario_params, mode_info):
        """Create a single scenario record."""
        # Apply seasonal and scenario adjustments
        cost_multiplier = scenario_params['cost_multiplier']
        time_multiplier = scenario_params['time_multiplier'] * season_params['time_multiplier']
        reliability_multiplier = max(0.1, scenario_params['reliability_multiplier'] + 
                                    season_params['reliability_boost'])
        
        # Base costs and times
        inland_cost = round(random.uniform(mode_info['cost_range'][0], mode_info['cost_range'][1]) * 
                           cost_multiplier, 2)
        worst_case_cost = round(random.uniform(mode_info['worst_case_cost_range'][0], 
                                              mode_info['worst_case_cost_range'][1]) * cost_multiplier, 2)
        
        base_time = random.uniform(mode_info['time_range'][0], mode_info['time_range'][1]) * time_multiplier
        time_variation = random.normalvariate(0, base_time * 0.08)
        stochastic_time = round(max(base_time + time_variation, base_time * 0.9), 1)
        
        worst_case_time = round(random.uniform(mode_info['worst_case_time_range'][0], 
                                              mode_info['worst_case_time_range'][1]) * time_multiplier, 1)
        
        base_reliability = random.uniform(mode_info['reliability_range'][0], mode_info['reliability_range'][1])
        reliability = round(max(0.1, min(0.99, base_reliability * reliability_multiplier)), 2)
        
        border_delay = round(random.uniform(mode_info['border_delay_range'][0], 
                                           mode_info['border_delay_range'][1]) * 
                            season_params['border_delay_multiplier'], 1)
        clearance_time = round(random.uniform(mode_info['clearance_time_range'][0], 
                                             mode_info['clearance_time_range'][1]), 1)
        
        # Additional cost components
        fob_cost, cif_cost, port_handling, clearance_cost = self.get_realistic_costs(
            destination, mode, corridor, scenario_name
        )
        
        total_cost = round(cif_cost + inland_cost + port_handling + clearance_cost, 2)
        
        teu_min, teu_max = mode_info['monthly_teu_range']
        monthly_teus = random.randint(teu_min, teu_max)
        
        return {
            'Destination': destination,
            'Key Import/Export Companies': company,
            'Mode': mode,
            'Cost (cijm) (USD/TEU)': inland_cost,
            'Worst-Case Cost (c~ijm) (USD/TEU)': worst_case_cost,
            'Stochastic Time (tijm,k) (Hours)': stochastic_time,
            'Worst-Case Time (t~ijm) (Hours)': worst_case_time,
            'Reliability (Relijm)': reliability,
            'Corridor': corridor,
            'Origin_Port': corridor_info['Origin_Port'],
            'Season': season_name,
            'Uncertainty_Scenario': scenario_name,
            'FOB_Cost_USD': fob_cost,
            'CIF_Cost_USD': cif_cost,
            'PortHandling_USD': port_handling,
            'ClearanceTime_hr': clearance_time,
            'ClearanceCost_USD': clearance_cost,
            'InlandCost_USD': inland_cost,
            'BorderDelay_hr': border_delay,
            'TotalDeliveredCost_USD': total_cost,
            'MonthlyTEUs': monthly_teus
        }
    
    def generate_dataset(self):
        """Generate the complete East African dataset."""
        records = []
        actual_max_coverage, route_breakdown = self.calculate_maximum_coverage()
        scenario_count = 0
        
        print(f"\n🚀 GENERATING TRUE MAXIMUM COVERAGE DATASET...")
        print("=" * 60)
        
        for corridor, corridor_info in corridor_data.items():
            corridor_generated = 0
            print(f"\nGenerating {corridor} scenarios...")
            
            for destination, destination_info in corridor_info['Destinations'].items():
                destination_generated = 0
                
                for mode, mode_info in destination_info.items():
                    mode_generated = 0
                    
                    # Local companies
                    for company in mode_info['companies']:
                        for season_name, season_params in seasons.items():
                            for scenario_name, scenario_params in uncertainty_scenarios.items():
                                scenario_count += 1
                                mode_generated += 1
                                destination_generated += 1
                                corridor_generated += 1
                                
                                record = self.create_scenario_record(
                                    destination, company, mode, corridor, corridor_info,
                                    season_name, season_params, scenario_name, scenario_params, mode_info
                                )
                                records.append(record)
                    
                    # International companies
                    for company in additional_companies[:self.max_international_companies]:
                        for season_name, season_params in seasons.items():
                            for scenario_name, scenario_params in uncertainty_scenarios.items():
                                scenario_count += 1
                                mode_generated += 1
                                destination_generated += 1
                                corridor_generated += 1
                                
                                record = self.create_scenario_record(
                                    destination, company, mode, corridor, corridor_info,
                                    season_name, season_params, scenario_name, scenario_params, mode_info
                                )
                                records.append(record)
                    
                    print(f"  {destination} - {mode}: {mode_generated} scenarios generated")
                
                print(f"  {destination} TOTAL: {destination_generated} scenarios generated")
            
            print(f"{corridor} TOTAL: {corridor_generated} scenarios generated")
        
        print(f"\n" + "=" * 60)
        print(f"✅ DATASET GENERATION COMPLETED")
        print(f"   Generated: {scenario_count} scenarios")
        
        return pd.DataFrame(records)


def filter_route_data(dataset, origin, destination):
    """Filter dataset for specific origin-destination pair."""
    return dataset[(dataset["Origin_Port"] == origin) & 
                   (dataset["Destination"] == destination)].copy()