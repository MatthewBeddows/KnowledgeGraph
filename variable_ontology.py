"""
Variable Ontology: Semantic Classification System
Defines hierarchical categories and classification logic for agricultural variables
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class VariableMetadata:
    """Metadata for understanding what a variable represents"""
    name: str
    unit: Optional[str]
    data_range: Tuple[float, float]
    measurement_type: str
    semantic_category: str
    aliases: List[str]
    physical_meaning: str
    
    # Pattern-based properties
    statistical_fingerprint: Optional[Dict] = None
    temporal_signature: Optional[Dict] = None
    predictive_power: float = 0.0
    behavior_cluster: int = -1


class VariableOntology:
    """Defines the semantic hierarchy with improved scoring"""
    
    def __init__(self):
        self.ontology = {
            'Temperature': {
                'unit_patterns': ['celsius', 'c', 'fahrenheit', 'f', 'kelvin', 'k', 'temp', 'degree'],
                'name_patterns': ['temp', 'temperature', 'thermal', 'heat', 'degrees', 'dewpoint'],
                'typical_range': [-20, 50],
                'subcategories': {
                    'Air_Temperature': ['air', 'ambient', 'atmospheric', 'era5'],
                    'Soil_Temperature': ['soil', 'ground'],
                    'Canopy_Temperature': ['canopy', 'leaf', 'plant']
                }
            },
            'Moisture': {
                'unit_patterns': ['%', 'percent', 'vwc', 'm3/m3', 'volumetric'],
                'name_patterns': ['moisture', 'water', 'humidity', 'wetness'],
                'typical_range': [0, 100],
                'subcategories': {
                    'Soil_Moisture': ['soil', 'ground'],
                    'Air_Humidity': ['air', 'relative', 'rh'],
                    'Leaf_Wetness': ['leaf', 'canopy']
                }
            },
            'Precipitation': {
                'unit_patterns': ['mm', 'inch', 'in', 'precipitation', 'rainfall'],
                'name_patterns': ['rain', 'rainfall', 'precipitation', 'precip', 'total_precipitation'],
                'typical_range': [0, 200],
                'subcategories': {
                    'Rainfall': ['rain'],
                    'Irrigation': ['irrigation', 'irrigated', 'watering']
                }
            },
            'Wind': {
                'unit_patterns': ['m/s', 'km/h', 'mph', 'component'],
                'name_patterns': ['wind', 'gust', 'breeze', 'wind_u', 'wind_v'],
                'typical_range': [-50, 50],
                'subcategories': {
                    'Wind_Speed': ['speed', 'velocity'],
                    'Wind_Component': ['component', 'u_component', 'v_component']
                }
            },
            'Pressure': {
                'unit_patterns': ['pa', 'hpa', 'mbar', 'pressure'],
                'name_patterns': ['pressure', 'atmospheric', 'surface_pressure'],
                'typical_range': [900, 1100],
                'subcategories': {
                    'Surface_Pressure': ['surface', 'atmospheric']
                }
            },
            'Nutrient': {
                'unit_patterns': ['ppm', 'mg/kg', 'mg/l', 'nutrient'],
                'name_patterns': ['nitrogen', 'phosphorus', 'potassium', 'nutrient', 'fertilizer', 
                                 'npk', 'nitrate', 'phosphate', 'potash'],
                'typical_range': [0, 500],
                'subcategories': {
                    'Nitrogen': ['nitrogen', 'nitrate', 'nh4', 'no3'],
                    'Phosphorus': ['phosphorus', 'phosphate'],
                    'Potassium': ['potassium', 'potash']
                }
            },
            'Vegetation_Index': {
                'unit_patterns': ['ndvi', 'evi', 'index', 'ratio'],
                'name_patterns': ['ndvi', 'evi', 'savi', 'vegetation', 'greenness', 'vigor'],
                'typical_range': [-1, 1],
                'subcategories': {
                    'NDVI': ['ndvi'],
                    'EVI': ['evi'],
                    'Other_Index': ['savi', 'gndvi']
                }
            },
            'Yield': {
                'unit_patterns': ['kg/ha', 'ton/ha', 'bu/acre', 'kg', 'ton', 'yield'],
                'name_patterns': ['yield', 'production', 'harvest', 'output', 'biomass', 'cumulativeyield'],
                'typical_range': [0, 20000],
                'subcategories': {
                    'Final_Yield': ['final', 'harvest', 'total'],
                    'Cumulative_Yield': ['cumulative', 'cum']
                }
            },
            'Growth_Metric': {
                'unit_patterns': ['cm', 'm', 'height', 'area', 'count', 'age'],
                'name_patterns': ['height', 'growth', 'biomass', 'lai', 'area', 'stand', 'age', 'plantage'],
                'typical_range': [0, 300],
                'subcategories': {
                    'Height': ['height', 'tall'],
                    'LAI': ['lai', 'leaf_area'],
                    'Age': ['age', 'plantage'],
                    'Biomass': ['biomass', 'weight', 'mass']
                }
            },
            'Location_Property': {
                'unit_patterns': ['acre', 'ha', 'hectare'],
                'name_patterns': ['acres', 'area', 'size'],
                'typical_range': [0, 1000],
                'subcategories': {
                    'Area': ['acres', 'area', 'hectare']
                }
            },
            'Encoded_Variable': {
                'unit_patterns': ['encoded', 'id'],
                'name_patterns': ['encoded', 'variety', 'tunnel', 'farm', 'lookup'],
                'typical_range': [0, 1000],
                'subcategories': {
                    'Categorical': ['variety', 'tunnel', 'type'],
                    'Identifier': ['farm', 'lookup', 'id']
                }
            },
            'Shifted_Variable': {
                'unit_patterns': [],
                'name_patterns': ['shifted'],
                'typical_range': None,
                'subcategories': {
                    'Lagged': ['shifted', 'lag']
                }
            }
        }
    
    def classify_variable(self, var_name: str, unit: Optional[str] = None, 
                         data_sample: Optional[np.ndarray] = None) -> Tuple[str, str, float]:
        """
        Classify a variable with improved scoring
        
        Args:
            var_name: Name of the variable
            unit: Unit of measurement (optional)
            data_sample: Sample of data values (optional)
            
        Returns:
            Tuple of (category, subcategory, confidence_score)
        """
        var_name_lower = var_name.lower()
        unit_lower = unit.lower() if unit else ""
        
        scores = {}
        
        for category, properties in self.ontology.items():
            score = 0.0
            matched_subcategory = None
            
            # Check name patterns with higher weight
            name_match_count = 0
            for pattern in properties['name_patterns']:
                if pattern in var_name_lower:
                    name_match_count += 1
            
            if name_match_count > 0:
                score += 3.0 * name_match_count
            
            # Check unit patterns
            if unit_lower:
                for pattern in properties['unit_patterns']:
                    if pattern in unit_lower:
                        score += 2.0
                        break
            
            # Check data range if available
            if data_sample is not None and len(data_sample) > 0 and properties['typical_range']:
                data_min, data_max = np.nanmin(data_sample), np.nanmax(data_sample)
                expected_min, expected_max = properties['typical_range']
                
                # More lenient range checking
                if expected_min * 0.5 <= data_min and data_max <= expected_max * 2:
                    score += 0.5
            
            # Check subcategories
            for subcat, subcat_patterns in properties['subcategories'].items():
                for pattern in subcat_patterns:
                    if pattern in var_name_lower:
                        score += 1.0
                        matched_subcategory = subcat
                        break
            
            if score > 0:
                scores[category] = (score, matched_subcategory or 'General')
        
        if not scores:
            return ('Unknown', 'Unknown', 0.0)
        
        best_category = max(scores.items(), key=lambda x: x[1][0])
        category_name = best_category[0]
        confidence = min(best_category[1][0] / 10.0, 1.0)  # Normalize to 0-1
        subcategory = best_category[1][1]
        
        return (category_name, subcategory, confidence)
    
    def find_similar_variables(self, target_var_metadata: VariableMetadata, 
                              known_variables: List[VariableMetadata]) -> List[Tuple[VariableMetadata, float]]:
        """
        Find variables similar to target based on semantic meaning
        
        Args:
            target_var_metadata: Metadata of the variable to match
            known_variables: List of known variable metadata
            
        Returns:
            List of (metadata, similarity_score) tuples, sorted by similarity
        """
        similarities = []
        
        for known_var in known_variables:
            similarity = 0.0
            
            # Semantic category match
            if known_var.semantic_category == target_var_metadata.semantic_category:
                similarity += 0.5
            
            # Measurement type match
            if known_var.measurement_type == target_var_metadata.measurement_type:
                similarity += 0.2
            
            # Unit compatibility
            if self._units_compatible(known_var.unit, target_var_metadata.unit):
                similarity += 0.2
            
            # Range overlap
            range_overlap = self._calculate_range_overlap(
                known_var.data_range, target_var_metadata.data_range
            )
            similarity += range_overlap * 0.1
            
            if similarity > 0.3:
                similarities.append((known_var, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)
    
    def _units_compatible(self, unit1: Optional[str], unit2: Optional[str]) -> bool:
        """Check if two units are compatible"""
        if not unit1 or not unit2:
            return False
        
        unit_families = [
            ['celsius', 'c', 'fahrenheit', 'f', 'kelvin', 'k'],
            ['mm', 'inch', 'in', 'cm'],
            ['%', 'percent', 'pct'],
            ['kg/ha', 'ton/ha', 'bu/acre'],
            ['ppm', 'mg/kg', 'mg/l']
        ]
        
        unit1_lower = unit1.lower()
        unit2_lower = unit2.lower()
        
        for family in unit_families:
            if any(u in unit1_lower for u in family) and any(u in unit2_lower for u in family):
                return True
        
        return unit1_lower == unit2_lower
    
    def _calculate_range_overlap(self, range1: Tuple[float, float], 
                                 range2: Tuple[float, float]) -> float:
        """
        Calculate overlap between two ranges (0 to 1)
        
        Args:
            range1: First range as (min, max)
            range2: Second range as (min, max)
            
        Returns:
            Overlap ratio from 0.0 to 1.0
        """
        min1, max1 = range1
        min2, max2 = range2
        
        overlap_start = max(min1, min2)
        overlap_end = min(max1, max2)
        
        if overlap_start >= overlap_end:
            return 0.0
        
        overlap = overlap_end - overlap_start
        total_span = max(max1, max2) - min(min1, min2)
        
        return overlap / total_span if total_span > 0 else 0.0
    
    def get_category_info(self, category: str) -> Optional[Dict]:
        """
        Get information about a category
        
        Args:
            category: Name of the category
            
        Returns:
            Dictionary with category information or None if not found
        """
        return self.ontology.get(category)
    
    def get_all_categories(self) -> List[str]:
        """Get list of all category names"""
        return list(self.ontology.keys())
    
    def get_all_subcategories(self, category: str) -> List[str]:
        """Get all subcategories for a given category"""
        if category in self.ontology:
            return list(self.ontology[category]['subcategories'].keys())
        return []