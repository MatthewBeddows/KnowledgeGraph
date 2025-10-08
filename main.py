"""
COMPLETE FIXED PATTERN-BASED SEMANTIC KNOWLEDGE GRAPH BUILDER
All original features preserved + fixes for classification and correlation issues
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import re
import warnings
warnings.filterwarnings('ignore')

# Statistical imports
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr


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
    """FIXED: Defines the semantic hierarchy with better scoring"""
    
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
        """FIXED: Classify a variable with improved scoring"""
        var_name_lower = var_name.lower()
        unit_lower = unit.lower() if unit else ""
        
        scores = {}
        
        for category, properties in self.ontology.items():
            score = 0.0
            matched_subcategory = None
            
            # FIXED: Check name patterns with higher weight
            name_match_count = 0
            for pattern in properties['name_patterns']:
                if pattern in var_name_lower:
                    name_match_count += 1
            
            if name_match_count > 0:
                score += 3.0 * name_match_count  # FIXED: Increased from 0.5
            
            # Check unit patterns
            if unit_lower:
                for pattern in properties['unit_patterns']:
                    if pattern in unit_lower:
                        score += 2.0  # FIXED: Increased from 0.3
                        break
            
            # Check data range if available
            if data_sample is not None and len(data_sample) > 0 and properties['typical_range']:
                data_min, data_max = np.nanmin(data_sample), np.nanmax(data_sample)
                expected_min, expected_max = properties['typical_range']
                
                # More lenient range checking
                if expected_min * 0.5 <= data_min and data_max <= expected_max * 2:
                    score += 0.5  # FIXED: Increased from 0.2
            
            # Check subcategories
            for subcat, subcat_patterns in properties['subcategories'].items():
                for pattern in subcat_patterns:
                    if pattern in var_name_lower:
                        score += 1.0  # FIXED: Increased from 0.3
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
        """Find variables similar to target based on semantic meaning"""
        similarities = []
        
        for known_var in known_variables:
            similarity = 0.0
            
            if known_var.semantic_category == target_var_metadata.semantic_category:
                similarity += 0.5
            
            if known_var.measurement_type == target_var_metadata.measurement_type:
                similarity += 0.2
            
            if self._units_compatible(known_var.unit, target_var_metadata.unit):
                similarity += 0.2
            
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
        """Calculate overlap between two ranges (0 to 1)"""
        min1, max1 = range1
        min2, max2 = range2
        
        overlap_start = max(min1, min2)
        overlap_end = min(max1, max2)
        
        if overlap_start >= overlap_end:
            return 0.0
        
        overlap = overlap_end - overlap_start
        total_span = max(max1, max2) - min(min1, min2)
        
        return overlap / total_span if total_span > 0 else 0.0


class PatternBasedAnalyzer:
    """FIXED: Analyze variables with better handling of skewed targets"""
    
    def __init__(self):
        self.fingerprints = {}
        
    def create_variable_fingerprint(self, data: np.ndarray, name: str = "") -> Dict:
        """Create comprehensive statistical signature of a variable"""
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) < 3:
            return {}
        
        fingerprint = {
            'name': name,
            # Basic statistics
            'mean': float(np.mean(clean_data)),
            'std': float(np.std(clean_data)),
            'min': float(np.min(clean_data)),
            'max': float(np.max(clean_data)),
            'range': float(np.max(clean_data) - np.min(clean_data)),
            'median': float(np.median(clean_data)),
            
            # Distribution shape
            'skewness': float(stats.skew(clean_data)),
            'kurtosis': float(stats.kurtosis(clean_data)),
            
            # Data properties
            'num_zeros': int(np.sum(clean_data == 0)),
            'num_negatives': int(np.sum(clean_data < 0)),
            'coefficient_of_variation': float(np.std(clean_data) / (np.mean(clean_data) + 1e-10)),
            
            # Percentiles
            'p25': float(np.percentile(clean_data, 25)),
            'p75': float(np.percentile(clean_data, 75)),
            'iqr': float(np.percentile(clean_data, 75) - np.percentile(clean_data, 25)),
            
            # Temporal properties
            'autocorrelation_lag1': self._safe_autocorr(clean_data, 1),
            'autocorrelation_lag7': self._safe_autocorr(clean_data, 7),
            'trend_strength': self._calculate_trend(clean_data),
        }
        
        return fingerprint
    
    def _safe_autocorr(self, data: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation safely"""
        if len(data) <= lag:
            return 0.0
        try:
            return float(np.corrcoef(data[:-lag], data[lag:])[0, 1])
        except:
            return 0.0
    
    def _calculate_trend(self, data: np.ndarray) -> float:
        """Calculate trend strength"""
        if len(data) < 3:
            return 0.0
        x = np.arange(len(data))
        try:
            slope, _, r_value, _, _ = stats.linregress(x, data)
            return float(r_value ** 2)  # R-squared as trend strength
        except:
            return 0.0
    
    def compare_fingerprints(self, fp1: Dict, fp2: Dict) -> float:
        """Compare two statistical fingerprints (0 to 1 similarity)"""
        if not fp1 or not fp2:
            return 0.0
        
        similarity = 0.0
        weights = {
            'range': 0.15,
            'skewness': 0.10,
            'coefficient_of_variation': 0.15,
            'autocorrelation_lag1': 0.15,
            'trend_strength': 0.10,
            'sign_pattern': 0.10,
            'distribution_shape': 0.15,
            'scale': 0.10
        }
        
        # Range similarity
        range_diff = abs(fp1['range'] - fp2['range']) / max(fp1['range'], fp2['range'], 1)
        similarity += (1 - min(range_diff, 1)) * weights['range']
        
        # Skewness similarity
        skew_diff = abs(fp1['skewness'] - fp2['skewness'])
        similarity += (1 - min(skew_diff / 3, 1)) * weights['skewness']
        
        # CV similarity
        cv_diff = abs(fp1['coefficient_of_variation'] - fp2['coefficient_of_variation'])
        similarity += (1 - min(cv_diff, 1)) * weights['coefficient_of_variation']
        
        # Autocorrelation similarity
        autocorr_diff = abs(fp1['autocorrelation_lag1'] - fp2['autocorrelation_lag1'])
        similarity += (1 - autocorr_diff) * weights['autocorrelation_lag1']
        
        # Trend similarity
        trend_diff = abs(fp1['trend_strength'] - fp2['trend_strength'])
        similarity += (1 - trend_diff) * weights['trend_strength']
        
        # Sign pattern
        both_positive = (fp1['num_negatives'] == 0) and (fp2['num_negatives'] == 0)
        both_mixed = (fp1['num_negatives'] > 0) and (fp2['num_negatives'] > 0)
        if both_positive or both_mixed:
            similarity += weights['sign_pattern']
        
        # Distribution shape (kurtosis)
        kurt_diff = abs(fp1['kurtosis'] - fp2['kurtosis'])
        similarity += (1 - min(kurt_diff / 5, 1)) * weights['distribution_shape']
        
        # Scale similarity (order of magnitude)
        scale_ratio = max(fp1['mean'], fp2['mean']) / (min(fp1['mean'], fp2['mean']) + 1e-10)
        if scale_ratio < 10:  # Within one order of magnitude
            similarity += weights['scale'] * (1 - np.log10(scale_ratio) / 1)
        
        return similarity
    
    def discover_predictive_variables(self, df: pd.DataFrame, target: str, 
                                     min_importance: float = 0.05) -> List[Dict]:
        """FIXED: Find predictive variables using non-zero targets"""
        predictive_vars = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c != target]
        
        if len(feature_cols) == 0 or target not in df.columns:
            return []
        
        # FIXED: Filter to non-zero targets for skewed yield data
        target_nonzero = df[target] > 0
        n_nonzero = target_nonzero.sum()
        
        if n_nonzero < 10:
            print(f"  ⚠️  Warning: Only {n_nonzero} non-zero target values")
            # Fall back to all data if too few non-zero
            df_analysis = df
        else:
            df_analysis = df[target_nonzero]
        
        # Prepare data
        X = df_analysis[feature_cols].fillna(df_analysis[feature_cols].median())
        y = df_analysis[target].fillna(df_analysis[target].median())
        
        if len(X) < 10:
            return []
        
        # Calculate mutual information
        try:
            mi_scores = mutual_info_regression(X, y, random_state=42)
        except:
            mi_scores = np.zeros(len(feature_cols))
        
        for i, col in enumerate(feature_cols):
            try:
                col_data = df_analysis[col].fillna(df_analysis[col].median())
                target_data = df_analysis[target].fillna(df_analysis[target].median())
                
                # Skip if no variance
                if col_data.std() == 0:
                    continue
                
                # Multiple correlation measures
                mi = mi_scores[i]
                
                # FIXED: Use Spearman for skewed data
                spearman_corr, spearman_p = spearmanr(col_data, target_data)
                if np.isnan(spearman_corr):
                    spearman_corr = 0
                
                pearson_corr = np.corrcoef(col_data, target_data)[0, 1]
                if np.isnan(pearson_corr):
                    pearson_corr = 0
                
                # FIXED: Weight Spearman more for skewed data
                importance = mi * 0.4 + abs(spearman_corr) * 0.4 + abs(pearson_corr) * 0.2
                
                if importance > min_importance:
                    rel_type = self._classify_relationship_type(pearson_corr, spearman_corr, mi)
                    
                    predictive_vars.append({
                        'variable': col,
                        'mutual_info': float(mi),
                        'spearman': float(spearman_corr),
                        'spearman_p': float(spearman_p),
                        'pearson': float(pearson_corr),
                        'combined_importance': float(importance),
                        'relationship_type': rel_type,
                        'n_samples': len(col_data)
                    })
            except:
                continue
        
        return sorted(predictive_vars, key=lambda x: x['combined_importance'], reverse=True)
    
    def _classify_relationship_type(self, pearson: float, spearman: float, mi: float) -> str:
        """Determine type of relationship"""
        if abs(pearson) > 0.7 and abs(spearman) > 0.7:
            return "strong_linear"
        elif abs(spearman) > 0.7 and abs(pearson) < 0.5:
            return "strong_monotonic_nonlinear"
        elif mi > 0.3 and abs(pearson) < 0.3:
            return "complex_nonlinear"
        elif abs(spearman) > 0.3 or mi > 0.2:
            return "moderate_dependency"
        else:
            return "weak_relationship"
    
    def cluster_variables_by_behavior(self, df: pd.DataFrame, target: str, 
                                     eps: float = 0.5) -> Dict[int, List[str]]:
        """Group variables by how they behave together"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c != target]
        
        if len(feature_cols) < 2:
            return {0: list(feature_cols)}
        
        # Create correlation matrix
        corr_matrix = df[feature_cols].corr().fillna(0).values
        
        # Create feature vectors for clustering
        feature_vectors = []
        for i, col in enumerate(feature_cols):
            corr_profile = corr_matrix[i, :]
            
            # Add target correlation if available
            if target in df.columns:
                try:
                    target_corr = np.corrcoef(
                        df[col].fillna(0), 
                        df[target].fillna(0)
                    )[0, 1]
                    if np.isnan(target_corr):
                        target_corr = 0
                except:
                    target_corr = 0
            else:
                target_corr = 0
            
            # Statistical properties
            stats_vec = [
                df[col].mean() / (df[col].std() + 1e-10),
                df[col].skew() if not np.isnan(df[col].skew()) else 0,
                df[col].kurt() if not np.isnan(df[col].kurt()) else 0
            ]
            
            feature_vectors.append(np.concatenate([corr_profile, [target_corr], stats_vec]))
        
        # Cluster
        scaler = StandardScaler()
        scaled_vectors = scaler.fit_transform(feature_vectors)
        
        clustering = DBSCAN(eps=eps, min_samples=2)
        clusters = clustering.fit_predict(scaled_vectors)
        
        # Organize results
        variable_clusters = {}
        for i, col in enumerate(feature_cols):
            cluster_id = int(clusters[i])
            if cluster_id not in variable_clusters:
                variable_clusters[cluster_id] = []
            variable_clusters[cluster_id].append(col)
        
        return variable_clusters
    
    def discover_temporal_predictors(self, df: pd.DataFrame, target: str,
                                    max_lag: int = 8, min_corr: float = 0.15) -> List[Dict]:
        """FIXED: Find time-lagged relationships with non-zero targets"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c != target]
        
        # FIXED: Use non-zero targets
        target_nonzero = df[target] > 0
        
        temporal_patterns = []
        
        for col in feature_cols:
            best_lag = 0
            best_corr = 0
            lag_profile = []
            
            for lag in range(0, max_lag + 1):
                feature_lagged = df[col].shift(lag)
                
                # Apply non-zero mask
                valid_mask = feature_lagged.notna() & df[target].notna() & target_nonzero
                if valid_mask.sum() < 10:
                    continue
                
                try:
                    # FIXED: Use Spearman
                    corr, _ = spearmanr(
                        feature_lagged[valid_mask].values,
                        df[target][valid_mask].values
                    )
                    
                    if np.isnan(corr):
                        corr = 0
                except:
                    corr = 0
                
                lag_profile.append({'lag': lag, 'correlation': float(corr)})
                
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
            
            if abs(best_corr) > min_corr:
                temporal_patterns.append({
                    'variable': col,
                    'best_lag': best_lag,
                    'best_correlation': float(best_corr),
                    'lag_profile': lag_profile,
                    'prediction_strength': abs(best_corr),
                    'direction': 'positive' if best_corr > 0 else 'negative'
                })
        
        return sorted(temporal_patterns, key=lambda x: x['prediction_strength'], reverse=True)
    
    def compute_feature_importance(self, df: pd.DataFrame, target: str) -> Dict[str, float]:
        """Use Random Forest to get feature importance"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c != target]
        
        if len(feature_cols) == 0 or target not in df.columns:
            return {}
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[target].fillna(df[target].median())
        
        if len(X) < 10:
            return {}
        
        try:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)
            
            importance = {}
            for col, imp in zip(feature_cols, model.feature_importances_):
                importance[col] = float(imp)
            
            return importance
        except:
            return {}


class SemanticKnowledgeGraph:
    """COMPLETE knowledge graph with all features + fixes"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.ontology = VariableOntology()
        self.pattern_analyzer = PatternBasedAnalyzer()
        self.variable_registry: Dict[str, VariableMetadata] = {}
        self.relationship_patterns: Dict[str, Dict] = {}
        self.pattern_analysis_results: Dict = {}
        
    def load_data(self, csv_path):
        """Load timeseries data from CSV"""
        df = pd.read_csv(csv_path)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        elif all(col in df.columns for col in ['year', 'month', 'dayofmonth']):
            print("Constructing datetime from year/month/day columns...")
            df['date'] = pd.to_datetime(df[['year', 'month', 'dayofmonth']].rename(
                columns={'dayofmonth': 'day'}))
            df = df.set_index('date')
            df = df.drop(columns=['year', 'month', 'dayofmonth', 'dayofyear', 
                                 'weekofyear', 'quarter', 'dayofweek'], errors='ignore')
        elif df.index.name is None or df.index.name == 'Unnamed: 0':
            try:
                df.index = pd.to_datetime(df.index)
            except:
                print("Warning: Could not parse index as datetime")
        
        return df
    
    def analyze_and_register_variables(self, df: pd.DataFrame):
        """FIXED: Analyze all variables using improved classification"""
        print("\n" + "="*60)
        print("ANALYZING VARIABLES (FIXED SEMANTIC + PATTERN-BASED)")
        print("="*60)
        
        skip_columns = ['Fieldname', 'datetime', 'lookupEncoded', 'FarmEncoded', 
                       'crop', 'variety', 'cultivar']
        
        for col in df.columns:
            if col in skip_columns:
                continue
            
            # Extract data
            data_sample = df[col].dropna().values
            if len(data_sample) == 0:
                continue
            
            data_range = (float(np.nanmin(data_sample)), float(np.nanmax(data_sample)))
            
            # Create statistical fingerprint
            fingerprint = self.pattern_analyzer.create_variable_fingerprint(data_sample, col)
            
            # Try to infer unit from column name
            unit = self._extract_unit_from_name(col)
            
            # FIXED: Semantic classification
            category, subcategory, confidence = self.ontology.classify_variable(
                col, unit, data_sample
            )
            
            # Create metadata
            metadata = VariableMetadata(
                name=col,
                unit=unit,
                data_range=data_range,
                measurement_type=self._infer_measurement_type(category),
                semantic_category=category,
                aliases=[col.lower(), col.replace('_', ''), col.replace(' ', '')],
                physical_meaning=f"{subcategory} measurement",
                statistical_fingerprint=fingerprint
            )
            
            self.variable_registry[col] = metadata
            
            # Display classification
            conf_indicator = "✓" if confidence > 0.5 else "?" if confidence > 0.2 else "✗"
            print(f"  {conf_indicator} {col:30s} → {category:20s} ({subcategory:15s}) "
                  f"[semantic: {confidence:.2f}]")
            if unit:
                print(f"      Unit: {unit}, Range: [{data_range[0]:.2f}, {data_range[1]:.2f}], "
                      f"CV: {fingerprint.get('coefficient_of_variation', 0):.2f}")
        
        print(f"\nRegistered {len(self.variable_registry)} variables")
        
        # Create ontology nodes
        self._create_ontology_nodes()
    
    def perform_pattern_analysis(self, df: pd.DataFrame, target: str):
        """FIXED: Perform comprehensive pattern-based analysis"""
        print("\n" + "="*60)
        print("PATTERN-BASED ANALYSIS (FIXED)")
        print("="*60)
        
        # Check target distribution
        target_data = df[target]
        n_nonzero = (target_data > 0).sum()
        n_zero = (target_data == 0).sum()
        print(f"\nTarget '{target}' distribution:")
        print(f"  Non-zero values: {n_nonzero} ({n_nonzero/len(target_data)*100:.1f}%)")
        print(f"  Zero values: {n_zero} ({n_zero/len(target_data)*100:.1f}%)")
        print(f"  Mean: {target_data.mean():.2f}, Median: {target_data.median():.2f}")
        
        results = {}
        
        # 1. Predictive power analysis
        print("\n[1/5] Analyzing predictive power...")
        predictive_vars = self.pattern_analyzer.discover_predictive_variables(df, target, min_importance=0.05)
        results['predictive_vars'] = predictive_vars
        print(f"  Found {len(predictive_vars)} predictive variables")
        
        if len(predictive_vars) > 0:
            print("  Top 5 predictors:")
            for pv in predictive_vars[:5]:
                print(f"    {pv['variable']:30s} importance: {pv['combined_importance']:.3f} "
                      f"({pv['relationship_type']})")
        
        # Update variable metadata with predictive power
        for pv in predictive_vars:
            if pv['variable'] in self.variable_registry:
                self.variable_registry[pv['variable']].predictive_power = pv['combined_importance']
        
        # 2. Behavioral clustering
        print("\n[2/5] Clustering variables by behavior...")
        clusters = self.pattern_analyzer.cluster_variables_by_behavior(df, target)
        results['behavior_clusters'] = clusters
        
        num_valid_clusters = len([c for c in clusters if c >= 0])
        print(f"  Found {num_valid_clusters} behavioral clusters")
        
        for cluster_id, variables in clusters.items():
            if cluster_id >= 0 and len(variables) > 1:
                print(f"    Cluster {cluster_id}: {len(variables)} variables")
                # Update metadata
                for var in variables:
                    if var in self.variable_registry:
                        self.variable_registry[var].behavior_cluster = cluster_id
        
        # 3. Temporal patterns
        print("\n[3/5] Discovering temporal patterns...")
        temporal = self.pattern_analyzer.discover_temporal_predictors(df, target, min_corr=0.15)
        results['temporal_patterns'] = temporal
        print(f"  Found {len(temporal)} time-lagged predictors")
        
        if len(temporal) > 0:
            print("  Top 5 temporal predictors:")
            for tp in temporal[:5]:
                print(f"    {tp['variable']:30s} lag: {tp['best_lag']} weeks, "
                      f"corr: {tp['best_correlation']:+.3f}")
        
        # Update metadata with temporal signatures
        for tp in temporal:
            if tp['variable'] in self.variable_registry:
                self.variable_registry[tp['variable']].temporal_signature = {
                    'best_lag': tp['best_lag'],
                    'best_correlation': tp['best_correlation'],
                    'lag_profile': tp['lag_profile']
                }
        
        # 4. Feature importance
        print("\n[4/5] Computing feature importance...")
        importance = self.pattern_analyzer.compute_feature_importance(df, target)
        results['feature_importance'] = importance
        
        if len(importance) > 0:
            print(f"  Computed importance for {len(importance)} features")
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            print("  Top 5 most important:")
            for var, imp in sorted_importance[:5]:
                print(f"    {var:30s} {imp:.3f}")
        
        # 5. Variable similarity analysis
        print("\n[5/5] Analyzing variable similarities...")
        similarity_matrix = self._compute_variable_similarities()
        results['similarity_matrix'] = similarity_matrix
        
        # Find groups of similar variables
        similar_groups = self._find_similar_variable_groups(similarity_matrix, threshold=0.7)
        results['similar_groups'] = similar_groups
        
        if similar_groups:
            print(f"  Found {len(similar_groups)} groups of similar variables:")
            for i, group in enumerate(similar_groups[:5]):
                print(f"    Group {i+1}: {', '.join(group)}")
        
        self.pattern_analysis_results = results
        
        print("\n✓ Pattern analysis complete!")
        return results
    
    def _compute_variable_similarities(self) -> Dict[Tuple[str, str], float]:
        """Compute pairwise similarities between all variables"""
        similarity_matrix = {}
        
        var_names = list(self.variable_registry.keys())
        
        for i, var1 in enumerate(var_names):
            for var2 in var_names[i+1:]:
                fp1 = self.variable_registry[var1].statistical_fingerprint
                fp2 = self.variable_registry[var2].statistical_fingerprint
                
                if fp1 and fp2:
                    similarity = self.pattern_analyzer.compare_fingerprints(fp1, fp2)
                    similarity_matrix[(var1, var2)] = similarity
                    similarity_matrix[(var2, var1)] = similarity
        
        return similarity_matrix
    
    def _find_similar_variable_groups(self, similarity_matrix: Dict, 
                                     threshold: float = 0.7) -> List[List[str]]:
        """Find groups of variables that are highly similar"""
        var_names = list(self.variable_registry.keys())
        
        # Build similarity graph
        sim_graph = nx.Graph()
        sim_graph.add_nodes_from(var_names)
        
        for (var1, var2), similarity in similarity_matrix.items():
            if similarity >= threshold:
                sim_graph.add_edge(var1, var2, weight=similarity)
        
        # Find connected components
        groups = list(nx.connected_components(sim_graph))
        
        # Filter to groups with more than 1 member
        return [list(group) for group in groups if len(group) > 1]
    
    def _extract_unit_from_name(self, name: str) -> Optional[str]:
        """Try to extract unit from variable name"""
        patterns = [
            r'_([a-z]+)',
            r'\(([^)]+)\)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, name.lower())
            if match:
                return match.group(1)
        
        return None
    
    def _infer_measurement_type(self, category: str) -> str:
        """Infer measurement type from category"""
        mapping = {
            'Temperature': 'environmental',
            'Moisture': 'environmental',
            'Precipitation': 'environmental',
            'Wind': 'environmental',
            'Pressure': 'environmental',
            'Nutrient': 'soil',
            'Vegetation_Index': 'plant',
            'Yield': 'yield',
            'Growth_Metric': 'plant',
            'Location_Property': 'metadata',
            'Encoded_Variable': 'metadata',
            'Shifted_Variable': 'derived'
        }
        return mapping.get(category, 'unknown')
    
    def _create_ontology_nodes(self):
        """Create nodes for semantic categories in the graph"""
        print("\nCreating ontology structure...")
        
        for category in self.ontology.ontology.keys():
            cat_id = f"Concept_{category}"
            self.graph.add_node(cat_id,
                               type="Concept",
                               category=category,
                               level="semantic_category")
            
            subcategories = self.ontology.ontology[category]['subcategories']
            for subcat in subcategories.keys():
                subcat_id = f"Concept_{category}_{subcat}"
                self.graph.add_node(subcat_id,
                                   type="Concept",
                                   category=subcat,
                                   level="subcategory")
                self.graph.add_edge(subcat_id, cat_id, relationship="IS_A")
        
        concept_nodes = len([n for n, d in self.graph.nodes(data=True) if d.get('type')=='Concept'])
        print(f"  Created {concept_nodes} concept nodes")
    
    def create_graph(self, df, field_name=None):
        """Convert timeseries dataframe to semantic knowledge graph"""
        
        # First, analyze variables
        self.analyze_and_register_variables(df)
        
        print("\n" + "="*60)
        print("BUILDING GRAPH STRUCTURE")
        print("="*60)
        
        # Detect crop information
        has_crop_info = any(col in df.columns for col in ['crop', 'Crop', 'crop_type'])
        crop_column = None
        if 'crop' in df.columns:
            crop_column = 'crop'
        elif 'Crop' in df.columns:
            crop_column = 'Crop'
        elif 'crop_type' in df.columns:
            crop_column = 'crop_type'
        
        if has_crop_info:
            unique_crops = df[crop_column].unique()
            print(f"Detected {len(unique_crops)} crops: {list(unique_crops)}")
            
            for crop_name in unique_crops:
                if pd.notna(crop_name):
                    crop_id = f"Crop_{str(crop_name).replace(' ', '_')}"
                    self.graph.add_node(crop_id,
                                       type="Crop",
                                       name=str(crop_name))
        
        # Detect fields/plots
        has_multiple_fields = 'lookupEncoded' in df.columns or 'FarmEncoded' in df.columns
        
        if has_multiple_fields:
            print(f"Detected multiple fields/plots in dataset")
            if 'lookupEncoded' in df.columns:
                unique_fields = df['lookupEncoded'].unique()
                print(f"Found {len(unique_fields)} unique plots")
            else:
                unique_fields = [0]
        else:
            unique_fields = [0]
        
        # Process each field
        for field_idx, field_code in enumerate(unique_fields):
            if has_multiple_fields and 'lookupEncoded' in df.columns:
                field_df = df[df['lookupEncoded'] == field_code]
                field_id = f"Field_Plot_{int(field_code)}"
                
                if 'FarmEncoded' in field_df.columns:
                    farm_code = field_df['FarmEncoded'].iloc[0]
                    field_name = f"Farm{int(farm_code)}_Plot{int(field_code)}"
                else:
                    field_name = f"Plot_{int(field_code)}"
                
                print(f"  Processing {field_name} with {len(field_df)} timestamps...")
            else:
                field_df = df
                field_id = f"Field_{field_name.replace(' ', '_')}" if field_name else "Field_1"
                field_name = field_name or "Field_1"
            
            # Determine crop for this field
            field_crop = None
            if has_crop_info and crop_column:
                field_crop = field_df[crop_column].mode()[0] if len(field_df) > 0 else None
            
            # Create field node
            field_attrs = {
                'type': "Field",
                'name': field_name,
                'lookup_code': int(field_code) if has_multiple_fields else None
            }
            if field_crop:
                field_attrs['crop'] = str(field_crop)
            
            self.graph.add_node(field_id, **field_attrs)
            
            # Link field to crop
            if field_crop:
                crop_id = f"Crop_{str(field_crop).replace(' ', '_')}"
                if crop_id in self.graph:
                    self.graph.add_edge(crop_id, field_id, relationship="GROWN_IN")
            
            # Create timestamps and measurements
            for idx, row in field_df.iterrows():
                timestamp = idx if isinstance(idx, (pd.Timestamp, datetime)) else pd.to_datetime(idx)
                ts_id = f"TS_{field_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
                
                # Create timestamp node
                ts_attrs = {
                    'type': "Timestamp",
                    'datetime': str(timestamp),
                    'year': timestamp.year,
                    'month': timestamp.month,
                    'day': timestamp.day,
                    'dayofweek': timestamp.dayofweek,
                    'dayofyear': timestamp.dayofyear,
                    'field': field_id
                }
                if field_crop:
                    ts_attrs['crop'] = str(field_crop)
                
                self.graph.add_node(ts_id, **ts_attrs)
                self.graph.add_edge(field_id, ts_id, relationship="HAS_MEASUREMENT")
                
                # Create measurement nodes with semantic links
                for col in field_df.columns:
                    if col in ['Fieldname', 'datetime', 'lookupEncoded', 'FarmEncoded', 
                              'crop', 'Crop', 'crop_type', 'variety']:
                        continue
                    
                    value = row[col]
                    if pd.notna(value):
                        measure_id = f"Measure_{ts_id}_{col}"
                        
                        # Get variable metadata
                        var_metadata = self.variable_registry.get(col)
                        
                        measure_attrs = {
                            'type': "Measurement",
                            'metric': col,
                            'value': float(value)
                        }
                        
                        if var_metadata:
                            measure_attrs['semantic_category'] = var_metadata.semantic_category
                            measure_attrs['measurement_type'] = var_metadata.measurement_type
                            measure_attrs['predictive_power'] = var_metadata.predictive_power
                            measure_attrs['behavior_cluster'] = var_metadata.behavior_cluster
                            if var_metadata.unit:
                                measure_attrs['unit'] = var_metadata.unit
                        
                        self.graph.add_node(measure_id, **measure_attrs)
                        
                        # Link to timestamp
                        self.graph.add_edge(ts_id, measure_id,
                                           relationship="HAS_VALUE",
                                           metric=col)
                        
                        # Link to semantic concept
                        if var_metadata:
                            concept_id = f"Concept_{var_metadata.semantic_category}"
                            if concept_id in self.graph:
                                self.graph.add_edge(measure_id, concept_id,
                                                   relationship="INSTANCE_OF")
            
            # Create temporal sequence
            field_timestamps = sorted([n for n, d in self.graph.nodes(data=True)
                                      if d.get('type') == 'Timestamp' and d.get('field') == field_id],
                                     key=lambda x: self.graph.nodes[x]['datetime'])
            
            for i in range(len(field_timestamps) - 1):
                self.graph.add_edge(field_timestamps[i], field_timestamps[i+1],
                                   relationship="NEXT")
        
        # Create temporal hierarchy
        self._create_temporal_hierarchy(df)
        
        # Create farm hierarchy if needed
        if 'FarmEncoded' in df.columns and 'lookupEncoded' in df.columns:
            self._create_farm_hierarchy(df)
        
        print(f"\nGraph created:")
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {self.graph.number_of_edges()}")
        
        return self.graph
    
    def learn_variable_relationships(self, df: pd.DataFrame, target_var: str = 'target',
                                    min_correlation: float = 0.15):
        """FIXED: Learn mathematical relationships between variables"""
        print("\n" + "="*60)
        print("LEARNING VARIABLE RELATIONSHIPS (FIXED)")
        print("="*60)
        
        if target_var not in df.columns:
            print(f"Warning: Target variable '{target_var}' not found")
            return []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c != target_var and c not in 
                       ['lookupEncoded', 'FarmEncoded']]
        
        print(f"Analyzing relationships for {len(feature_cols)} features → {target_var}")
        
        # FIXED: Use non-zero targets
        target_nonzero = df[target_var] > 0
        n_nonzero = target_nonzero.sum()
        
        if n_nonzero < 10:
            print(f"Warning: Only {n_nonzero} non-zero targets, using all data")
            df_analysis = df
        else:
            df_analysis = df[target_nonzero]
            print(f"Using {len(df_analysis)} non-zero target samples")
        
        relationships = []
        
        for feature in feature_cols:
            valid_mask = df_analysis[[feature, target_var]].notna().all(axis=1)
            if valid_mask.sum() < 10:
                continue
            
            feature_data = df_analysis.loc[valid_mask, feature].values
            target_data = df_analysis.loc[valid_mask, target_var].values
            
            # Skip if no variance
            if np.std(feature_data) == 0:
                continue
            
            # FIXED: Use Spearman correlation
            correlation, p_value = spearmanr(feature_data, target_data)
            
            if np.isnan(correlation):
                continue
            
            if abs(correlation) >= min_correlation:
                # Get variable metadata
                feature_meta = self.variable_registry.get(feature)
                target_meta = self.variable_registry.get(target_var)
                
                relationship = {
                    'source': feature,
                    'target': target_var,
                    'correlation': float(correlation),
                    'p_value': float(p_value),
                    'relationship_type': 'positive' if correlation > 0 else 'negative',
                    'strength': abs(correlation),
                    'source_category': feature_meta.semantic_category if feature_meta else 'Unknown',
                    'target_category': target_meta.semantic_category if target_meta else 'Unknown',
                    'source_cluster': feature_meta.behavior_cluster if feature_meta else -1,
                    'predictive_power': feature_meta.predictive_power if feature_meta else 0.0,
                    'n_samples': int(valid_mask.sum())
                }
                
                relationships.append(relationship)
                
                # Store in graph
                self.relationship_patterns[f"{feature}→{target_var}"] = relationship
                
                print(f"  {feature:30s} → {target_var:15s}: {correlation:+.3f} "
                      f"({feature_meta.semantic_category if feature_meta else '?'}, "
                      f"p={p_value:.3f})")
        
        print(f"\nFound {len(relationships)} significant relationships")
        
        # Store relationships as edges between concepts
        self._create_concept_relationships(relationships)
        
        return relationships
    
    def _create_concept_relationships(self, relationships: List[Dict]):
        """Create edges between semantic concepts based on learned relationships"""
        print("\nCreating concept-level relationships...")
        
        # Group by semantic categories
        concept_pairs = {}
        for rel in relationships:
            source_cat = rel['source_category']
            target_cat = rel['target_category']
            
            key = (source_cat, target_cat)
            if key not in concept_pairs:
                concept_pairs[key] = []
            concept_pairs[key].append(rel['correlation'])
        
        # Create weighted edges between concepts
        for (source_cat, target_cat), correlations in concept_pairs.items():
            source_id = f"Concept_{source_cat}"
            target_id = f"Concept_{target_cat}"
            
            if source_id in self.graph and target_id in self.graph:
                avg_correlation = np.mean(correlations)
                edge_attrs = {
                    'relationship': 'PREDICTS',
                    'weight': abs(avg_correlation),
                    'correlation': float(avg_correlation),
                    'sample_size': len(correlations)
                }
                self.graph.add_edge(source_id, target_id, **edge_attrs)
                print(f"  {source_cat} → {target_cat}: {avg_correlation:+.3f} (n={len(correlations)})")
    
    def find_transferable_knowledge(self, new_var_name: str, 
                                   new_var_data: Optional[np.ndarray] = None,
                                   new_crop: Optional[str] = None) -> Dict:
        """When a new variable appears, find what we know that could transfer"""
        print(f"\n{'='*60}")
        print(f"TRANSFER LEARNING: {new_var_name}")
        print(f"{'='*60}")
        
        results = {
            'variable_name': new_var_name,
            'semantic_match': None,
            'pattern_matches': [],
            'transferable_relationships': [],
            'recommendations': []
        }
        
        # 1. Semantic classification
        print("\n[1/3] Semantic Classification")
        category, subcategory, confidence = self.ontology.classify_variable(
            new_var_name, 
            data_sample=new_var_data if new_var_data is not None else None
        )
        
        print(f"  Category: {category} ({subcategory})")
        print(f"  Confidence: {confidence:.2f}")
        
        results['semantic_match'] = {
            'category': category,
            'subcategory': subcategory,
            'confidence': float(confidence)
        }
        
        # 2. Pattern-based matching (if we have data)
        if new_var_data is not None:
            print("\n[2/3] Pattern-Based Matching")
            new_fingerprint = self.pattern_analyzer.create_variable_fingerprint(
                new_var_data, new_var_name
            )
            
            # Compare with all known variables
            pattern_matches = []
            for known_var, metadata in self.variable_registry.items():
                if metadata.statistical_fingerprint:
                    similarity = self.pattern_analyzer.compare_fingerprints(
                        new_fingerprint,
                        metadata.statistical_fingerprint
                    )
                    
                    if similarity > 0.5:
                        pattern_matches.append({
                            'variable': known_var,
                            'similarity': float(similarity),
                            'category': metadata.semantic_category,
                            'predictive_power': metadata.predictive_power
                        })
            
            pattern_matches = sorted(pattern_matches, key=lambda x: x['similarity'], reverse=True)
            results['pattern_matches'] = pattern_matches[:10]
            
            print(f"  Found {len(pattern_matches)} similar variables:")
            for match in pattern_matches[:5]:
                print(f"    {match['variable']:30s} similarity: {match['similarity']:.3f} "
                      f"({match['category']})")
        else:
            print("\n[2/3] Pattern-Based Matching: SKIPPED (no data provided)")
        
        # 3. Find transferable relationships
        print("\n[3/3] Transferable Relationships")
        
        # From semantic matches
        semantic_transfers = []
        for pattern_key, pattern_data in self.relationship_patterns.items():
            if pattern_data['source_category'] == category:
                semantic_transfers.append({
                    'source': pattern_data['source'],
                    'target': pattern_data['target'],
                    'correlation': pattern_data['correlation'],
                    'transfer_basis': 'semantic_category',
                    'confidence': confidence
                })
        
        # From pattern matches
        pattern_transfers = []
        if new_var_data is not None and results['pattern_matches']:
            for match in results['pattern_matches'][:3]:
                for pattern_key, pattern_data in self.relationship_patterns.items():
                    if pattern_data['source'] == match['variable']:
                        pattern_transfers.append({
                            'source': match['variable'],
                            'target': pattern_data['target'],
                            'correlation': pattern_data['correlation'],
                            'transfer_basis': 'statistical_similarity',
                            'confidence': match['similarity']
                        })
        
        all_transfers = semantic_transfers + pattern_transfers
        # Deduplicate and sort
        seen = set()
        unique_transfers = []
        for t in sorted(all_transfers, key=lambda x: abs(x['correlation']), reverse=True):
            key = (t['source'], t['target'])
            if key not in seen:
                seen.add(key)
                unique_transfers.append(t)
        
        results['transferable_relationships'] = unique_transfers[:10]
        
        print(f"  Found {len(unique_transfers)} transferable relationships:")
        for transfer in unique_transfers[:5]:
            print(f"    {transfer['source']} → {transfer['target']}: "
                  f"{transfer['correlation']:+.3f} "
                  f"(via {transfer['transfer_basis']}, conf: {transfer['confidence']:.2f})")
        
        # 4. Generate recommendations
        print("\n[4/4] Recommendations")
        recommendations = []
        
        if confidence > 0.5:
            recommendations.append(
                f"✓ Strong semantic match to {category}. Can confidently transfer {len(semantic_transfers)} relationships."
            )
        elif confidence > 0.2:
            recommendations.append(
                f"⚠ Weak semantic match to {category}. Verify before using."
            )
        else:
            recommendations.append(
                f"✗ No clear semantic category. Rely on pattern matching."
            )
        
        if new_var_data is not None and len(results['pattern_matches']) > 0:
            best_match = results['pattern_matches'][0]
            recommendations.append(
                f"✓ Statistically similar to '{best_match['variable']}' "
                f"(similarity: {best_match['similarity']:.2f})"
            )
        
        if len(unique_transfers) > 0:
            recommendations.append(
                f"✓ Can transfer {len(unique_transfers)} relationship patterns to predict outcomes"
            )
        
        results['recommendations'] = recommendations
        
        for rec in recommendations:
            print(f"  {rec}")
        
        return results
    
    def _create_temporal_hierarchy(self, df):
        """Create year/month/week nodes"""
        ts_nodes = [(n, d) for n, d in self.graph.nodes(data=True)
                    if d.get('type') == 'Timestamp']
        
        years, months, weeks = {}, {}, {}
        
        for ts_id, ts_data in ts_nodes:
            year = ts_data['year']
            month = ts_data['month']
            timestamp = pd.to_datetime(ts_data['datetime'])
            week = timestamp.isocalendar()[1]
            
            year_id = f"Year_{year}"
            if year_id not in years:
                self.graph.add_node(year_id, type="Year", year=year)
                years[year_id] = True
            self.graph.add_edge(ts_id, year_id, relationship="IN_YEAR")
            
            month_id = f"Month_{year}_{month:02d}"
            if month_id not in months:
                self.graph.add_node(month_id, type="Month", year=year, month=month)
                self.graph.add_edge(month_id, year_id, relationship="PART_OF_YEAR")
                months[month_id] = True
            self.graph.add_edge(ts_id, month_id, relationship="IN_MONTH")
            
            week_id = f"Week_{year}_W{week:02d}"
            if week_id not in weeks:
                self.graph.add_node(week_id, type="Week", year=year, week=week)
                self.graph.add_edge(week_id, year_id, relationship="PART_OF_YEAR")
                weeks[week_id] = True
            self.graph.add_edge(ts_id, week_id, relationship="IN_WEEK")
    
    def _create_farm_hierarchy(self, df):
        """Create farm hierarchy"""
        print("Creating farm hierarchy...")
        unique_farms = df['FarmEncoded'].unique()
        
        for farm_code in unique_farms:
            farm_id = f"Farm_{int(farm_code)}"
            if farm_id not in self.graph:
                self.graph.add_node(farm_id, type="Farm", farm_code=int(farm_code))
            
            farm_plots = df[df['FarmEncoded'] == farm_code]['lookupEncoded'].unique()
            for plot_code in farm_plots:
                plot_id = f"Field_Plot_{int(plot_code)}"
                if plot_id in self.graph:
                    self.graph.add_edge(farm_id, plot_id, relationship="CONTAINS_PLOT")
    
    def export_variable_registry(self, output_file='outputs/variable_registry.json'):
        """Export variable registry for inspection"""
        registry_export = {}
        for var_name, metadata in self.variable_registry.items():
            registry_export[var_name] = {
                'semantic_category': metadata.semantic_category,
                'measurement_type': metadata.measurement_type,
                'unit': metadata.unit,
                'range': list(metadata.data_range),
                'physical_meaning': metadata.physical_meaning,
                'predictive_power': metadata.predictive_power,
                'behavior_cluster': metadata.behavior_cluster,
                'statistical_fingerprint': metadata.statistical_fingerprint
            }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(registry_export, f, indent=2)
        print(f"\nVariable registry exported to: {output_file}")
    
    def export_relationships(self, output_file='outputs/variable_relationships.json'):
        """Export learned relationships"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(self.relationship_patterns, f, indent=2)
        print(f"Relationships exported to: {output_file}")
    
    def export_pattern_analysis(self, output_file='outputs/pattern_analysis.json'):
        """Export pattern analysis results"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert to serializable format
        export_data = {}
        for key, value in self.pattern_analysis_results.items():
            if key == 'similarity_matrix':
                # Convert tuple keys to strings
                export_data[key] = {f"{k[0]}→{k[1]}": v for k, v in value.items()}
            else:
                export_data[key] = value
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"Pattern analysis exported to: {output_file}")
    
    def get_statistics(self):
        """Get graph statistics"""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': {},
            'variables_registered': len(self.variable_registry),
            'relationships_learned': len(self.relationship_patterns)
        }
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        return stats
    
    def generate_comprehensive_report(self, output_file='outputs/comprehensive_report.txt'):
        """Generate comprehensive report"""
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FIXED PATTERN-BASED SEMANTIC KNOWLEDGE GRAPH REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Statistics
            stats = self.get_statistics()
            f.write("GRAPH STATISTICS:\n")
            f.write("-" * 70 + "\n")
            for key, value in stats.items():
                if key != 'node_types':
                    f.write(f"  {key}: {value}\n")
            f.write("\nNODE TYPES:\n")
            for node_type, count in sorted(stats['node_types'].items()):
                f.write(f"  {node_type}: {count}\n")
            
            # Variable categories
            f.write("\n" + "="*70 + "\n")
            f.write("VARIABLE ANALYSIS\n")
            f.write("="*70 + "\n")
            
            categories = {}
            for var_name, metadata in self.variable_registry.items():
                cat = metadata.semantic_category
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(var_name)
            
            for category, variables in sorted(categories.items()):
                f.write(f"\n{category} ({len(variables)} variables):\n")
                f.write("-" * 70 + "\n")
                for var in sorted(variables):
                    meta = self.variable_registry[var]
                    f.write(f"  {var:30s} ")
                    f.write(f"[{meta.unit or 'no unit':8s}] ")
                    f.write(f"range: [{meta.data_range[0]:8.2f}, {meta.data_range[1]:8.2f}] ")
                    f.write(f"pred: {meta.predictive_power:.3f} ")
                    f.write(f"cluster: {meta.behavior_cluster}\n")
            
            # Pattern analysis results
            if self.pattern_analysis_results:
                f.write("\n" + "="*70 + "\n")
                f.write("PATTERN ANALYSIS RESULTS\n")
                f.write("="*70 + "\n")
                
                # Predictive variables
                if 'predictive_vars' in self.pattern_analysis_results:
                    f.write("\nTop Predictive Variables:\n")
                    f.write("-" * 70 + "\n")
                    for pv in self.pattern_analysis_results['predictive_vars'][:20]:
                        f.write(f"  {pv['variable']:30s} ")
                        f.write(f"importance: {pv['combined_importance']:.3f} ")
                        f.write(f"spearman: {pv['spearman']:+.3f} ")
                        f.write(f"({pv['relationship_type']})\n")
                
                # Behavioral clusters
                if 'behavior_clusters' in self.pattern_analysis_results:
                    f.write("\nBehavioral Clusters:\n")
                    f.write("-" * 70 + "\n")
                    for cluster_id, variables in self.pattern_analysis_results['behavior_clusters'].items():
                        if cluster_id >= 0:
                            f.write(f"\n  Cluster {cluster_id} ({len(variables)} variables):\n")
                            for var in variables[:10]:
                                f.write(f"    - {var}\n")
                            if len(variables) > 10:
                                f.write(f"... and {len(variables) - 10} more\n")
                
                # Temporal patterns
                if 'temporal_patterns' in self.pattern_analysis_results:
                    f.write("\nTemporal Predictors (Top 20):\n")
                    f.write("-" * 70 + "\n")
                    for tp in self.pattern_analysis_results['temporal_patterns'][:20]:
                        f.write(f"  {tp['variable']:30s} ")
                        f.write(f"lag: {tp['best_lag']} weeks, ")
                        f.write(f"corr: {tp['best_correlation']:+.3f} ")
                        f.write(f"({tp['direction']})\n")
                
                # Similar variable groups
                if 'similar_groups' in self.pattern_analysis_results:
                    f.write("\nSimilar Variable Groups:\n")
                    f.write("-" * 70 + "\n")
                    for i, group in enumerate(self.pattern_analysis_results['similar_groups']):
                        f.write(f"\n  Group {i+1} ({len(group)} variables):\n")
                        for var in group:
                            f.write(f"    - {var}\n")
            
            # Relationships
            if self.relationship_patterns:
                f.write("\n" + "="*70 + "\n")
                f.write("LEARNED RELATIONSHIPS (Top 30)\n")
                f.write("="*70 + "\n")
                
                sorted_rels = sorted(self.relationship_patterns.items(),
                                   key=lambda x: abs(x[1]['correlation']),
                                   reverse=True)
                
                for rel_key, rel_data in sorted_rels[:30]:
                    f.write(f"\n{rel_data['source']} → {rel_data['target']}\n")
                    f.write(f"  Correlation: {rel_data['correlation']:+.3f}\n")
                    f.write(f"  Type: {rel_data['relationship_type']}\n")
                    f.write(f"  Categories: {rel_data['source_category']} → {rel_data['target_category']}\n")
                    f.write(f"  Predictive Power: {rel_data['predictive_power']:.3f}\n")
                    f.write(f"  Samples: {rel_data.get('n_samples', 'N/A')}\n")
            
            # Concept relationships
            f.write("\n" + "="*70 + "\n")
            f.write("CONCEPT-LEVEL RELATIONSHIPS\n")
            f.write("="*70 + "\n")
            
            concept_edges = [(u, v, d) for u, v, d in self.graph.edges(data=True)
                           if d.get('relationship') == 'PREDICTS']
            
            for source, target, data in sorted(concept_edges, 
                                              key=lambda x: abs(x[2].get('correlation', 0)),
                                              reverse=True):
                source_name = source.replace('Concept_', '')
                target_name = target.replace('Concept_', '')
                f.write(f"\n{source_name} → {target_name}\n")
                f.write(f"  Avg Correlation: {data['correlation']:+.3f}\n")
                f.write(f"  Weight: {data['weight']:.3f}\n")
                f.write(f"  Based on: {data['sample_size']} variable pairs\n")
        
        print(f"Comprehensive report saved to: {output_file}")
    
    def save_graph(self, filepath='outputs/graphs/semantic_knowledge_graph.pkl'):
        """Save graph and all metadata"""
        import pickle
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_data = {
            'graph': self.graph,
            'variable_registry': self.variable_registry,
            'relationship_patterns': self.relationship_patterns,
            'pattern_analysis_results': self.pattern_analysis_results,
            'stats': self.get_statistics()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Semantic knowledge graph saved to: {filepath}")
    
    def load_graph(self, filepath='outputs/graphs/semantic_knowledge_graph.pkl'):
        """Load graph and metadata"""
        import pickle
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.graph = save_data['graph']
        self.variable_registry = save_data.get('variable_registry', {})
        self.relationship_patterns = save_data.get('relationship_patterns', {})
        self.pattern_analysis_results = save_data.get('pattern_analysis_results', {})
        
        print(f"Semantic knowledge graph loaded from: {filepath}")
        print(f"  Variables: {len(self.variable_registry)}")
        print(f"  Relationships: {len(self.relationship_patterns)}")
        return self.graph
    
    def export_sample_json(self, output_file='outputs/exports/kg_sample.json', num_nodes=100):
        """Export sample as JSON"""
        nodes = list(self.graph.nodes())[:num_nodes]
        subgraph = self.graph.subgraph(nodes)
        
        data = {'nodes': [], 'edges': []}
        for node, attrs in subgraph.nodes(data=True):
            node_data = {'id': node}
            # Convert non-serializable types
            for k, v in attrs.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    node_data[k] = v
                else:
                    node_data[k] = str(v)
            data['nodes'].append(node_data)
        
        for src, dst, attrs in subgraph.edges(data=True):
            edge_data = {'source': src, 'target': dst}
            for k, v in attrs.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    edge_data[k] = v
                else:
                    edge_data[k] = str(v)
            data['edges'].append(edge_data)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Sample graph exported to {output_file}")
    
    def visualize_subgraph(self, num_nodes=50, output_file='outputs/visualizations/kg_visualization.png'):
        """2D visualization"""
        nodes = list(self.graph.nodes())[:num_nodes]
        subgraph = self.graph.subgraph(nodes)
        
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
        
        color_map = {
            'Concept': '#9B59B6',
            'Crop': '#E67E22',
            'Field': '#FF6B6B',
            'Farm': '#E74C3C',
            'Timestamp': '#4ECDC4',
            'Measurement': '#95E1D3',
            'Year': '#FFA07A',
            'Month': '#FFD93D',
            'Week': '#A8DADC'
        }
        node_colors = [color_map.get(self.graph.nodes[n].get('type'), '#CCCCCC')
                      for n in subgraph.nodes()]
        
        nx.draw(subgraph, pos, node_color=node_colors, node_size=300,
               with_labels=False, arrows=True, edge_color='#999999', width=0.5, alpha=0.7)
        
        plt.title("Semantic Knowledge Graph Visualization", fontsize=16)
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {output_file}")
    
    def create_3d_visualization(self, output_file='outputs/visualizations/kg_3d.html',
                               max_nodes=5000, sample_timestamps=10):
        """Create 3D visualization with semantic coloring"""
        
        # Sample nodes intelligently
        sampled_nodes = set()
        
        # Always include concept nodes
        for node, data in self.graph.nodes(data=True):
            if data.get('type') in ['Concept', 'Crop', 'Field', 'Farm', 'Year']:
                sampled_nodes.add(node)
        
        # Sample timestamps
        timestamp_nodes = [n for n, d in self.graph.nodes(data=True)
                         if d.get('type') == 'Timestamp']
        sampled_nodes.update(timestamp_nodes[::sample_timestamps])
        
        # Add their measurements
        for ts in list(sampled_nodes):
            if self.graph.nodes.get(ts, {}).get('type') == 'Timestamp':
                for neighbor in self.graph.neighbors(ts):
                    if self.graph.nodes[neighbor].get('type') == 'Measurement':
                        sampled_nodes.add(neighbor)
                        # Add semantic concept link
                        for concept_neighbor in self.graph.neighbors(neighbor):
                            if self.graph.nodes[concept_neighbor].get('type') == 'Concept':
                                sampled_nodes.add(concept_neighbor)
        
        subgraph = self.graph.subgraph(list(sampled_nodes))
        
        # Layout
        pos = nx.spring_layout(subgraph, dim=3, k=2.0, iterations=50, seed=42)
        
        # Prepare visualization data
        node_x, node_y, node_z, node_text, node_colors, node_sizes = [], [], [], [], [], []
        color_map = {
            'Concept': '#9B59B6',
            'Crop': '#E67E22',
            'Field': '#FF6B6B',
            'Farm': '#E74C3C',
            'Timestamp': '#4ECDC4',
            'Measurement': '#95E1D3',
            'Year': '#FFA07A',
            'Month': '#FFD93D',
            'Week': '#A8DADC'
        }
        
        for node in subgraph.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            
            node_data = self.graph.nodes[node]
            node_type = node_data.get('type', 'Unknown')
            
            # Create hover text
            hover_text = f"<b>{node}</b><br>Type: {node_type}"
            if node_type == 'Measurement':
                hover_text += f"<br>Category: {node_data.get('semantic_category', 'Unknown')}"
                hover_text += f"<br>Value: {node_data.get('value', 'N/A')}"
                hover_text += f"<br>Predictive Power: {node_data.get('predictive_power', 0):.3f}"
                hover_text += f"<br>Cluster: {node_data.get('behavior_cluster', -1)}"
            elif node_type == 'Concept':
                hover_text += f"<br>Category: {node_data.get('category', 'Unknown')}"
            
            node_text.append(hover_text)
            node_colors.append(color_map.get(node_type, '#CCCCCC'))
            
            base_size = {
                'Concept': 20,
                'Crop': 18,
                'Field': 15,
                'Farm': 20,
                'Timestamp': 8,
                'Measurement': 5
            }.get(node_type, 6)
            node_sizes.append(min(base_size + subgraph.degree(node) * 0.5, 25))
        
        # Prepare edges
        edge_x, edge_y, edge_z = [], [], []
        for edge in subgraph.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
        
        # Create figure
        fig = go.Figure(data=[
            go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='rgba(125,125,125,0.2)', width=1),
                hoverinfo='none',
                name='Relationships'
            ),
            go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode='markers',
                marker=dict(size=node_sizes, color=node_colors, opacity=0.9),
                text=node_text,
                hoverinfo='text',
                name='Nodes'
            )
        ])
        
        fig.update_layout(
            title='Interactive 3D Semantic Knowledge Graph',
            showlegend=False,
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False),
                yaxis=dict(showbackground=False, showticklabels=False),
                zaxis=dict(showbackground=False, showticklabels=False),
                camera=dict(eye=dict(x=2.5, y=2.5, z=1.5))
            ),
            height=900
        )
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        fig.write_html(output_file)
        print(f"3D visualization saved to: {output_file}")


def main():
    print("="*70)
    print("COMPLETE FIXED SEMANTIC KNOWLEDGE GRAPH BUILDER")
    print("="*70)
    
    # CONFIG
    input_file = 'full_dataset.csv'
    target_variable = 'target'
    build_graph = True
    learn_relationships = True
    perform_pattern_analysis = True
    create_visualizations = True
    test_transfer_learning = True
    
    # Create output directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/graphs', exist_ok=True)
    os.makedirs('outputs/visualizations', exist_ok=True)
    os.makedirs('outputs/exports', exist_ok=True)
    
    # Initialize semantic knowledge graph
    kg = SemanticKnowledgeGraph()
    
    if build_graph:
        print("\n[STEP 1] Loading data...")
        df_full = kg.load_data(input_file)
        print(f"Loaded {len(df_full)} rows")
        print(f"Columns: {list(df_full.columns)}")
        print(f"Date range: {df_full.index.min()} to {df_full.index.max()}")
        
        print("\n[STEP 2] Building semantic graph...")
        kg.create_graph(df_full)
        
        if perform_pattern_analysis:
            print("\n[STEP 3] Performing pattern analysis...")
            kg.perform_pattern_analysis(df_full, target_variable)
        
        if learn_relationships:
            print("\n[STEP 4] Learning variable relationships...")
            kg.learn_variable_relationships(df_full, target_var=target_variable, min_correlation=0.15)
        
        print("\n[STEP 5] Exporting results...")
        kg.export_variable_registry('outputs/variable_registry.json')
        kg.export_relationships('outputs/variable_relationships.json')
        kg.export_pattern_analysis('outputs/pattern_analysis.json')
        kg.generate_comprehensive_report('outputs/comprehensive_report.txt')
        kg.export_sample_json('outputs/exports/kg_sample.json')
        kg.save_graph('outputs/graphs/semantic_knowledge_graph.pkl')
        
    else:
        print("\nLoading existing semantic graph...")
        kg.load_graph('outputs/graphs/semantic_knowledge_graph.pkl')
    
    if create_visualizations:
        print("\n[STEP 6] Creating visualizations...")
        kg.visualize_subgraph(num_nodes=100,
                             output_file='outputs/visualizations/semantic_kg_2d.png')
        kg.create_3d_visualization('outputs/visualizations/semantic_kg_3d.html',
                                   sample_timestamps=50)
    
    if test_transfer_learning:
        print("\n[STEP 7] Testing transfer learning capability...")
        print("\n" + "="*70)
        print("TRANSFER LEARNING TESTS")
        print("="*70)
        
        # Test 1: Similar variable with different name
        print("\n--- Test 1: New temperature variable ---")
        kg.find_transferable_knowledge("ambient_temp_celsius")
        
        # Test 2: Variable from different crop
        print("\n--- Test 2: New crop yield variable ---")
        kg.find_transferable_knowledge("tomato_harvest_kg_ha")
        
        # Test 3: Unknown variable type with data
        print("\n--- Test 3: Completely unknown variable (with simulated data) ---")
        # Simulate some data that looks like NDVI
        simulated_ndvi = np.random.uniform(0.2, 0.9, 100)
        kg.find_transferable_knowledge("mystery_sensor_042", new_var_data=simulated_ndvi)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print("\nKey Fixes Applied:")
    print("  ✓ Fixed semantic classification (improved scoring weights)")
    print("  ✓ Use Spearman correlation for skewed data")
    print("  ✓ Analyze only non-zero target values")
    print("  ✓ Lower correlation thresholds (0.05-0.15)")
    print("  ✓ Better handling of shifted/lagged variables")
    print("  ✓ ALL original features preserved")
    print("\nGenerated files:")
    print("  - outputs/variable_registry.json")
    print("  - outputs/variable_relationships.json")
    print("  - outputs/pattern_analysis.json")
    print("  - outputs/comprehensive_report.txt")
    print("  - outputs/graphs/semantic_knowledge_graph.pkl")
    print("  - outputs/visualizations/ (2D and 3D)")
    
    return kg


if __name__ == "__main__":
    kg = main()