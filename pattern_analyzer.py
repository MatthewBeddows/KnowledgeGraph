"""
Pattern-Based Analyzer: Statistical and Temporal Pattern Discovery
Analyzes variables for predictive relationships and behavioral patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor


class PatternBasedAnalyzer:
    """Analyze variables with better handling of skewed targets"""
    
    def __init__(self):
        self.fingerprints = {}
        
    def create_variable_fingerprint(self, data: np.ndarray, name: str = "") -> Dict:
        """
        Create comprehensive statistical signature of a variable
        
        Args:
            data: Array of variable values
            name: Name of the variable
            
        Returns:
            Dictionary containing statistical properties
        """
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
        """
        Compare two statistical fingerprints (0 to 1 similarity)
        
        Args:
            fp1: First fingerprint dictionary
            fp2: Second fingerprint dictionary
            
        Returns:
            Similarity score from 0.0 to 1.0
        """
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
        """
        Find predictive variables using non-zero targets for skewed data
        
        Args:
            df: DataFrame with features and target
            target: Name of target variable
            min_importance: Minimum importance threshold
            
        Returns:
            List of dictionaries with predictive variable information
        """
        predictive_vars = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c != target]
        
        if len(feature_cols) == 0 or target not in df.columns:
            return []
        
        # Filter to non-zero targets for skewed yield data
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
                
                # Use Spearman for skewed data
                spearman_corr, spearman_p = spearmanr(col_data, target_data)
                if np.isnan(spearman_corr):
                    spearman_corr = 0
                
                pearson_corr = np.corrcoef(col_data, target_data)[0, 1]
                if np.isnan(pearson_corr):
                    pearson_corr = 0
                
                # Weight Spearman more for skewed data
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
        """
        Group variables by how they behave together
        
        Args:
            df: DataFrame with variables
            target: Target variable name
            eps: DBSCAN epsilon parameter
            
        Returns:
            Dictionary mapping cluster_id to list of variable names
        """
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
        """
        Find time-lagged relationships with non-zero targets
        
        Args:
            df: DataFrame with time series data
            target: Target variable name
            max_lag: Maximum lag to test
            min_corr: Minimum correlation threshold
            
        Returns:
            List of temporal patterns discovered
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c != target]
        
        # Use non-zero targets
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
                    # Use Spearman
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
        """
        Use Random Forest to get feature importance
        
        Args:
            df: DataFrame with features and target
            target: Target variable name
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
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