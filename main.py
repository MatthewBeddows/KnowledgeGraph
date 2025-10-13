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

#other files
from variable_ontology import VariableOntology, VariableMetadata
from pattern_analyzer import PatternBasedAnalyzer
from gnn_single_pred import run_single_date_forecast

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


"""
COMPLETE MAIN FUNCTION
Replace your existing main() function with this
"""

def main():
    print("="*70)
    print("COMPLETE FIXED SEMANTIC KNOWLEDGE GRAPH BUILDER")
    print("WITH INCREMENTAL ROLLING FORECAST")
    print("="*70)
    
    # ==================== CONFIGURATION ====================
    # Data files
    train_file = 'AngusTrain.csv'      # Training data for knowledge graph
    test_file = 'AngusTest.csv'        # Test data (added incrementally)
    target_variable = 'target'
    
    # Knowledge Graph Building
    build_graph = True
    learn_relationships = True
    perform_pattern_analysis = True
    create_visualizations = False  # Set to False for faster runs
    test_transfer_learning = False  # Set to False for faster runs
    
    # Incremental Rolling Forecast
    run_incremental_forecast = True  # Set to True to run incremental forecast
    forecast_config = {
        'forecast_weeks': 4,           # Predict 4 weeks ahead
        'target_crop': None,           # None = all crops, or specify: 'Tomato', 'Strawberry'
        'use_semantic_filtering': True, # Use KG to filter relevant features
        'use_attention': True,         # Use Graph Attention Networks (GAT)
        'epochs_per_retrain': 100      # Training epochs each week (100-200 recommended)
    }
    # ======================================================
    
    # Create output directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/graphs', exist_ok=True)
    os.makedirs('outputs/visualizations', exist_ok=True)
    os.makedirs('outputs/exports', exist_ok=True)
    os.makedirs('outputs/gnn', exist_ok=True)
    os.makedirs('outputs/gnn/plots', exist_ok=True)
    
    # Initialize semantic knowledge graph
    kg = SemanticKnowledgeGraph()
    
    if build_graph:
        print("\n[STEP 1] Loading training data...")
        # Load training data only for initial KG
        df_train = kg.load_data(train_file)
        print(f"Loaded {len(df_train)} training rows")
        print(f"Columns: {list(df_train.columns)}")
        print(f"Date range: {df_train.index.min()} to {df_train.index.max()}")
        
        print("\n[STEP 2] Building semantic graph from training data...")
        kg.create_graph(df_train)
        
        if perform_pattern_analysis:
            print("\n[STEP 3] Performing pattern analysis...")
            kg.perform_pattern_analysis(df_train, target_variable)
        
        if learn_relationships:
            print("\n[STEP 4] Learning variable relationships...")
            kg.learn_variable_relationships(df_train, target_var=target_variable, 
                                           min_correlation=0.15)
        
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
    
    # ==================== INCREMENTAL ROLLING FORECAST ====================
    if run_incremental_forecast:
        print("\n" + "="*70)
        print("[STEP 8] RUNNING INCREMENTAL ROLLING FORECAST")
        print("="*70)
        print("\n🔄 This simulates real-world deployment:")
        print("  1. Start with training data (AngusTrain.csv)")
        print("  2. Each week:")
        print("     ├─ Add new week from test data (AngusTest.csv)")
        print("     ├─ Rebuild knowledge graph with all data seen so far")
        print("     ├─ Retrain GNN model")
        print("     ├─ Predict 4 weeks ahead for all fields")
        print("     └─ Store predictions")
        print("  3. Evaluate all predictions")
        
        try:
            # Import the incremental forecast module
            print("\n📦 Importing incremental forecast module...")
            from gnn_predictor import run_incremental_rolling_forecast
            
            print(f"\n⚙️  Configuration:")
            print(f"  Training file: {train_file}")
            print(f"  Test file: {test_file}")
            print(f"  Forecast horizon: {forecast_config['forecast_weeks']} weeks ahead")
            print(f"  Target crop: {forecast_config['target_crop'] or 'All crops'}")
            print(f"  Semantic filtering: {forecast_config['use_semantic_filtering']}")
            print(f"  Architecture: {'GAT (attention)' if forecast_config['use_attention'] else 'GCN'}")
            print(f"  Epochs per retrain: {forecast_config['epochs_per_retrain']}")
            
            # # Run incremental forecast
            # print("\n🚀 Starting incremental rolling forecast...")
            # gnn_predictor, gnn_results, gnn_metrics = run_incremental_rolling_forecast(
            #     kg,
            #     train_csv=train_file,
            #     test_csv=test_file,
            #     forecast_weeks=forecast_config['forecast_weeks'],
            #     target_crop=forecast_config['target_crop'],
            #     use_semantic_filtering=forecast_config['use_semantic_filtering'],
            #     use_attention=forecast_config['use_attention'],
            #     epochs_per_retrain=forecast_config['epochs_per_retrain']
            # )

            # Run single forecast
            print("\n🚀 Starting single  forecast...")
            gnn_predictor, gnn_results, gnn_metrics = run_single_date_forecast(
                kg,
                train_csv='AngusTrain.csv',
                test_csv='AngusTest.csv',
                cutoff_date='2023-05-01',  # Only use data before this date
                only_predict_fields_with_actuals=True,
                forecast_weeks=4,          
                epochs=200
            ) 


            
            print("\n" + "="*70)
            print("✓ INCREMENTAL ROLLING FORECAST COMPLETE!")
            print("="*70)
            
            # Display results
            if gnn_metrics['r2'] is not None:
                print(f"\n📊 Overall Performance (on predictions with actual values):")
                print(f"  {'Metric':<20} {'Value':>10}")
                print(f"  {'-'*20} {'-'*10}")
                print(f"  {'R² Score':<20} {gnn_metrics['r2']:>10.4f}")
                print(f"  {'RMSE':<20} {gnn_metrics['rmse']:>10.4f}")
                print(f"  {'MAE':<20} {gnn_metrics['mae']:>10.4f}")
            else:
                print(f"\n⚠️  No actual values available for evaluation")
            
            print(f"\n📈 Prediction Summary:")
            print(f"  Total predictions made: {gnn_metrics['n_predictions']}")
            print(f"  Predictions with actuals: {gnn_metrics['n_with_actuals']}")
            if gnn_metrics['n_predictions'] > 0:
                coverage = (gnn_metrics['n_with_actuals'] / gnn_metrics['n_predictions']) * 100
                print(f"  Coverage: {coverage:.1f}%")
            
            print(f"\n💾 Output Files:")
            print(f"  📄 Results CSV: outputs/gnn/incremental_rolling_forecast_results.csv")
            print(f"  📊 Plots: outputs/gnn/plots/")
            print(f"     ├─ incremental_forecast_performance.png")
            print(f"     └─ predictions_over_time.png")
            
            # Show sample predictions
            if len(gnn_results) > 0:
                print(f"\n📋 Sample Predictions (first 10):")
                print("-" * 70)
                sample_cols = ['prediction_date', 'target_date', 'field_code', 
                              'predicted', 'actual', 'weeks_ahead']
                available_cols = [col for col in sample_cols if col in gnn_results.columns]
                print(gnn_results[available_cols].head(10).to_string(index=False))
                
                # Show statistics by field
                if 'field_code' in gnn_results.columns and len(gnn_results) > 0:
                    print(f"\n📍 Predictions by Field:")
                    print("-" * 70)
                    field_stats = gnn_results.groupby('field_code').agg({
                        'predicted': ['count', 'mean'],
                        'actual': lambda x: x.notna().sum()
                    }).round(2)
                    field_stats.columns = ['N_Predictions', 'Avg_Predicted', 'N_Actuals']
                    print(field_stats.head(10).to_string())
            
        except ImportError as e:
            print("\n❌ ERROR: Incremental GNN module not found!")
            print(f"   {e}")
            print("\n📝 To fix this:")
            print("   1. Save the incremental_rolling_forecast_gnn.py file")
            print("   2. Make sure it's in the same directory as this script")
            print("   3. Re-run this script")
        except FileNotFoundError as e:
            print(f"\n❌ ERROR: Data file not found!")
            print(f"   {e}")
            print("\n📝 To fix this:")
            print(f"   1. Make sure {train_file} exists")
            print(f"   2. Make sure {test_file} exists")
            print("   3. Check file paths are correct")
        except Exception as e:
            print(f"\n❌ ERROR: Incremental forecast failed!")
            print(f"   {e}")
            print("\n📝 Details:")
            import traceback
            traceback.print_exc()
            print("\nNote: Knowledge graph was built successfully,")
            print("      but the incremental forecast encountered an error.")
    # ======================================================================
    
    print("\n" + "="*70)
    print("🎉 COMPLETE!")
    print("="*70)
    
    print("\n✅ Key Features Applied:")
    print("  ✓ Fixed semantic classification (improved scoring)")
    print("  ✓ Spearman correlation for skewed data")
    print("  ✓ Non-zero target analysis")
    print("  ✓ Lower correlation thresholds (0.05-0.15)")
    print("  ✓ Better handling of shifted/lagged variables")
    if run_incremental_forecast:
        print("  ✓ Incremental rolling forecast (week-by-week)")
        print("  ✓ Real-world deployment simulation")
    
    print("\n📂 Generated Files:")
    print("  Knowledge Graph:")
    print("    ├─ outputs/variable_registry.json")
    print("    ├─ outputs/variable_relationships.json")
    print("    ├─ outputs/pattern_analysis.json")
    print("    ├─ outputs/comprehensive_report.txt")
    print("    ├─ outputs/graphs/semantic_knowledge_graph.pkl")
    print("    └─ outputs/exports/kg_sample.json")
    
    if create_visualizations:
        print("  Visualizations:")
        print("    ├─ outputs/visualizations/semantic_kg_2d.png")
        print("    └─ outputs/visualizations/semantic_kg_3d.html")
    
    if run_incremental_forecast:
        print("  Incremental Forecast:")
        print("    ├─ outputs/gnn/incremental_rolling_forecast_results.csv")
        print("    └─ outputs/gnn/plots/")
        print("        ├─ incremental_forecast_performance.png")
        print("        └─ predictions_over_time.png")
    
    print("\n" + "="*70)
    
    return kg


if __name__ == "__main__":
    kg = main()