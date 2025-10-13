"""
Knowledge Graph Builder - Standalone Module
Can be called to build/rebuild KG from any dataframe
"""

import pandas as pd
import networkx as nx
from datetime import datetime


class VariableMetadata:
    """Metadata for a variable in the knowledge graph"""
    def __init__(self, name, semantic_category='Unknown', data_type='numeric',
                 unit=None, description=None, predictive_power=0.0):
        self.name = name
        self.semantic_category = semantic_category
        self.data_type = data_type
        self.unit = unit
        self.description = description
        self.predictive_power = predictive_power


class SimpleKnowledgeGraph:
    """Lightweight knowledge graph for time series data"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.variable_registry = {}
        self.pattern_analysis_results = {
            'predictive_vars': [],
            'temporal_patterns': [],
            'feature_importance': {}
        }
    
    def build_from_dataframe(self, df, target_column='target', 
                            field_column='lookupEncoded', 
                            date_column='date'):
        """
        Build knowledge graph from dataframe
        
        Args:
            df: DataFrame with time series data
            target_column: Name of target variable
            field_column: Name of field/plot identifier column
            date_column: Name of date column
        """
        print(f"\n{'='*60}")
        print("BUILDING KNOWLEDGE GRAPH FROM DATAFRAME")
        print(f"{'='*60}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        
        # Ensure date column is datetime
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found")
        
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column])
        
        # Register variables
        self._register_variables(df, target_column)
        
        # Build graph nodes and edges
        self._build_graph_structure(df, target_column, field_column, date_column)
        
        print(f"\n✓ Knowledge graph built:")
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {self.graph.number_of_edges()}")
        print(f"  Variables: {len(self.variable_registry)}")
        
        return self
    
    def _register_variables(self, df, target_column):
        """Register all variables with metadata"""
        print(f"\nRegistering variables...")
        
        # Categorize variables
        for col in df.columns:
            if col == target_column:
                category = 'Yield'
            elif 'temp' in col.lower() or 'temperature' in col.lower():
                category = 'Temperature'
            elif 'moisture' in col.lower() or 'humid' in col.lower():
                category = 'Moisture'
            elif 'precip' in col.lower() or 'rain' in col.lower():
                category = 'Precipitation'
            elif 'ndvi' in col.lower() or 'evi' in col.lower():
                category = 'Vegetation_Index'
            elif 'age' in col.lower() or 'growth' in col.lower():
                category = 'Growth_Metric'
            elif 'encoded' in col.lower():
                category = 'Encoded_Variable'
            elif col in ['year', 'month', 'day', 'dayofmonth', 'dayofweek', 'dayofyear']:
                category = 'Temporal'
            else:
                category = 'Other'
            
            self.variable_registry[col] = VariableMetadata(
                name=col,
                semantic_category=category,
                data_type='numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical'
            )
        
        print(f"  Registered {len(self.variable_registry)} variables")
    
    def _build_graph_structure(self, df, target_column, field_column, date_column):
        """Build graph nodes and edges"""
        print(f"\nBuilding graph structure...")
        
        # Group by field and date
        for idx, row in df.iterrows():
            field_id = f"Field_Plot_{int(row[field_column])}" if field_column in row else "unknown"
            date_val = row[date_column]
            
            # Create timestamp node
            ts_node_id = f"{field_id}_{date_val.strftime('%Y%m%d')}"
            
            self.graph.add_node(
                ts_node_id,
                type='Timestamp',
                datetime=date_val,
                field=field_id,
                year=date_val.year,
                month=date_val.month,
                day=date_val.day,
                dayofweek=date_val.dayofweek,
                dayofyear=date_val.dayofyear
            )
            
            # Add measurement nodes
            for col in df.columns:
                if col in [date_column, field_column]:
                    continue
                
                if pd.notna(row[col]):
                    measurement_node = f"{ts_node_id}_{col}"
                    
                    self.graph.add_node(
                        measurement_node,
                        type='Measurement',
                        metric=col,
                        value=row[col]
                    )
                    
                    # Connect timestamp to measurement
                    self.graph.add_edge(ts_node_id, measurement_node)
            
            if idx % 1000 == 0 and idx > 0:
                print(f"  Processed {idx}/{len(df)} rows...")
        
        print(f"  ✓ Graph structure complete")


def build_knowledge_graph_from_dataframe(df, target_column='target',
                                        field_column='lookupEncoded',
                                        date_column='date'):
    """
    Convenience function to build a knowledge graph from a dataframe
    
    Args:
        df: DataFrame with time series data
        target_column: Name of target variable (default: 'target')
        field_column: Name of field identifier (default: 'lookupEncoded')
        date_column: Name of date column (default: 'date')
    
    Returns:
        SimpleKnowledgeGraph object
    """
    kg = SimpleKnowledgeGraph()
    kg.build_from_dataframe(df, target_column, field_column, date_column)
    return kg


if __name__ == "__main__":
    print("="*70)
    print("KNOWLEDGE GRAPH BUILDER - STANDALONE MODULE")
    print("="*70)
    print("\nThis module provides standalone KG building functionality.")
    print()
    print("Usage:")
    print()
    print("  from knowledge_graph_builder import build_knowledge_graph_from_dataframe")
    print()
    print("  # Build KG from dataframe")
    print("  kg = build_knowledge_graph_from_dataframe(")
    print("      df,")
    print("      target_column='target',")
    print("      field_column='lookupEncoded',")
    print("      date_column='date'")
    print("  )")
    print()
    print("  # Use in forecasting")
    print("  predictor, predictions, metrics = run_single_date_forecast(")
    print("      kg,")
    print("      cutoff_date='2023-05-01'")
    print("  )")
    print()
    print("="*70)