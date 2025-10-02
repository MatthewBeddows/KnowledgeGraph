"""
ALL-IN-ONE KNOWLEDGE GRAPH BUILDER AND VIEWER
Builds knowledge graph from timeseries and creates interactive 3D visualization
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import json
import os

class TimeSeriesKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        
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
    
    def create_graph(self, df, field_name=None):
        """Convert timeseries dataframe to knowledge graph"""
        
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
            
            self.graph.add_node(field_id, 
                               type="Field",
                               name=field_name,
                               lookup_code=int(field_code) if has_multiple_fields else None)
            
            for idx, row in field_df.iterrows():
                timestamp = idx if isinstance(idx, (pd.Timestamp, datetime)) else pd.to_datetime(idx)
                ts_id = f"TS_{field_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
                
                self.graph.add_node(ts_id,
                                   type="Timestamp",
                                   datetime=str(timestamp),
                                   year=timestamp.year,
                                   month=timestamp.month,
                                   day=timestamp.day,
                                   dayofweek=timestamp.dayofweek,
                                   dayofyear=timestamp.dayofyear,
                                   field=field_id)
                
                self.graph.add_edge(field_id, ts_id, relationship="HAS_MEASUREMENT")
                
                for col in field_df.columns:
                    if col in ['Fieldname', 'datetime', 'lookupEncoded', 'FarmEncoded']:
                        continue
                        
                    value = row[col]
                    if pd.notna(value):
                        measure_id = f"Measure_{ts_id}_{col}"
                        self.graph.add_node(measure_id,
                                           type="Measurement",
                                           metric=col,
                                           value=float(value))
                        
                        self.graph.add_edge(ts_id, measure_id, 
                                           relationship="HAS_VALUE",
                                           metric=col)
            
            field_timestamps = sorted([n for n, d in self.graph.nodes(data=True) 
                                      if d.get('type') == 'Timestamp' and d.get('field') == field_id],
                                     key=lambda x: self.graph.nodes[x]['datetime'])
            
            for i in range(len(field_timestamps) - 1):
                self.graph.add_edge(field_timestamps[i], field_timestamps[i+1], 
                                   relationship="NEXT")
        
        self._create_temporal_hierarchy(df)
        
        if 'FarmEncoded' in df.columns and 'lookupEncoded' in df.columns:
            self._create_farm_hierarchy(df)
        
        return self.graph
    
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
    
    def get_statistics(self):
        """Get graph statistics"""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': {}
        }
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        return stats
    
    def generate_debug_report(self, output_file='debug_report.txt'):
        """Generate debug report"""
        with open(output_file, 'w') as f:
            f.write("=== KNOWLEDGE GRAPH DEBUG REPORT ===\n\n")
            
            stats = self.get_statistics()
            f.write("NODE COUNTS:\n")
            for node_type, count in stats['node_types'].items():
                f.write(f"  {node_type}: {count}\n")
            f.write(f"  TOTAL: {stats['total_nodes']}\n")
            f.write(f"  EDGES: {stats['total_edges']}\n\n")
            
            field_nodes = [(n, d) for n, d in self.graph.nodes(data=True) 
                          if d.get('type') == 'Field']
            f.write("FIELD NODES:\n")
            for node, data in field_nodes[:20]:
                f.write(f"  {node}: {data.get('name', 'N/A')}\n")
            if len(field_nodes) > 20:
                f.write(f"  ... and {len(field_nodes) - 20} more\n")
            f.write(f"  Total Fields: {len(field_nodes)}\n\n")
        
        print(f"Debug report saved to: {output_file}")
    
    def save_graph(self, filepath='knowledge_graph.pkl'):
        """Save graph to disk"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({'graph': self.graph, 'stats': self.get_statistics()}, f)
        print(f"Knowledge graph saved to: {filepath}")
    
    def load_graph(self, filepath='knowledge_graph.pkl'):
        """Load graph from disk"""
        import pickle
        with open(filepath, 'rb') as f:
            graph_data = pickle.load(f)
        self.graph = graph_data['graph']
        print(f"Knowledge graph loaded from: {filepath}")
        return self.graph
    
    def export_sample_json(self, output_file='kg_sample.json', num_nodes=100):
        """Export sample as JSON"""
        nodes = list(self.graph.nodes())[:num_nodes]
        subgraph = self.graph.subgraph(nodes)
        
        data = {'nodes': [], 'edges': []}
        for node, attrs in subgraph.nodes(data=True):
            node_data = {'id': node}
            node_data.update(attrs)
            data['nodes'].append(node_data)
        
        for src, dst, attrs in subgraph.edges(data=True):
            edge_data = {'source': src, 'target': dst}
            edge_data.update(attrs)
            data['edges'].append(edge_data)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Sample graph exported to {output_file}")
    
    def export_to_neo4j_cypher(self, output_file='kg_cypher.txt'):
        """Export as Neo4j Cypher"""
        with open(output_file, 'w') as f:
            for node, data in self.graph.nodes(data=True):
                props = ', '.join([f"{k}: '{v}'" if isinstance(v, str) else f"{k}: {v}" 
                                  for k, v in data.items()])
                node_type = data.get('type', 'Node')
                f.write(f"CREATE (n:{node_type} {{id: '{node}', {props}}});\n")
            
            for src, dst, data in self.graph.edges(data=True):
                rel_type = data.get('relationship', 'RELATES_TO')
                f.write(f"MATCH (a {{id: '{src}'}}), (b {{id: '{dst}'}}) "
                       f"CREATE (a)-[:{rel_type}]->(b);\n")
        print(f"Exported Cypher to {output_file}")
    
    def visualize_subgraph(self, num_nodes=50, output_file='kg_visualization.png'):
        """2D visualization"""
        nodes = list(self.graph.nodes())[:num_nodes]
        subgraph = self.graph.subgraph(nodes)
        
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
        
        color_map = {'Field': '#FF6B6B', 'Timestamp': '#4ECDC4', 'Measurement': '#95E1D3',
                    'Year': '#FFA07A', 'Month': '#FFD93D', 'Week': '#A8DADC'}
        node_colors = [color_map.get(self.graph.nodes[n].get('type'), '#CCCCCC') 
                      for n in subgraph.nodes()]
        
        nx.draw(subgraph, pos, node_color=node_colors, node_size=300,
               with_labels=False, arrows=True, edge_color='#999999', width=0.5, alpha=0.7)
        
        plt.title("Knowledge Graph Visualization", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {output_file}")
    
    def create_3d_visualization(self, output_file='kg_3d.html', max_nodes=5000,
                               layout_type='spatial', filter_plots=None, 
                               sample_timestamps=10, report_file=None):
        """Create 3D visualization"""
        
        if filter_plots is not None:
            filtered_nodes = []
            for node, data in self.graph.nodes(data=True):
                if data.get('type') == 'Field' and data.get('lookup_code') in filter_plots:
                    filtered_nodes.append(node)
                elif data.get('type') == 'Timestamp':
                    field_id = data.get('field')
                    if any(f"Plot_{p}" in field_id for p in filter_plots):
                        filtered_nodes.append(node)
                elif data.get('type') == 'Measurement':
                    for pred in self.graph.predecessors(node):
                        if pred in filtered_nodes:
                            filtered_nodes.append(node)
                            break
                else:
                    filtered_nodes.append(node)
            subgraph = self.graph.subgraph(filtered_nodes)
        else:
            sampled_nodes = set()
            for node, data in self.graph.nodes(data=True):
                if data.get('type') in ['Field', 'Farm', 'Year', 'Month', 'Week']:
                    sampled_nodes.add(node)
            
            timestamp_nodes = [n for n, d in self.graph.nodes(data=True) 
                             if d.get('type') == 'Timestamp']
            field_timestamps = {}
            for ts in timestamp_nodes:
                field_id = self.graph.nodes[ts].get('field', 'unknown')
                if field_id not in field_timestamps:
                    field_timestamps[field_id] = []
                field_timestamps[field_id].append(ts)
            
            for field_id, timestamps in field_timestamps.items():
                sorted_ts = sorted(timestamps, key=lambda x: self.graph.nodes[x]['datetime'])
                sampled = sorted_ts[::sample_timestamps]
                sampled_nodes.update(sampled)
                
                for ts in sampled:
                    for neighbor in self.graph.neighbors(ts):
                        if self.graph.nodes[neighbor].get('type') == 'Measurement':
                            sampled_nodes.add(neighbor)
            
            subgraph = self.graph.subgraph(list(sampled_nodes))
        
        if subgraph.number_of_nodes() > max_nodes:
            priority_nodes = set()
            other_nodes = []
            for node, data in subgraph.nodes(data=True):
                if data.get('type') in ['Field', 'Farm']:
                    priority_nodes.add(node)
                else:
                    other_nodes.append(node)
            remaining = max_nodes - len(priority_nodes)
            if remaining > 0:
                priority_nodes.update(other_nodes[:remaining])
            subgraph = self.graph.subgraph(list(priority_nodes))
        
        if layout_type == 'spatial':
            pos = self._spatial_layout(subgraph)
        else:
            pos = nx.spring_layout(subgraph, dim=3, k=2.0, iterations=50, seed=42)
        
        node_x, node_y, node_z, node_text, node_colors, node_sizes = [], [], [], [], [], []
        color_map = {'Field': '#FF6B6B', 'Farm': '#E74C3C', 'Timestamp': '#4ECDC4',
                    'Measurement': '#95E1D3', 'Year': '#FFA07A', 'Month': '#FFD93D', 'Week': '#A8DADC'}
        
        for node in subgraph.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            
            node_data = self.graph.nodes[node]
            node_type = node_data.get('type', 'Unknown')
            node_text.append(f"<b>{node}</b><br>Type: {node_type}")
            node_colors.append(color_map.get(node_type, '#CCCCCC'))
            
            base_size = {'Field': 15, 'Farm': 20, 'Timestamp': 8, 
                        'Measurement': 5}.get(node_type, 6)
            node_sizes.append(min(base_size + subgraph.degree(node) * 0.5, 25))
        
        edge_x, edge_y, edge_z = [], [], []
        for edge in subgraph.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
        
        fig = go.Figure(data=[
            go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines',
                        line=dict(color='rgba(125,125,125,0.2)', width=1), hoverinfo='none'),
            go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers',
                        marker=dict(size=node_sizes, color=node_colors, opacity=0.9),
                        text=node_text, hoverinfo='text')
        ])
        
        fig.update_layout(
            title='Interactive 3D Knowledge Graph',
            showlegend=False,
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False),
                yaxis=dict(showbackground=False, showticklabels=False),
                zaxis=dict(showbackground=False, showticklabels=False),
                camera=dict(eye=dict(x=2.5, y=2.5, z=1.5))
            ),
            height=900
        )
        
        fig.write_html(output_file)
        print(f"3D visualization saved to: {output_file}")
    
    def create_future_prediction_nodes(self, year=2024, plots=None):
        """Create blank structure for future year prediction
        
        WARNING: This creates nodes without actual data. Predictions will be
        based only on temporal patterns and graph structure, not real conditions.
        """
        
        print(f"\nCreating blank prediction structure for {year}...")
        
        # Get existing plots if not specified
        if plots is None:
            field_nodes = [n for n, d in self.graph.nodes(data=True) 
                          if d.get('type') == 'Field']
            plots = [d.get('lookup_code') for n, d in self.graph.nodes(data=True) 
                    if d.get('type') == 'Field' and d.get('lookup_code') is not None]
        
        print(f"  Creating nodes for {len(plots)} plots")
        
        # Create weekly timestamps for the year
        dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='W')
        print(f"  Creating {len(dates)} weekly timestamps")
        
        nodes_created = 0
        for plot_code in plots:
            field_id = f"Field_Plot_{int(plot_code)}"
            
            if field_id not in self.graph:
                print(f"  Warning: {field_id} not in graph, skipping")
                continue
            
            for date in dates:
                ts_id = f"TS_{field_id}_{date.strftime('%Y%m%d_%H%M%S')}"
                
                # Create timestamp node (no target/yield value)
                self.graph.add_node(ts_id,
                                   type="Timestamp",
                                   datetime=str(date),
                                   year=date.year,
                                   month=date.month,
                                   day=date.day,
                                   dayofweek=date.dayofweek,
                                   dayofyear=date.dayofyear,
                                   field=field_id,
                                   is_future=True)  # Mark as future prediction
                
                self.graph.add_edge(field_id, ts_id, relationship="HAS_MEASUREMENT")
                nodes_created += 1
        
        print(f"  ✓ Created {nodes_created} future timestamp nodes")
        print(f"\n  NOTE: These nodes have NO actual yield data")
        print(f"  Predictions will be based on:")
        print(f"    - Temporal patterns (time of year)")
        print(f"    - Graph structure (which plot)")
        print(f"    - Historical patterns from similar periods")
        print(f"  But will NOT account for:")
        print(f"    - Actual {year} weather")
        print(f"    - Real plant conditions")
        print(f"    - Recent yield trends")
        
        return self.graph
        """Spatial layout with fields in circle"""
        pos = {}
        field_nodes = [n for n, d in subgraph.nodes(data=True) if d.get('type') == 'Field']
        
        for i, field in enumerate(field_nodes):
            angle = 2 * np.pi * i / max(len(field_nodes), 1)
            pos[field] = (50 * np.cos(angle), 50 * np.sin(angle), 0)
            
            field_ts = [n for n in subgraph.neighbors(field) 
                       if subgraph.nodes[n].get('type') == 'Timestamp']
            for j, ts in enumerate(sorted(field_ts, key=lambda x: subgraph.nodes[x]['datetime'])):
                pos[ts] = (pos[field][0] + 5*np.cos(j), pos[field][1] + 5*np.sin(j), j*1.5)
        
        for node in subgraph.nodes():
            if node not in pos:
                pos[node] = (0, 0, 0)
        
        return pos


def main():
    print("="*60)
    print("TIMESERIES KNOWLEDGE GRAPH")
    print("="*60)
    
    # CONFIG
    input_file = 'full_dataset.csv'
    build_graph = True
    create_visualizations = False
    run_gnn = True
    test_year = 2023
    forecast_weeks = 4  # NEW: Predict N weeks ahead
    
    # Create output directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/graphs', exist_ok=True)
    os.makedirs('outputs/visualizations', exist_ok=True)
    os.makedirs('outputs/exports', exist_ok=True)
    os.makedirs('outputs/gnn', exist_ok=True)
    
    kg = TimeSeriesKnowledgeGraph()
    
    if build_graph:
        print("\n[1/4] Loading data...")
        df_full = kg.load_data(input_file)
        print(f"Loaded {len(df_full)} rows")
        print(f"Date range: {df_full.index.min()} to {df_full.index.max()}")
        print(f"Years in data: {sorted(df_full.index.year.unique())}")
        
        # NEW: Build graph with ALL data (need test year features for rolling forecast)
        print(f"\n[2/4] Building graph with ALL data...")
        kg.create_graph(df_full)
        
        print(f"\nGraph includes years: {sorted(df_full.index.year.unique())}")
        print(f"Test year {test_year} data IS in the graph (we need its features)")
        
        print("\n[3/4] Exporting...")
        kg.generate_debug_report('outputs/debug_report.txt')
        kg.export_sample_json('outputs/exports/kg_sample.json')
        kg.export_to_neo4j_cypher('outputs/exports/kg_cypher.txt')
        kg.visualize_subgraph(num_nodes=100, output_file='outputs/visualizations/kg_2d.png')
        kg.save_graph('outputs/graphs/knowledge_graph.pkl')
    else:
        print("\nLoading existing graph...")
        kg.load_graph('outputs/graphs/knowledge_graph.pkl')
    
    if create_visualizations:
        print("\n[4/4] Creating 3D visualizations...")
        kg.create_3d_visualization('outputs/visualizations/kg_3d_overview.html', 
                                   sample_timestamps=100)
    
    if run_gnn:
        try:
            from gnn_predictor import run_rolling_forecast
            print("\n" + "="*60)
            print("RUNNING ROLLING FORECAST GNN")
            print("="*60)
            
            predictor, results, metrics = run_rolling_forecast(
                kg,
                forecast_weeks=forecast_weeks,
                test_year=test_year
            )
            
            print("\n" + "="*60)
            print("SUCCESS!")
            print("="*60)
            print(f"\nTest Set Results:")
            print(f"  RMSE: {metrics['test']['rmse']:.4f}")
            print(f"  MAE:  {metrics['test']['mae']:.4f}")
            print(f"  R²:   {metrics['test']['r2']:.4f}")
            
        except ImportError as e:
            print("\nMissing dependency:")
            print("  pip install torch torch-geometric")
        except Exception as e:
            print(f"\nGNN failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nDONE!")
    return kg

if __name__ == "__main__":
    main()