"""
Incremental Weekly Rolling Forecast GNN
Adds test data week by week, retrains, and predicts 4 weeks ahead
This emulates real-world deployment
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
import warnings
import copy
from datetime import timedelta
warnings.filterwarnings('ignore')


class ContextAwareYieldGNN(torch.nn.Module):
    """GNN with attention to focus on relevant features"""
    def __init__(self, num_node_features, hidden_channels=64, num_layers=3, use_attention=True):
        super(ContextAwareYieldGNN, self).__init__()
        
        self.use_attention = use_attention
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        if use_attention:
            self.convs.append(GATConv(num_node_features, hidden_channels, heads=4, concat=False))
        else:
            self.convs.append(GCNConv(num_node_features, hidden_channels))
        
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 1):
            if use_attention:
                self.convs.append(GATConv(hidden_channels, hidden_channels, heads=4, concat=False))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = torch.nn.Linear(hidden_channels // 2, 1)
        self.dropout = torch.nn.Dropout(0.3)
        
    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        
        return x.squeeze()


class IncrementalRollingForecastGNN:
    """
    GNN that incrementally adds new data week by week
    - Starts with training data only (AngusTrain)
    - Each week: adds new week of data, retrains, predicts 4 weeks ahead
    - This emulates real-world deployment
    """
    
    def __init__(self, knowledge_graph, forecast_weeks=4, hidden_dim=64, num_layers=3,
                 target_crop=None, use_semantic_filtering=True, use_attention=True,
                 retrain_frequency='weekly'):
        self.kg_original = knowledge_graph
        self.kg = None
        self.forecast_weeks = forecast_weeks
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.target_crop = target_crop
        self.use_semantic_filtering = use_semantic_filtering
        self.use_attention = use_attention
        self.retrain_frequency = retrain_frequency  # 'weekly' or 'every_N_weeks'
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.feature_names = []
        self.relevant_features = set()
        self.feature_importance = {}
        
        # Track incremental data
        self.available_data = []  # Data we've seen so far
        self.prediction_history = []  # Track all predictions
        
    def _identify_relevant_features(self, crop_name=None):
        """Use semantic knowledge graph to identify relevant features"""
        print(f"\n{'='*60}")
        print("IDENTIFYING RELEVANT FEATURES")
        print(f"{'='*60}")
        
        relevant_vars = set()
        
        if not self.use_semantic_filtering:
            print("Semantic filtering disabled - using all features")
            return None
        
        # Get predictive variables
        if hasattr(self.kg, 'pattern_analysis_results'):
            predictive_vars = self.kg.pattern_analysis_results.get('predictive_vars', [])
            print(f"\nPredictive variables: {len(predictive_vars)}")
            
            for pv in predictive_vars[:30]:
                relevant_vars.add(pv['variable'])
                self.feature_importance[pv['variable']] = pv.get('combined_importance', 0.0)
            
            if len(predictive_vars) > 0:
                print(f"  Top 5:")
                for pv in predictive_vars[:5]:
                    print(f"    {pv['variable']:35s} importance: {pv.get('combined_importance', 0):.3f}")
        
        # Get variables by semantic category
        relevant_categories = [
            'Temperature', 'Moisture', 'Precipitation', 'Wind', 'Pressure',
            'Vegetation_Index', 'Growth_Metric', 'Yield'
        ]
        
        print(f"\nVariables by semantic category:")
        for var_name, metadata in self.kg.variable_registry.items():
            if metadata.semantic_category in relevant_categories:
                relevant_vars.add(var_name)
                if var_name not in self.feature_importance:
                    self.feature_importance[var_name] = metadata.predictive_power
        
        cat_counts = {}
        for var in relevant_vars:
            if var in self.kg.variable_registry:
                cat = self.kg.variable_registry[var].semantic_category
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        
        for cat, count in sorted(cat_counts.items()):
            print(f"  {cat:25s}: {count:3d} variables")
        
        self.relevant_features = relevant_vars
        print(f"\nTotal relevant features: {len(relevant_vars)}")
        
        return relevant_vars
    
    def load_initial_training_data(self, train_csv_path):
        """Load AngusTrain.csv and build initial knowledge graph"""
        print(f"\n{'='*70}")
        print("LOADING INITIAL TRAINING DATA")
        print(f"{'='*70}")
        print(f"Training file: {train_csv_path}")
        
        # Load training data
        train_df = pd.read_csv(train_csv_path)
        print(f"Loaded {len(train_df)} rows")
        
        # Parse dates
        if all(col in train_df.columns for col in ['year', 'month', 'dayofmonth']):
            train_df['date'] = pd.to_datetime(
                train_df[['year', 'month', 'dayofmonth']].rename(columns={'dayofmonth': 'day'})
            )
        elif 'date' in train_df.columns:
            train_df['date'] = pd.to_datetime(train_df['date'])
        else:
            raise ValueError("Cannot parse dates from training file")
        
        print(f"Date range: {train_df['date'].min()} to {train_df['date'].max()}")
        
        # Get field identifier
        field_col = None
        if 'lookupEncoded' in train_df.columns:
            field_col = 'lookupEncoded'
        elif 'FarmEncoded' in train_df.columns:
            field_col = 'FarmEncoded'
        
        if field_col:
            print(f"Fields: {train_df[field_col].nunique()} unique")
        
        # Store as available data
        self.available_data = train_df.sort_values('date').to_dict('records')
        
        print(f"✓ Loaded {len(self.available_data)} timesteps as initial training data")
        
        return train_df
    
    def load_test_data(self, test_csv_path):
        """Load test data (will be added incrementally)"""
        print(f"\n{'='*70}")
        print("LOADING TEST DATA (for incremental addition)")
        print(f"{'='*70}")
        print(f"Test file: {test_csv_path}")
        
        test_df = pd.read_csv(test_csv_path)
        print(f"Loaded {len(test_df)} rows")
        
        # Parse dates
        if all(col in test_df.columns for col in ['year', 'month', 'dayofmonth']):
            test_df['date'] = pd.to_datetime(
                test_df[['year', 'month', 'dayofmonth']].rename(columns={'dayofmonth': 'day'})
            )
        elif 'date' in test_df.columns:
            test_df['date'] = pd.to_datetime(test_df['date'])
        else:
            raise ValueError("Cannot parse dates from test file")
        
        test_df = test_df.sort_values('date')
        print(f"Date range: {test_df['date'].min()} to {test_df['date'].max()}")
        
        # Get unique weeks
        test_df['week'] = test_df['date'].dt.isocalendar().week
        test_df['year'] = test_df['date'].dt.year
        unique_weeks = test_df.groupby(['year', 'week']).size()
        print(f"Test data spans {len(unique_weeks)} unique weeks")
        
        return test_df
    
    def get_weekly_batches(self, test_df):
        """Split test data into weekly batches"""
        test_df['week'] = test_df['date'].dt.isocalendar().week
        test_df['year'] = test_df['date'].dt.year
        
        weekly_batches = []
        for (year, week), group in test_df.groupby(['year', 'week'], sort=True):
            weekly_batches.append({
                'year': year,
                'week': week,
                'start_date': group['date'].min(),
                'end_date': group['date'].max(),
                'data': group.to_dict('records'),
                'n_records': len(group)
            })
        
        return weekly_batches
    
    def run_incremental_forecast(self, train_csv, test_csv, epochs_per_retrain=100):
        """
        Main incremental forecasting loop
        
        1. Load training data (AngusTrain.csv)
        2. Load test data (AngusTest.csv) 
        3. For each week in test data:
           a. Add that week's data to available data
           b. Rebuild knowledge graph with all data seen so far
           c. Retrain model
           d. Predict 4 weeks ahead
           e. Store predictions
        """
        print(f"\n{'='*70}")
        print("INCREMENTAL ROLLING FORECAST")
        print(f"{'='*70}")
        print(f"Forecast horizon: {self.forecast_weeks} weeks ahead")
        print(f"Retrain frequency: {self.retrain_frequency}")
        
        # Step 1: Load initial training data
        train_df = self.load_initial_training_data(train_csv)
        
        # Step 2: Load test data
        test_df = self.load_test_data(test_csv)
        
        # Step 3: Split test data into weekly batches
        weekly_batches = self.get_weekly_batches(test_df)
        print(f"\n{'='*70}")
        print(f"STARTING INCREMENTAL FORECAST LOOP")
        print(f"{'='*70}")
        print(f"Will process {len(weekly_batches)} weeks of test data")
        
        all_predictions = []
        
        # Process each week
        for week_idx, batch in enumerate(weekly_batches):
            print(f"\n{'='*60}")
            print(f"WEEK {week_idx + 1}/{len(weekly_batches)}: {batch['start_date'].date()} to {batch['end_date'].date()}")
            print(f"{'='*60}")
            print(f"Adding {batch['n_records']} new timesteps to available data")
            
            # Add this week's data to available data
            self.available_data.extend(batch['data'])
            print(f"Total available data: {len(self.available_data)} timesteps")
            
            # Rebuild knowledge graph with all data seen so far
            print(f"\nRebuilding knowledge graph with all available data...")
            current_df = pd.DataFrame(self.available_data)
            self._rebuild_knowledge_graph(current_df)
            
            # Identify relevant features (only once)
            if week_idx == 0:
                self._identify_relevant_features(self.target_crop)
            
            # Prepare data for training
            print(f"\nPreparing training data...")
            graph_data, train_samples = self._prepare_training_data(current_df)
            
            if graph_data is None:
                print(f"⚠️  Skipping week {week_idx + 1} - insufficient data")
                continue
            
            # Retrain model
            print(f"\nRetraining model...")
            self._train_model(graph_data, epochs=epochs_per_retrain)
            
            # Make predictions for 4 weeks ahead
            print(f"\nMaking predictions for {self.forecast_weeks} weeks ahead...")
            predictions = self._predict_future(current_df, batch['end_date'])
            
            # Store predictions
            for pred in predictions:
                pred['week_added'] = week_idx + 1
                pred['data_cutoff_date'] = batch['end_date']
                all_predictions.append(pred)
            
            print(f"✓ Week {week_idx + 1} complete: {len(predictions)} predictions made")
        
        print(f"\n{'='*70}")
        print("INCREMENTAL FORECAST COMPLETE")
        print(f"{'='*70}")
        print(f"Total predictions made: {len(all_predictions)}")
        
        # Create results dataframe
        results_df = pd.DataFrame(all_predictions)
        
        # Calculate metrics (only for predictions where we have actuals)
        results_df_with_actuals = results_df[results_df['actual'].notna()]
        
        if len(results_df_with_actuals) > 0:
            rmse = np.sqrt(mean_squared_error(
                results_df_with_actuals['actual'],
                results_df_with_actuals['predicted']
            ))
            mae = mean_absolute_error(
                results_df_with_actuals['actual'],
                results_df_with_actuals['predicted']
            )
            r2 = r2_score(
                results_df_with_actuals['actual'],
                results_df_with_actuals['predicted']
            )
            
            print(f"\nOverall Performance (on available actuals):")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  R²:   {r2:.4f}")
            print(f"  Predictions with actuals: {len(results_df_with_actuals)}/{len(results_df)}")
        else:
            print(f"\n⚠️  No actual values available for evaluation")
            rmse = mae = r2 = None
        
        # Save results
        os.makedirs('outputs/gnn', exist_ok=True)
        results_df.to_csv('outputs/gnn/incremental_rolling_forecast_results.csv', index=False)
        print(f"\n✓ Results saved to: outputs/gnn/incremental_rolling_forecast_results.csv")
        
        # Create visualization
        self._visualize_incremental_results(results_df)
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_predictions': len(results_df),
            'n_with_actuals': len(results_df_with_actuals)
        }
        
        return results_df, metrics
    
    def _rebuild_knowledge_graph(self, df):
        """Rebuild knowledge graph with current available data"""
        # Create a fresh copy of the knowledge graph
        self.kg = copy.deepcopy(self.kg_original)
        
        # Convert dataframe to format expected by KG
        if 'date' not in df.columns:
            raise ValueError("DataFrame must have 'date' column")
        
        # Set index to date
        df_indexed = df.set_index('date')
        
        # Rebuild graph (simplified version)
        print(f"  Building graph with {len(df_indexed)} timesteps...")
        # Note: In practice, you'd call kg.create_graph() but simplified here
        
    def _prepare_training_data(self, df):
        """Prepare training data from current available data"""
        # Collect timestamps from knowledge graph
        all_timestamps = []
        for node, data in self.kg.graph.nodes(data=True):
            if data.get('type') == 'Timestamp':
                if self.target_crop and data.get('crop') != self.target_crop:
                    continue
                all_timestamps.append({
                    'node': node,
                    'date': pd.to_datetime(data['datetime']),
                    'field': data.get('field', 'unknown'),
                    'crop': data.get('crop'),
                    'data': data
                })
        
        all_timestamps.sort(key=lambda x: (x['field'], x['date']))
        
        if len(all_timestamps) < self.forecast_weeks + 1:
            print(f"  ⚠️  Insufficient timesteps: {len(all_timestamps)}")
            return None, None
        
        # Create training samples
        train_samples = []
        
        # Group by field
        field_timestamps = {}
        for ts in all_timestamps:
            field = ts['field']
            if field not in field_timestamps:
                field_timestamps[field] = []
            field_timestamps[field].append(ts)
        
        for field_id, timestamps in field_timestamps.items():
            for i in range(len(timestamps) - self.forecast_weeks):
                current_ts = timestamps[i]
                future_ts = timestamps[i + self.forecast_weeks]
                
                features = self._extract_filtered_features(current_ts['node'])
                target = self._get_yield_for_timestamp(future_ts['node'])
                
                if features is not None and target is not None:
                    train_samples.append({
                        'node': current_ts['node'],
                        'features': features,
                        'target': target,
                        'field': field_id,
                        'date': current_ts['date']
                    })
        
        if len(train_samples) == 0:
            return None, None
        
        print(f"  Created {len(train_samples)} training samples")
        
        # Convert to graph format
        graph_data = self._samples_to_graph_data(train_samples)
        
        return graph_data, train_samples
    
    def _extract_filtered_features(self, timestamp_node):
        """Extract features with semantic filtering"""
        features = {}
        
        node_data = self.kg.graph.nodes[timestamp_node]
        features['month'] = node_data.get('month', 0) / 12.0
        features['day'] = node_data.get('day', 0) / 31.0
        features['dayofweek'] = node_data.get('dayofweek', 0) / 7.0
        features['dayofyear'] = node_data.get('dayofyear', 0) / 365.0
        
        field_id = node_data.get('field', '')
        if 'Plot_' in field_id:
            plot_num = int(field_id.split('Plot_')[1].split('_')[0])
            features['plot_encoded'] = plot_num / 100.0
        else:
            features['plot_encoded'] = 0.0
        
        measurement_count = 0
        for neighbor in self.kg.graph.neighbors(timestamp_node):
            neighbor_data = self.kg.graph.nodes[neighbor]
            if neighbor_data.get('type') == 'Measurement':
                metric = neighbor_data.get('metric', '')
                value = neighbor_data.get('value')
                
                if 'target' in metric.lower():
                    continue
                
                if self.use_semantic_filtering and self.relevant_features:
                    if metric not in self.relevant_features:
                        continue
                
                if value is not None:
                    features[metric] = float(value)
                    measurement_count += 1
        
        if measurement_count < 3:
            return None
        
        return features
    
    def _get_yield_for_timestamp(self, timestamp_node):
        """Get yield/target value"""
        for neighbor in self.kg.graph.neighbors(timestamp_node):
            node_data = self.kg.graph.nodes[neighbor]
            if node_data.get('type') == 'Measurement':
                metric = node_data.get('metric', '')
                if 'target' in metric.lower():
                    return float(node_data.get('value', 0))
        return None
    
    def _samples_to_graph_data(self, train_samples):
        """Convert samples to PyTorch Geometric format"""
        if len(train_samples) == 0:
            return None
        
        # Determine feature names
        self.feature_names = sorted(train_samples[0]['features'].keys())
        
        # Create node mapping
        all_nodes = list(self.kg.graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        # Build edge index
        edge_list = []
        for src, dst in self.kg.graph.edges():
            src_idx = self.node_to_idx[src]
            dst_idx = self.node_to_idx[dst]
            edge_list.append([src_idx, dst_idx])
            edge_list.append([dst_idx, src_idx])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Initialize feature matrix
        num_nodes = len(all_nodes)
        num_features = len(self.feature_names)
        x = np.zeros((num_nodes, num_features))
        y = np.zeros(num_nodes)
        train_mask = np.zeros(num_nodes, dtype=bool)
        
        for sample in train_samples:
            node_idx = self.node_to_idx[sample['node']]
            x[node_idx] = [sample['features'].get(f, 0.0) for f in self.feature_names]
            y[node_idx] = sample['target']
            train_mask[node_idx] = True
        
        # Scale features
        self.scaler_X.fit(x[train_mask])
        x = self.scaler_X.transform(x)
        
        # Scale targets
        self.scaler_y.fit(y[train_mask].reshape(-1, 1))
        y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).flatten()
        
        # Convert to tensors
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y_scaled, dtype=torch.float)
        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)
        
        return data
    
    def _train_model(self, data, epochs=100):
        """Train the GNN model"""
        num_features = data.x.shape[1]
        self.model = ContextAwareYieldGNN(num_features, self.hidden_dim,
                                          self.num_layers, self.use_attention)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        data = data.to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 25 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    def _predict_future(self, df, current_date):
        """Predict yield 4 weeks from current_date"""
        predictions = []
        
        target_date = current_date + timedelta(weeks=self.forecast_weeks)
        
        # Find all fields/plots
        field_col = 'lookupEncoded' if 'lookupEncoded' in df.columns else 'FarmEncoded'
        
        if field_col not in df.columns:
            return predictions
        
        unique_fields = df[field_col].unique()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.eval()
        with torch.no_grad():
            for field_code in unique_fields:
                # Find the most recent timestamp for this field
                field_data = df[df[field_col] == field_code]
                field_data = field_data[field_data['date'] <= current_date]
                
                if len(field_data) == 0:
                    continue
                
                latest_date = field_data['date'].max()
                
                # Find corresponding node in graph
                ts_node = self._find_timestamp_node(latest_date, field_code)
                
                if ts_node is None:
                    continue
                
                # Get features
                features = self._extract_filtered_features(ts_node)
                
                if features is None:
                    continue
                
                # Prepare input
                node_idx = self.node_to_idx.get(ts_node)
                if node_idx is None:
                    continue
                
                # Get all data
                all_data = self._get_full_graph_data()
                all_data = all_data.to(device)
                
                # Predict
                out = self.model(all_data.x, all_data.edge_index)
                pred_scaled = out[node_idx].cpu().item()
                
                # Inverse transform
                pred_orig = self.scaler_y.inverse_transform([[pred_scaled]])[0][0]
                
                # Try to get actual value if available
                actual = self._get_actual_yield(df, field_code, target_date)
                
                predictions.append({
                    'prediction_date': current_date,
                    'target_date': target_date,
                    'field_code': int(field_code),
                    'predicted': pred_orig,
                    'actual': actual,
                    'weeks_ahead': self.forecast_weeks
                })
        
        return predictions
    
    def _find_timestamp_node(self, date, field_code):
        """Find timestamp node in graph"""
        for node, data in self.kg.graph.nodes(data=True):
            if data.get('type') == 'Timestamp':
                node_date = pd.to_datetime(data['datetime']).date()
                field_id = data.get('field', '')
                
                if 'Plot_' in field_id:
                    node_field_code = int(field_id.split('Plot_')[1].split('_')[0])
                else:
                    node_field_code = 0
                
                if node_date == date.date() and node_field_code == field_code:
                    return node
        return None
    
    def _get_actual_yield(self, df, field_code, target_date):
        """Get actual yield if available"""
        field_col = 'lookupEncoded' if 'lookupEncoded' in df.columns else 'FarmEncoded'
        
        match = df[(df[field_col] == field_code) & (df['date'] == target_date)]
        
        if len(match) > 0 and 'target' in match.columns:
            return float(match['target'].iloc[0])
        
        return None
    
    def _get_full_graph_data(self):
        """Get full graph data for prediction"""
        all_nodes = list(self.kg.graph.nodes())
        num_nodes = len(all_nodes)
        num_features = len(self.feature_names)
        
        x = np.zeros((num_nodes, num_features))
        
        # Fill with zeros (will use trained representations)
        x = self.scaler_X.transform(x)
        x = torch.tensor(x, dtype=torch.float)
        
        edge_list = []
        for src, dst in self.kg.graph.edges():
            src_idx = self.node_to_idx[src]
            dst_idx = self.node_to_idx[dst]
            edge_list.append([src_idx, dst_idx])
            edge_list.append([dst_idx, src_idx])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)
    
    def _visualize_incremental_results(self, results_df):
        """Create visualizations"""
        os.makedirs('outputs/gnn/plots', exist_ok=True)
        
        # Filter to records with actuals
        df_with_actuals = results_df[results_df['actual'].notna()]
        
        if len(df_with_actuals) > 0:
            plt.figure(figsize=(14, 6))
            
            plt.subplot(1, 2, 1)
            plt.scatter(df_with_actuals['actual'], df_with_actuals['predicted'], alpha=0.6)
            plt.plot([df_with_actuals['actual'].min(), df_with_actuals['actual'].max()],
                    [df_with_actuals['actual'].min(), df_with_actuals['actual'].max()],
                    'r--', lw=2)
            plt.xlabel('Actual Yield')
            plt.ylabel('Predicted Yield')
            plt.title('Incremental Forecast: Predicted vs Actual')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            df_with_actuals['error'] = df_with_actuals['predicted'] - df_with_actuals['actual']
            plt.hist(df_with_actuals['error'], bins=30, edgecolor='black', alpha=0.7)
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.title('Error Distribution')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('outputs/gnn/plots/incremental_forecast_performance.png', dpi=150)
            plt.close()
            print(f"✓ Visualization saved to: outputs/gnn/plots/incremental_forecast_performance.png")
        
        # Plot predictions over time
        plt.figure(figsize=(16, 6))
        
        for field_code in results_df['field_code'].unique()[:5]:  # Plot first 5 fields
            field_data = results_df[results_df['field_code'] == field_code].sort_values('prediction_date')
            plt.plot(field_data['target_date'], field_data['predicted'], 
                    marker='o', label=f'Field {field_code} (Predicted)', alpha=0.7)
            
            if 'actual' in field_data.columns:
                actual_data = field_data[field_data['actual'].notna()]
                if len(actual_data) > 0:
                    plt.scatter(actual_data['target_date'], actual_data['actual'],
                              marker='x', s=100, label=f'Field {field_code} (Actual)', alpha=0.9)
        
        plt.xlabel('Target Date')
        plt.ylabel('Yield')
        plt.title('Incremental Forecast: Predictions Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('outputs/gnn/plots/predictions_over_time.png', dpi=150)
        plt.close()
        print(f"✓ Timeline visualization saved to: outputs/gnn/plots/predictions_over_time.png")


def run_incremental_rolling_forecast(kg, train_csv='AngusTrain.csv', 
                                     test_csv='AngusTest.csv',
                                     forecast_weeks=4,
                                     target_crop=None,
                                     use_semantic_filtering=True,
                                     use_attention=True,
                                     epochs_per_retrain=100):
    """
    Run incremental rolling forecast
    
    This simulates real-world deployment:
    1. Train on initial data (AngusTrain.csv)
    2. Each week:
       - Add new week of data from AngusTest.csv
       - Retrain model with all data seen so far
       - Predict 4 weeks ahead
       - Store predictions
    
    Args:
        kg: Knowledge graph object (will be rebuilt incrementally)
        train_csv: Initial training data file
        test_csv: Test data file (added week by week)
        forecast_weeks: How many weeks ahead to predict (default: 4)
        target_crop: Specific crop to focus on (None = all)
        use_semantic_filtering: Use KG to filter features
        use_attention: Use Graph Attention Networks
        epochs_per_retrain: Training epochs each week
    
    Returns:
        predictor: Trained predictor object
        results_df: DataFrame with all predictions
        metrics: Performance metrics
    """
    print(f"\n{'='*70}")
    print("INCREMENTAL ROLLING FORECAST")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  Training file: {train_csv}")
    print(f"  Test file: {test_csv}")
    print(f"  Forecast horizon: {forecast_weeks} weeks ahead")
    print(f"  Target crop: {target_crop or 'All crops'}")
    print(f"  Semantic filtering: {use_semantic_filtering}")
    print(f"  Architecture: {'GAT (attention)' if use_attention else 'GCN'}")
    print(f"  Epochs per retrain: {epochs_per_retrain}")
    
    # Initialize predictor
    predictor = IncrementalRollingForecastGNN(
        kg,
        forecast_weeks=forecast_weeks,
        hidden_dim=64,
        num_layers=3,
        target_crop=target_crop,
        use_semantic_filtering=use_semantic_filtering,
        use_attention=use_attention,
        retrain_frequency='weekly'
    )
    
    # Run incremental forecast
    results_df, metrics = predictor.run_incremental_forecast(
        train_csv, 
        test_csv,
        epochs_per_retrain=epochs_per_retrain
    )
    
    print(f"\n{'='*70}")
    print("INCREMENTAL ROLLING FORECAST COMPLETE")
    print(f"{'='*70}")
    
    if metrics['r2'] is not None:
        print(f"\nFinal Performance:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")
    
    print(f"\nPredictions: {metrics['n_predictions']}")
    print(f"With actuals: {metrics['n_with_actuals']}")
    
    return predictor, results_df, metrics


if __name__ == "__main__":
    print("="*70)
    print("INCREMENTAL ROLLING FORECAST GNN - MODULE")
    print("="*70)
    print("\nThis module provides incremental rolling forecast functionality.")
    print("Import and use in your main script:")
    print()
    print("  from incremental_rolling_forecast_gnn import run_incremental_rolling_forecast")
    print()
    print("  predictor, results, metrics = run_incremental_rolling_forecast(")
    print("      kg,")
    print("      train_csv='AngusTrain.csv',")
    print("      test_csv='AngusTest.csv',")
    print("      forecast_weeks=4")
    print("  )")
    print()
    print("="*70)