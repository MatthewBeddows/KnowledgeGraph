"""
Single Date Forecast GNN
Provide a cutoff date, train on all data up to that date, predict 4 weeks ahead
Much faster than incremental rolling forecast
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


class SingleDateForecastGNN:
    """
    GNN that trains on data up to a cutoff date and predicts 4 weeks ahead
    Much faster than incremental approach - only trains once
    """
    
    def __init__(self, knowledge_graph, forecast_weeks=4, hidden_dim=64, num_layers=3,
                 target_crop=None, use_semantic_filtering=True, use_attention=True):
        self.kg = knowledge_graph
        self.forecast_weeks = forecast_weeks
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.target_crop = target_crop
        self.use_semantic_filtering = use_semantic_filtering
        self.use_attention = use_attention
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.feature_names = []
        self.relevant_features = set()
        self.feature_importance = {}
        
    def _identify_relevant_features(self):
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
    
    def load_data_up_to_date(self, train_csv, test_csv, cutoff_date):
        """Load training data and only test data up to cutoff_date (prevents leakage)"""
        print(f"\n{'='*70}")
        print(f"LOADING DATA UP TO {cutoff_date}")
        print(f"{'='*70}")
        print(f"Training file: {train_csv}")
        print(f"Test file: {test_csv}")
        
        # Load training data
        df_train = pd.read_csv(train_csv)
        print(f"Training rows: {len(df_train)}")
        
        # Load test data
        df_test = pd.read_csv(test_csv)
        print(f"Test rows: {len(df_test)}")
        
        # Parse dates for train
        if all(col in df_train.columns for col in ['year', 'month', 'dayofmonth']):
            df_train['date'] = pd.to_datetime(
                df_train[['year', 'month', 'dayofmonth']].rename(columns={'dayofmonth': 'day'})
            )
        elif 'date' in df_train.columns:
            df_train['date'] = pd.to_datetime(df_train['date'])
        else:
            raise ValueError("Cannot parse dates from training file")
        
        # Parse dates for test
        if all(col in df_test.columns for col in ['year', 'month', 'dayofmonth']):
            df_test['date'] = pd.to_datetime(
                df_test[['year', 'month', 'dayofmonth']].rename(columns={'dayofmonth': 'day'})
            )
        elif 'date' in df_test.columns:
            df_test['date'] = pd.to_datetime(df_test['date'])
        else:
            raise ValueError("Cannot parse dates from test file")
        
        # Convert cutoff_date to datetime if string
        if isinstance(cutoff_date, str):
            cutoff_date = pd.to_datetime(cutoff_date)
        
        print(f"\nCutoff date: {cutoff_date.date()}")
        print(f"Training data range: {df_train['date'].min()} to {df_train['date'].max()}")
        print(f"Test data range: {df_test['date'].min()} to {df_test['date'].max()}")
        
        # Only use test data up to cutoff date for training
        df_test_for_training = df_test[df_test['date'] <= cutoff_date].copy()
        
        if len(df_test_for_training) > 0:
            print(f"\nUsing {len(df_test_for_training)} test rows up to cutoff date")
            print(f"Test data used: {df_test_for_training['date'].min()} to {df_test_for_training['date'].max()}")
            # Combine for training
            df_combined_train = pd.concat([df_train, df_test_for_training], ignore_index=True)
        else:
            print(f"\n⚠️  No test data before cutoff date - using only training data")
            df_combined_train = df_train
        
        print(f"\nTotal training samples: {len(df_combined_train)}")
        
        # Get field identifier
        field_col = None
        if 'lookupEncoded' in df_combined_train.columns:
            field_col = 'lookupEncoded'
        elif 'FarmEncoded' in df_combined_train.columns:
            field_col = 'FarmEncoded'
        
        if field_col:
            print(f"Unique fields in training data: {df_combined_train[field_col].nunique()}")
        
        # Return training data and full test data (for getting actuals later)
        return df_combined_train, df_test
    
    def _rebuild_knowledge_graph(self, df):
        """Rebuild knowledge graph with current available data"""
        print(f"  Rebuilding knowledge graph with {len(df)} timesteps...")
        
        # Import the KG builder
        try:
            from knowledge_graph_builder import build_knowledge_graph_from_dataframe
            
            # Determine field column
            field_col = 'lookupEncoded' if 'lookupEncoded' in df.columns else 'FarmEncoded'
            if field_col not in df.columns:
                field_col = None
            
            # Rebuild the graph with current data
            self.kg = build_knowledge_graph_from_dataframe(
                df,
                target_column='target',
                field_column=field_col,
                date_column='date'
            )
            
            print(f"  ✓ Knowledge graph rebuilt successfully")
            
        except ImportError as e:
            print(f"  ⚠️  Cannot import knowledge_graph_builder: {e}")
            print(f"  This will cause prediction failures!")
            raise ImportError("knowledge_graph_builder module is required but not found")
    
    def prepare_training_data(self, df_train):
        """Prepare training data from dataframe up to cutoff date"""
        print(f"\n{'='*60}")
        print("PREPARING TRAINING DATA")
        print(f"{'='*60}")
        
        # REBUILD KNOWLEDGE GRAPH with training data only
        print(f"Rebuilding knowledge graph with training data...")
        self._rebuild_knowledge_graph(df_train)
        
        # Collect timestamps from knowledge graph
        all_timestamps = []
        for node, data in self.kg.graph.nodes(data=True):
            if data.get('type') == 'Timestamp':
                if self.target_crop and data.get('crop') != self.target_crop:
                    continue
                
                node_date = pd.to_datetime(data['datetime'])
                
                all_timestamps.append({
                    'node': node,
                    'date': node_date,
                    'field': data.get('field', 'unknown'),
                    'crop': data.get('crop'),
                    'data': data
                })
        
        all_timestamps.sort(key=lambda x: (x['field'], x['date']))
        print(f"Found {len(all_timestamps)} timestamp nodes in graph")
        
        if len(all_timestamps) < self.forecast_weeks + 1:
            raise ValueError(f"Insufficient timesteps: {len(all_timestamps)}")
        
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
        
        print(f"Created {len(train_samples)} training samples from {len(field_timestamps)} fields")
        
        if len(train_samples) == 0:
            raise ValueError("No valid training samples created")
        
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
        
        # Track categorical variables for one-hot encoding
        categorical_vars = {}
        
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
                    # Only one-hot encode specific categorical variables
                    if metric in ['VarietyEncoded', 'TunnelTypeEncoded']:
                        categorical_vars[metric] = int(value)
                    else:
                        features[metric] = float(value)
                    measurement_count += 1
        
        # Store categorical variables separately (will be one-hot encoded later)
        if categorical_vars:
            features['_categoricals'] = categorical_vars
        
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
        """Convert samples to PyTorch Geometric format with one-hot encoding for categoricals"""
        
        # First pass: identify categorical variables and their unique values
        print(f"\nProcessing features...")
        categorical_features = {}
        
        for sample in train_samples:
            if '_categoricals' in sample['features']:
                for cat_name, cat_value in sample['features']['_categoricals'].items():
                    if cat_name not in categorical_features:
                        categorical_features[cat_name] = set()
                    categorical_features[cat_name].add(cat_value)
        
        # Build feature names with one-hot encoded categoricals
        numeric_features = set()
        for sample in train_samples:
            for key in sample['features'].keys():
                if key != '_categoricals':
                    numeric_features.add(key)
        
        self.feature_names = sorted(list(numeric_features))
        
        # Add one-hot encoded categorical features
        categorical_mapping = {}
        if categorical_features:
            print(f"Found {len(categorical_features)} categorical variables:")
            for cat_name, cat_values in categorical_features.items():
                cat_values_sorted = sorted(list(cat_values))
                categorical_mapping[cat_name] = cat_values_sorted
                print(f"  {cat_name}: {len(cat_values_sorted)} unique values")
                
                # Add one-hot features
                for cat_val in cat_values_sorted:
                    self.feature_names.append(f"{cat_name}_{cat_val}")
        
        self.categorical_mapping = categorical_mapping
        print(f"Total features after one-hot encoding: {len(self.feature_names)}")
        
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
        
        # Fill feature matrix
        for sample in train_samples:
            node_idx = self.node_to_idx[sample['node']]
            
            # Numeric features
            for i, feat_name in enumerate(self.feature_names):
                if feat_name in sample['features']:
                    x[node_idx, i] = sample['features'][feat_name]
                elif '_' in feat_name and not feat_name.startswith('shifted_'):
                    # One-hot encoded feature
                    cat_name, cat_val = feat_name.rsplit('_', 1)
                    if '_categoricals' in sample['features']:
                        if cat_name in sample['features']['_categoricals']:
                            if str(sample['features']['_categoricals'][cat_name]) == cat_val:
                                x[node_idx, i] = 1.0
            
            y[node_idx] = sample['target']
            train_mask[node_idx] = True
        
        # Check feature variance BEFORE scaling
        train_x_raw = x[train_mask]
        feature_variance = np.var(train_x_raw, axis=0)
        
        # Identify features with actual variance
        variance_threshold = 1e-6
        valid_features = feature_variance > variance_threshold
        n_zero_var = (~valid_features).sum()
        
        if n_zero_var > 0:
            print(f"\n⚠️  Removing {n_zero_var} zero-variance features")
            print(f"  Features with variance: {valid_features.sum()}/{len(self.feature_names)}")
            
            # Keep only features with variance
            self.feature_names = [f for i, f in enumerate(self.feature_names) if valid_features[i]]
            x = x[:, valid_features]
            
            print(f"  Remaining features: {len(self.feature_names)}")
        
        if len(self.feature_names) == 0:
            raise ValueError("All features have zero variance! Cannot train model.")
        
        # Scale features (but not one-hot encoded ones)
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
    
    def train_model(self, data, epochs=200):
        """Train the GNN model"""
        print(f"\n{'='*60}")
        print("TRAINING MODEL")
        print(f"{'='*60}")
        
        num_features = data.x.shape[1]
        print(f"Number of features: {num_features}")
        print(f"Training samples: {data.train_mask.sum().item()}")
        print(f"Total nodes: {data.x.shape[0]}")
        
        # Check if features have variance
        train_x = data.x[data.train_mask]
        feature_variance = train_x.var(dim=0)
        zero_variance_features = (feature_variance < 1e-6).sum().item()
        
        if zero_variance_features > 0:
            print(f"⚠️  WARNING: {zero_variance_features}/{num_features} features have near-zero variance!")
        
        # Check target distribution
        train_y = data.y[data.train_mask]
        print(f"Target range: [{train_y.min().item():.2f}, {train_y.max().item():.2f}]")
        print(f"Target mean: {train_y.mean().item():.2f}, std: {train_y.std().item():.2f}")
        
        self.model = ContextAwareYieldGNN(num_features, self.hidden_dim,
                                          self.num_layers, self.use_attention)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        self.model = self.model.to(device)
        data = data.to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        
        print(f"\nTraining for {epochs} epochs...")
        self.model.train()
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
            
            if (epoch + 1) % 50 == 0:
                # Check prediction variance
                with torch.no_grad():
                    train_preds = out[data.train_mask]
                    pred_std = train_preds.std().item()
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Pred std: {pred_std:.4f}")
        
        print(f"✓ Training complete, best loss: {best_loss:.4f}")
        
        # Final check on predictions
        self.model.eval()
        with torch.no_grad():
            final_out = self.model(data.x, data.edge_index)
            train_preds = final_out[data.train_mask]
            print(f"\nFinal training predictions:")
            print(f"  Range: [{train_preds.min().item():.2f}, {train_preds.max().item():.2f}]")
            print(f"  Mean: {train_preds.mean().item():.2f}, Std: {train_preds.std().item():.2f}")
            
            if train_preds.std().item() < 0.01:
                print(f"⚠️  WARNING: Model predictions have very low variance!")
                print(f"  This suggests the model is predicting nearly constant values")
    
    def predict_future(self, df_train, df_test, cutoff_date, only_predict_fields_with_actuals=True):
        """Predict 4 weeks ahead from cutoff_date for all fields"""
        print(f"\n{'='*60}")
        print(f"PREDICTING {self.forecast_weeks} WEEKS AHEAD FROM {cutoff_date.date()}")
        print(f"{'='*60}")
        
        target_date = cutoff_date + timedelta(weeks=self.forecast_weeks)
        print(f"Target date: {target_date.date()}")
        
        predictions = []
        
        # Find all fields
        field_col = 'lookupEncoded' if 'lookupEncoded' in df_train.columns else 'FarmEncoded'
        
        if field_col not in df_train.columns:
            raise ValueError(f"Field column not found")
        
        # Determine which fields to predict for
        if only_predict_fields_with_actuals:
            # Only predict for fields that have data at target_date in test set
            test_at_target = df_test[df_test['date'] == target_date]
            unique_fields = test_at_target[field_col].unique()
            print(f"Predicting only for {len(unique_fields)} fields with actuals at target date")
        else:
            # Predict for all fields in training data
            unique_fields = df_train[field_col].unique()
            print(f"Predicting for all {len(unique_fields)} fields in training data")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # PRE-BUILD timestamp lookup (date, field_code) -> node
        print(f"Building timestamp lookup index...")
        timestamp_lookup = {}
        field_codes_in_graph = set()
        sample_keys = []
        date_range_in_graph = {'min': None, 'max': None}
        
        for node, data in self.kg.graph.nodes(data=True):
            if data.get('type') == 'Timestamp':
                node_date = pd.to_datetime(data['datetime']).date()
                field_id = data.get('field', '')
                
                # Track date range
                if date_range_in_graph['min'] is None or node_date < date_range_in_graph['min']:
                    date_range_in_graph['min'] = node_date
                if date_range_in_graph['max'] is None or node_date > date_range_in_graph['max']:
                    date_range_in_graph['max'] = node_date
                
                if 'Plot_' in field_id:
                    node_field_code = int(field_id.split('Plot_')[1].split('_')[0])
                else:
                    node_field_code = 0
                
                key = (node_date, node_field_code)
                timestamp_lookup[key] = node
                field_codes_in_graph.add(node_field_code)
                
                if len(sample_keys) < 5:
                    sample_keys.append((key, field_id))
        
        print(f"  Indexed {len(timestamp_lookup)} timestamp nodes")
        print(f"  Date range in graph: {date_range_in_graph['min']} to {date_range_in_graph['max']}")
        print(f"  Cutoff date: {cutoff_date.date()}")
        print(f"  Field codes in graph: {sorted(list(field_codes_in_graph))[:10]}... ({len(field_codes_in_graph)} unique)")
        print(f"  Sample lookup keys: {sample_keys[:3]}")
        
        # Check what field codes we're looking for
        print(f"\nTarget field codes from data: {sorted(unique_fields)[:10]}... ({len(unique_fields)} total)")
        
        # Check for matches
        matching_fields = set(unique_fields) & field_codes_in_graph
        print(f"Overlapping field codes: {len(matching_fields)}")
        if len(matching_fields) == 0:
            print("⚠️  WARNING: No field code overlap between data and graph!")
            print(f"  Data field codes example: {sorted(unique_fields)[:5]}")
            print(f"  Graph field codes example: {sorted(list(field_codes_in_graph))[:5]}")
        
        
        # PRE-COMPUTE: Create graph data once (not in the loop!)
        print(f"Preparing graph data for inference...")
        all_data = self._get_full_graph_data()
        all_data = all_data.to(device)
        
        # Run inference once for ALL nodes
        print(f"Running inference on all nodes...")
        self.model.eval()
        with torch.no_grad():
            all_predictions = self.model(all_data.x, all_data.edge_index)
        
        print(f"Collecting predictions for {len(unique_fields)} target fields...")
        successful_predictions = 0
        failed_reasons = {
            'no_field_data': 0,
            'no_timestamp_node': 0,
            'no_features': 0,
            'no_node_idx': 0
        }
        
        # Debug: track first few failed lookups
        failed_lookups = []
        
        for field_code in unique_fields:
            # Find the most recent timestamp for this field before cutoff
            field_data = df_train[df_train[field_col] == field_code]
            field_data = field_data[field_data['date'] <= cutoff_date]
            
            if len(field_data) == 0:
                failed_reasons['no_field_data'] += 1
                continue
            
            latest_date = field_data['date'].max().date()
            
            # Fast lookup using pre-built index
            key = (latest_date, int(field_code))
            ts_node = timestamp_lookup.get(key)
            
            if ts_node is None:
                failed_reasons['no_timestamp_node'] += 1
                if len(failed_lookups) < 5:
                    failed_lookups.append({
                        'field_code': field_code,
                        'latest_date': latest_date,
                        'key': key
                    })
                continue
            
            # Get features (just for validation)
            features = self._extract_filtered_features(ts_node)
            
            if features is None:
                failed_reasons['no_features'] += 1
                continue
            
            # Get node index
            node_idx = self.node_to_idx.get(ts_node)
            if node_idx is None:
                failed_reasons['no_node_idx'] += 1
                continue
            
            # Get prediction (already computed!)
            pred_scaled = all_predictions[node_idx].cpu().item()
            
            # Inverse transform
            pred_orig = self.scaler_y.inverse_transform([[pred_scaled]])[0][0]
            
            # Get actual value from test data (no leakage - only for evaluation)
            actual = self._get_actual_yield(df_test, field_code, target_date)
            
            predictions.append({
                'cutoff_date': cutoff_date,
                'target_date': target_date,
                'field_code': int(field_code),
                'predicted': pred_orig,
                'actual': actual,
                'weeks_ahead': self.forecast_weeks
            })
            
            successful_predictions += 1
            if successful_predictions % 20 == 0:
                print(f"  Processed {successful_predictions}/{len(unique_fields)} fields...")
        
        # Print failure summary
        if sum(failed_reasons.values()) > 0:
            print(f"\n⚠️  Failed to generate predictions for {sum(failed_reasons.values())} fields:")
            for reason, count in failed_reasons.items():
                if count > 0:
                    print(f"  - {reason}: {count}")
            
            if failed_lookups:
                print(f"\nSample failed lookups:")
                for fl in failed_lookups[:3]:
                    print(f"  Looking for: field={fl['field_code']}, date={fl['latest_date']}, key={fl['key']}")
        
        print(f"✓ Made {len(predictions)} predictions")
        if only_predict_fields_with_actuals:
            predictions_with_actuals = [p for p in predictions if p['actual'] is not None]
            print(f"  {len(predictions_with_actuals)} have actual values for evaluation")
        
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
        
        # Initialize with zeros
        x = np.zeros((num_nodes, num_features))
        
        # IMPORTANT: Fill in actual features for each node!
        print(f"  Extracting features for {num_nodes} nodes...")
        nodes_with_features = 0
        sample_features_extracted = []
        
        for node in all_nodes:
            node_data = self.kg.graph.nodes[node]
            if node_data.get('type') == 'Timestamp':
                # Extract features for this timestamp
                features = self._extract_filtered_features(node)
                if features is not None:
                    node_idx = self.node_to_idx[node]
                    
                    # Debug: Save first few feature extractions
                    if len(sample_features_extracted) < 3:
                        sample_features_extracted.append({
                            'node': node,
                            'features': features.copy()
                        })
                    
                    # Fill numeric features
                    for i, feat_name in enumerate(self.feature_names):
                        if feat_name in features:
                            x[node_idx, i] = features[feat_name]
                        elif '_' in feat_name and not feat_name.startswith('shifted_'):
                            # One-hot encoded feature
                            parts = feat_name.rsplit('_', 1)
                            if len(parts) == 2:
                                cat_name, cat_val = parts
                                if '_categoricals' in features:
                                    if cat_name in features['_categoricals']:
                                        if str(features['_categoricals'][cat_name]) == cat_val:
                                            x[node_idx, i] = 1.0
                    
                    nodes_with_features += 1
            
            if (nodes_with_features % 2000 == 0 and nodes_with_features > 0):
                print(f"    Processed {nodes_with_features} timestamp nodes...")
        
        print(f"  ✓ Extracted features for {nodes_with_features} timestamp nodes")
        
        # Debug: Check feature variance
        if nodes_with_features > 0:
            feature_variance = np.var(x, axis=0)
            nonzero_features = (feature_variance > 1e-6).sum()
            print(f"  Features with variance: {nonzero_features}/{num_features}")
            
            if nonzero_features < 5:
                print(f"  ⚠️  WARNING: Very few features have variance!")
                print(f"  Sample extracted features:")
                for sample in sample_features_extracted[:2]:
                    print(f"    Node {sample['node']}: {len(sample['features'])} features")
                    # Show a few feature values
                    feature_items = list(sample['features'].items())[:5]
                    for k, v in feature_items:
                        if k != '_categoricals':
                            print(f"      {k}: {v}")
        
        # Scale features
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
    
    def visualize_results(self, predictions):
        """Create visualizations"""
        print(f"\n{'='*60}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*60}")
        
        os.makedirs('outputs/gnn/plots', exist_ok=True)
        
        results_df = pd.DataFrame(predictions)
        df_with_actuals = results_df[results_df['actual'].notna()]
        
        if len(df_with_actuals) == 0:
            print("⚠️  No actual values available for visualization")
            return
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(df_with_actuals['actual'], df_with_actuals['predicted']))
        mae = mean_absolute_error(df_with_actuals['actual'], df_with_actuals['predicted'])
        r2 = r2_score(df_with_actuals['actual'], df_with_actuals['predicted'])
        
        print(f"\nPerformance Metrics:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        # Create plots
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(df_with_actuals['actual'], df_with_actuals['predicted'], alpha=0.6)
        plt.plot([df_with_actuals['actual'].min(), df_with_actuals['actual'].max()],
                [df_with_actuals['actual'].min(), df_with_actuals['actual'].max()],
                'r--', lw=2)
        plt.xlabel('Actual Yield')
        plt.ylabel('Predicted Yield')
        plt.title(f'Predicted vs Actual (R² = {r2:.3f})')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        df_with_actuals['error'] = df_with_actuals['predicted'] - df_with_actuals['actual']
        plt.hist(df_with_actuals['error'], bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution (MAE = {mae:.3f})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/gnn/plots/single_date_forecast_results.png', dpi=150)
        plt.close()
        
        print(f"✓ Saved: outputs/gnn/plots/single_date_forecast_results.png")


def run_single_date_forecast(kg, cutoff_date, train_csv='AngusTrain.csv', 
                             test_csv='AngusTest.csv',
                             forecast_weeks=4,
                             target_crop=None,
                             use_semantic_filtering=True,
                             use_attention=True,
                             epochs=200,
                             only_predict_fields_with_actuals=True):
    """
    Train on data up to cutoff_date and predict 4 weeks ahead
    
    NO DATA LEAKAGE: Only uses train data + test data up to cutoff for training
    
    Args:
        kg: Knowledge graph object
        cutoff_date: Date to cut training data (str or datetime)
                    e.g., '2024-08-01' or pd.to_datetime('2024-08-01')
        train_csv: Training data CSV file (AngusTrain.csv)
        test_csv: Test data CSV file (AngusTest.csv)
        forecast_weeks: How many weeks ahead to predict (default: 4)
        target_crop: Specific crop to focus on (None = all)
        use_semantic_filtering: Use KG to filter features
        use_attention: Use Graph Attention Networks
        epochs: Number of training epochs
        only_predict_fields_with_actuals: If True, only predict for fields that have
                                         actual data at target date (default: True)
                                         Set to False to predict for all fields in training
    
    Returns:
        predictor: Trained predictor object
        predictions: List of prediction dictionaries
        metrics: Performance metrics (if actuals available)
    """
    print(f"\n{'='*70}")
    print("SINGLE DATE FORECAST GNN")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  Training file: {train_csv}")
    print(f"  Test file: {test_csv}")
    print(f"  Cutoff date: {cutoff_date}")
    print(f"  Forecast horizon: {forecast_weeks} weeks ahead")
    print(f"  Target crop: {target_crop or 'All crops'}")
    print(f"  Semantic filtering: {use_semantic_filtering}")
    print(f"  Architecture: {'GAT (attention)' if use_attention else 'GCN'}")
    print(f"  Training epochs: {epochs}")
    
    # Initialize predictor
    predictor = SingleDateForecastGNN(
        kg,
        forecast_weeks=forecast_weeks,
        hidden_dim=64,
        num_layers=3,
        target_crop=target_crop,
        use_semantic_filtering=use_semantic_filtering,
        use_attention=use_attention
    )
    
    # Identify relevant features
    predictor._identify_relevant_features()
    
    # Load data up to cutoff date (NO LEAKAGE)
    df_train, df_test = predictor.load_data_up_to_date(train_csv, test_csv, cutoff_date)
    
    # Prepare training data
    graph_data, train_samples = predictor.prepare_training_data(df_train)
    
    # Train model
    predictor.train_model(graph_data, epochs=epochs)
    
    # Make predictions (actuals from test set only for evaluation)
    predictions = predictor.predict_future(df_train, df_test, pd.to_datetime(cutoff_date),
                                          only_predict_fields_with_actuals=only_predict_fields_with_actuals)
    
    # Calculate metrics if actuals available
    predictions_df = pd.DataFrame(predictions)
    
    # Check if we have any predictions
    if len(predictions_df) == 0:
        print("\n⚠️  WARNING: No predictions were generated!")
        print("This could mean:")
        print("  - No matching timestamp nodes found in graph")
        print("  - Feature extraction failed for all fields")
        print("  - No fields had sufficient data before cutoff date")
        
        metrics = {
            'rmse': None,
            'mae': None,
            'r2': None,
            'n_predictions': 0,
            'n_with_actuals': 0
        }
        
        # Return empty DataFrame instead of list
        return predictor, pd.DataFrame(), metrics
    
    # Check if 'actual' column exists
    if 'actual' not in predictions_df.columns:
        print("\n⚠️  WARNING: Predictions generated but 'actual' column missing!")
        print(f"Columns in predictions: {predictions_df.columns.tolist()}")
        
        metrics = {
            'rmse': None,
            'mae': None,
            'r2': None,
            'n_predictions': len(predictions),
            'n_with_actuals': 0
        }
        
        # Return DataFrame instead of list
        return predictor, predictions_df, metrics
    
    df_with_actuals = predictions_df[predictions_df['actual'].notna()]
    
    if len(df_with_actuals) > 0:
        rmse = np.sqrt(mean_squared_error(df_with_actuals['actual'], df_with_actuals['predicted']))
        mae = mean_absolute_error(df_with_actuals['actual'], df_with_actuals['predicted'])
        r2 = r2_score(df_with_actuals['actual'], df_with_actuals['predicted'])
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_predictions': len(predictions),
            'n_with_actuals': len(df_with_actuals)
        }
    else:
        metrics = {
            'rmse': None,
            'mae': None,
            'r2': None,
            'n_predictions': len(predictions),
            'n_with_actuals': 0
        }
    
    # Visualize if actuals available
    if len(df_with_actuals) > 0:
        predictor.visualize_results(predictions)
    
    # Save predictions
    os.makedirs('outputs/gnn', exist_ok=True)
    predictions_df.to_csv('outputs/gnn/single_date_forecast_predictions.csv', index=False)
    print(f"\n✓ Predictions saved to: outputs/gnn/single_date_forecast_predictions.csv")
    
    print(f"\n{'='*70}")
    print("FORECAST COMPLETE")
    print(f"{'='*70}")
    
    # Return DataFrame instead of list
    return predictor, predictions_df, metrics


if __name__ == "__main__":
    print("="*70)
    print("SINGLE DATE FORECAST GNN - MODULE")
    print("="*70)
    print("\nThis module provides single-date forecast functionality.")
    print("Much faster than incremental rolling forecast!")
    print()
    print("✓ NO DATA LEAKAGE: Keeps train/test separate")
    print()
    print("Usage Example:")
    print()
    print("  from single_date_forecast_gnn import run_single_date_forecast")
    print()
    print("  predictor, predictions, metrics = run_single_date_forecast(")
    print("      kg,")
    print("      train_csv='AngusTrain.csv',")
    print("      test_csv='AngusTest.csv',")
    print("      cutoff_date='2024-08-01',  # Train up to this date")
    print("      forecast_weeks=4,          # Predict 4 weeks ahead")
    print("      epochs=200")
    print("  )")
    print()
    print("How it prevents leakage:")
    print("  1. Uses all of AngusTrain.csv")
    print("  2. Only uses AngusTest.csv rows with date <= cutoff_date")
    print("  3. Predicts 4 weeks from cutoff_date")
    print("  4. Compares with actual values from AngusTest.csv (evaluation only)")
    print()
    print("="*70)