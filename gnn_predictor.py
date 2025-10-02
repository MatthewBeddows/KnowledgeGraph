"""
Graph Neural Network for Rolling Forecast Yield Prediction
Uses features from week T to predict yield at week T+N
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
import warnings
import copy
warnings.filterwarnings('ignore')


class YieldGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, num_layers=3):
        super(YieldGNN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(GCNConv(num_node_features, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 1):
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


class RollingForecastGNN:
    def __init__(self, knowledge_graph, forecast_weeks=4, hidden_dim=64, num_layers=3):
        self.kg_original = knowledge_graph
        self.kg = None  # Working copy
        self.forecast_weeks = forecast_weeks
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.feature_names = []
        
    def prepare_rolling_forecast_data(self, test_year=2023):
        """
        Create training samples where:
        - X = all features at week T (soil, weather, temporal)
        - Y = yield at week T + forecast_weeks
        """
        print(f"\n{'='*60}")
        print(f"PREPARING ROLLING FORECAST DATA")
        print(f"{'='*60}")
        print(f"Forecast horizon: {self.forecast_weeks} weeks ahead")
        print(f"Test year: {test_year}")
        
        # Create working copy of graph
        print("\nCreating working copy of knowledge graph...")
        self.kg = copy.deepcopy(self.kg_original)
        
        # Collect all timestamps with data
        print("Collecting timestamps...")
        all_timestamps = []
        for node, data in self.kg.graph.nodes(data=True):
            if data.get('type') == 'Timestamp':
                year = data.get('year', 0)
                if year > 0:  # Valid timestamp
                    all_timestamps.append({
                        'node': node,
                        'date': pd.to_datetime(data['datetime']),
                        'year': year,
                        'field': data.get('field', 'unknown'),
                        'data': data
                    })
        
        # Sort by field and date
        all_timestamps.sort(key=lambda x: (x['field'], x['date']))
        print(f"Found {len(all_timestamps)} timestamps")
        
        # Group by field
        field_timestamps = {}
        for ts in all_timestamps:
            field = ts['field']
            if field not in field_timestamps:
                field_timestamps[field] = []
            field_timestamps[field].append(ts)
        
        print(f"Across {len(field_timestamps)} fields/plots")
        
        # Create training samples
        train_samples = []
        test_samples = []
        
        for field_id, timestamps in field_timestamps.items():
            for i in range(len(timestamps) - self.forecast_weeks):
                current_ts = timestamps[i]
                future_ts = timestamps[i + self.forecast_weeks]
                
                # Extract features from current timestamp
                features = self._extract_all_features(current_ts['node'])
                
                # Extract target from future timestamp
                target = self._get_yield_for_timestamp(future_ts['node'])
                
                if features is not None and target is not None:
                    sample = {
                        'node': current_ts['node'],
                        'features': features,
                        'target': target,
                        'field': field_id,
                        'date': current_ts['date'],
                        'target_date': future_ts['date'],
                        'year': current_ts['year']
                    }
                    
                    if current_ts['year'] < test_year:
                        train_samples.append(sample)
                    else:
                        test_samples.append(sample)
        
        print(f"\nSample Creation:")
        print(f"  Training samples: {len(train_samples)} (years < {test_year})")
        print(f"  Test samples: {len(test_samples)} (year = {test_year})")
        
        # Convert to graph format
        graph_data = self._samples_to_graph_data(train_samples, test_samples)
        
        return graph_data, train_samples, test_samples
    
    def _extract_all_features(self, timestamp_node):
        """Extract all available features for a timestamp"""
        features = {}
        
        # Temporal features
        node_data = self.kg.graph.nodes[timestamp_node]
        features['month'] = node_data.get('month', 0) / 12.0
        features['day'] = node_data.get('day', 0) / 31.0
        features['dayofweek'] = node_data.get('dayofweek', 0) / 7.0
        features['dayofyear'] = node_data.get('dayofyear', 0) / 365.0
        
        # Get field encoding
        field_id = node_data.get('field', '')
        if 'Plot_' in field_id:
            plot_num = int(field_id.split('Plot_')[1].split('_')[0])
            features['plot_encoded'] = plot_num / 100.0
        else:
            features['plot_encoded'] = 0.0
        
        # Extract all measurements (soil, weather, etc.) EXCEPT target
        measurement_count = 0
        for neighbor in self.kg.graph.neighbors(timestamp_node):
            neighbor_data = self.kg.graph.nodes[neighbor]
            if neighbor_data.get('type') == 'Measurement':
                metric = neighbor_data.get('metric', '')
                value = neighbor_data.get('value')
                
                # Skip target variable
                if 'target' in metric.lower():
                    continue
                
                if value is not None:
                    features[metric] = float(value)
                    measurement_count += 1
        
        # Need at least some measurements beyond temporal
        if measurement_count < 3:
            return None
        
        return features
    
    def _get_yield_for_timestamp(self, timestamp_node):
        """Get yield/target value for a timestamp"""
        for neighbor in self.kg.graph.neighbors(timestamp_node):
            node_data = self.kg.graph.nodes[neighbor]
            if node_data.get('type') == 'Measurement':
                metric = node_data.get('metric', '')
                if 'target' in metric.lower():
                    return float(node_data.get('value', 0))
        return None
    
    def _samples_to_graph_data(self, train_samples, test_samples):
        """Convert samples to PyTorch Geometric format"""
        print("\nConverting to graph format...")
        
        # Build feature matrix
        all_samples = train_samples + test_samples
        
        # Determine feature names from first sample
        if len(all_samples) > 0:
            self.feature_names = sorted(all_samples[0]['features'].keys())
            print(f"Features: {len(self.feature_names)}")
            print(f"  Temporal: month, day, dayofweek, dayofyear, plot_encoded")
            print(f"  Measurements: {len(self.feature_names) - 5}")
        
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
        
        # Initialize feature matrix and labels
        num_nodes = len(all_nodes)
        num_features = len(self.feature_names)
        x = np.zeros((num_nodes, num_features))
        y = np.zeros(num_nodes)
        train_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        
        # Fill in training samples
        for sample in train_samples:
            node_idx = self.node_to_idx[sample['node']]
            x[node_idx] = [sample['features'].get(f, 0.0) for f in self.feature_names]
            y[node_idx] = sample['target']
            train_mask[node_idx] = True
        
        # Fill in test samples
        for sample in test_samples:
            node_idx = self.node_to_idx[sample['node']]
            x[node_idx] = [sample['features'].get(f, 0.0) for f in self.feature_names]
            y[node_idx] = sample['target']
            test_mask[node_idx] = True
        
        # Scale features
        print("\nScaling features...")
        self.scaler_X.fit(x[train_mask])
        x = self.scaler_X.transform(x)
        
        # Scale targets
        self.scaler_y.fit(y[train_mask].reshape(-1, 1))
        y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).flatten()
        
        # Convert to tensors
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y_scaled, dtype=torch.float)
        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        test_mask = torch.tensor(test_mask, dtype=torch.bool)
        
        print(f"\nGraph data prepared:")
        print(f"  Nodes: {num_nodes}")
        print(f"  Edges: {edge_index.shape[1] // 2}")
        print(f"  Features: {num_features}")
        print(f"  Training nodes: {train_mask.sum().item()}")
        print(f"  Test nodes: {test_mask.sum().item()}")
        
        data = Data(x=x, edge_index=edge_index, y=y,
                   train_mask=train_mask, test_mask=test_mask)
        
        return data
    
    def train(self, data, epochs=200, lr=0.01, weight_decay=5e-4):
        """Train the GNN model"""
        print(f"\n{'='*60}")
        print("TRAINING GNN MODEL")
        print(f"{'='*60}")
        
        num_features = data.x.shape[1]
        self.model = YieldGNN(num_features, self.hidden_dim, self.num_layers)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        self.model = self.model.to(device)
        data = data.to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=lr, weight_decay=weight_decay)
        
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        print(f"\nTraining for up to {epochs} epochs...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    
    def predict_and_evaluate(self, data, test_samples):
        """Make predictions and evaluate"""
        print(f"\n{'='*60}")
        print("EVALUATION")
        print(f"{'='*60}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            
            # Training predictions
            train_pred = out[data.train_mask].cpu().numpy()
            train_true = data.y[data.train_mask].cpu().numpy()
            
            # Test predictions
            test_pred = out[data.test_mask].cpu().numpy()
            test_true = data.y[data.test_mask].cpu().numpy()
        
        # Inverse transform
        train_pred_orig = self.scaler_y.inverse_transform(train_pred.reshape(-1, 1)).flatten()
        train_true_orig = self.scaler_y.inverse_transform(train_true.reshape(-1, 1)).flatten()
        test_pred_orig = self.scaler_y.inverse_transform(test_pred.reshape(-1, 1)).flatten()
        test_true_orig = self.scaler_y.inverse_transform(test_true.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        print("\nTraining Set Performance:")
        train_rmse = np.sqrt(mean_squared_error(train_true_orig, train_pred_orig))
        train_mae = mean_absolute_error(train_true_orig, train_pred_orig)
        train_r2 = r2_score(train_true_orig, train_pred_orig)
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  MAE:  {train_mae:.4f}")
        print(f"  R²:   {train_r2:.4f}")
        
        print("\nTest Set Performance:")
        test_rmse = np.sqrt(mean_squared_error(test_true_orig, test_pred_orig))
        test_mae = mean_absolute_error(test_true_orig, test_pred_orig)
        test_r2 = r2_score(test_true_orig, test_pred_orig)
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE:  {test_mae:.4f}")
        print(f"  R²:   {test_r2:.4f}")
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'date': [s['date'] for s in test_samples],
            'target_date': [s['target_date'] for s in test_samples],
            'field': [s['field'] for s in test_samples],
            'predicted': test_pred_orig,
            'actual': test_true_orig
        })
        
        # Extract plot numbers
        results_df['plot'] = results_df['field'].apply(
            lambda x: int(x.split('Plot_')[1].split('_')[0]) if 'Plot_' in x else 0
        )
        
        # Save results
        os.makedirs('outputs/gnn', exist_ok=True)
        results_df.to_csv('outputs/gnn/rolling_forecast_results.csv', index=False)
        print(f"\nResults saved to: outputs/gnn/rolling_forecast_results.csv")
        
        # Create visualizations
        self._create_visualizations(results_df)
        
        return results_df, {
            'train': {'rmse': train_rmse, 'mae': train_mae, 'r2': train_r2},
            'test': {'rmse': test_rmse, 'mae': test_mae, 'r2': test_r2}
        }
    
    def _create_visualizations(self, results_df):
        """Create visualization plots"""
        print("\nCreating visualizations...")
        os.makedirs('outputs/gnn/plots', exist_ok=True)
        
        # Overall scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(results_df['actual'], results_df['predicted'], alpha=0.5, s=50)
        
        # Perfect prediction line
        min_val = min(results_df['actual'].min(), results_df['predicted'].min())
        max_val = max(results_df['actual'].max(), results_df['predicted'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Yield', fontsize=12)
        plt.ylabel('Predicted Yield', fontsize=12)
        plt.title(f'Rolling Forecast: {self.forecast_weeks}-Week Ahead Prediction', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/gnn/plots/overall_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Time series by plot
        unique_plots = sorted(results_df['plot'].unique())
        print(f"  Creating time series for {len(unique_plots)} plots...")
        
        for plot_num in unique_plots:
            plot_data = results_df[results_df['plot'] == plot_num].copy()
            plot_data = plot_data.sort_values('target_date')
            
            plt.figure(figsize=(12, 6))
            plt.plot(plot_data['target_date'], plot_data['actual'], 
                    label='Actual', marker='o', linewidth=2, markersize=6)
            plt.plot(plot_data['target_date'], plot_data['predicted'],
                    label='Predicted', marker='s', linewidth=2, markersize=6, linestyle='--')
            
            plt.xlabel('Target Date (Predicted Week)', fontsize=12)
            plt.ylabel('Yield', fontsize=12)
            plt.title(f'Plot {int(plot_num)} - {self.forecast_weeks}-Week Rolling Forecast',
                     fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plt.savefig(f'outputs/gnn/plots/plot_{int(plot_num)}_timeseries.png',
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  Saved plots to: outputs/gnn/plots/")
    
    def save_model(self, filepath='outputs/gnn/rolling_forecast_model.pt'):
        """Save model and scalers"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'node_to_idx': self.node_to_idx,
            'idx_to_node': self.idx_to_node,
            'feature_names': self.feature_names,
            'forecast_weeks': self.forecast_weeks,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers
        }
        
        torch.save(save_dict, filepath)
        print(f"\nModel saved to: {filepath}")


def run_rolling_forecast(kg, forecast_weeks=4, test_year=2023):
    """
    Main function to run rolling forecast
    
    Args:
        kg: Knowledge graph object
        forecast_weeks: How many weeks ahead to predict
        test_year: Year to use for testing
    """
    print(f"\n{'='*60}")
    print("ROLLING FORECAST GNN")
    print(f"{'='*60}")
    print(f"\nConfiguration:")
    print(f"  Forecast horizon: {forecast_weeks} weeks ahead")
    print(f"  Test year: {test_year}")
    print(f"  Training: years < {test_year}")
    
    # Initialize predictor
    predictor = RollingForecastGNN(kg, forecast_weeks=forecast_weeks,
                                   hidden_dim=64, num_layers=3)
    
    # Prepare data
    graph_data, train_samples, test_samples = predictor.prepare_rolling_forecast_data(
        test_year=test_year
    )
    
    # Train model
    predictor.train(graph_data, epochs=200, lr=0.01)
    
    # Evaluate
    results_df, metrics = predictor.predict_and_evaluate(graph_data, test_samples)
    
    # Save model
    predictor.save_model()
    
    print(f"\n{'='*60}")
    print("ROLLING FORECAST COMPLETE")
    print(f"{'='*60}")
    
    return predictor, results_df, metrics


if __name__ == "__main__":
    print("Import and run with:")
    print("  from rolling_forecast_gnn import run_rolling_forecast")
    print("  predictor, results, metrics = run_rolling_forecast(kg, forecast_weeks=4, test_year=2023)")