"""
Graph Neural Network for Strawberry Yield Prediction
Uses knowledge graph structure to predict yields
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
import warnings
warnings.filterwarnings('ignore')


class YieldGNN(torch.nn.Module):
    """Graph Neural Network for yield prediction"""
    
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


class GNNYieldPredictor:
    """Wrapper for GNN-based yield prediction"""
    
    def __init__(self, knowledge_graph, hidden_dim=64, num_layers=3):
        self.kg = knowledge_graph
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.node_to_idx = {}
        self.idx_to_node = {}
        
    def prepare_graph_data(self, test_year=2023):
        """Convert knowledge graph to PyTorch Geometric format"""
        
        print("Preparing graph data...")
        print(f"  Train mask: years < {test_year}")
        print(f"  Test mask: year = {test_year}")
        
        all_nodes = list(self.kg.graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        edge_list = []
        for src, dst in self.kg.graph.edges():
            src_idx = self.node_to_idx[src]
            dst_idx = self.node_to_idx[dst]
            edge_list.append([src_idx, dst_idx])
            edge_list.append([dst_idx, src_idx])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        node_features = []
        node_labels = []
        train_mask = []
        test_mask = []
        
        for node in all_nodes:
            node_data = self.kg.graph.nodes[node]
            node_type = node_data.get('type', 'Unknown')
            
            features = self._create_node_features(node, node_data, node_type)
            node_features.append(features)
            
            if node_type == 'Timestamp':
                yield_val = self._get_yield_for_timestamp(node)
                node_labels.append(yield_val if yield_val is not None else 0.0)
                
                year = node_data.get('year', 2020)
                if yield_val is not None:
                    if year < test_year:
                        train_mask.append(True)
                        test_mask.append(False)
                    else:
                        train_mask.append(False)
                        test_mask.append(True)
                else:
                    train_mask.append(False)
                    test_mask.append(False)
            else:
                node_labels.append(0.0)
                train_mask.append(False)
                test_mask.append(False)
        
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.float)
        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        test_mask = torch.tensor(test_mask, dtype=torch.bool)
        
        x_train = x[train_mask]
        if len(x_train) > 0:
            self.scaler_X.fit(x_train.numpy())
            x = torch.tensor(self.scaler_X.transform(x.numpy()), dtype=torch.float)
        
        y_train = y[train_mask].reshape(-1, 1)
        if len(y_train) > 0:
            self.scaler_y.fit(y_train.numpy())
            y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).flatten()
            y = torch.tensor(y_scaled, dtype=torch.float)
        
        print(f"Graph prepared:")
        print(f"  Total nodes: {len(all_nodes)}")
        print(f"  Total edges: {len(edge_list)}")
        print(f"  Feature dimension: {x.shape[1]}")
        print(f"  Training samples: {train_mask.sum().item()}")
        print(f"  Test samples: {test_mask.sum().item()}")
        
        print(f"\nYears found in timestamp nodes:")
        years_found = {}
        for node in all_nodes:
            node_data = self.kg.graph.nodes[node]
            if node_data.get('type') == 'Timestamp':
                year = node_data.get('year')
                if year:
                    years_found[year] = years_found.get(year, 0) + 1
        for year in sorted(years_found.keys()):
            is_train = "TRAIN" if year < test_year else "TEST"
            print(f"  {year}: {years_found[year]} timestamps ({is_train})")
        
        data = Data(x=x, edge_index=edge_index, y=y, 
                   train_mask=train_mask, test_mask=test_mask)
        
        return data
    
    def _create_node_features(self, node_id, node_data, node_type):
        """Create feature vector for a node"""
        features = []
        
        type_encoding = [0] * 7
        type_map = {'Field': 0, 'Timestamp': 1, 'Measurement': 2, 
                   'Year': 3, 'Month': 4, 'Week': 5, 'Farm': 6}
        if node_type in type_map:
            type_encoding[type_map[node_type]] = 1
        features.extend(type_encoding)
        
        if node_type == 'Timestamp':
            features.extend([
                node_data.get('year', 0) / 2023.0,
                node_data.get('month', 0) / 12.0,
                node_data.get('day', 0) / 31.0,
                node_data.get('dayofweek', 0) / 7.0,
                node_data.get('dayofyear', 0) / 365.0
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        if node_type == 'Measurement':
            features.append(node_data.get('value', 0))
        else:
            features.append(0)
        
        degree = self.kg.graph.degree(node_id)
        features.append(degree / 100.0)
        
        return features
    
    def _get_yield_for_timestamp(self, timestamp_node):
        """Extract yield value for a timestamp"""
        for neighbor in self.kg.graph.neighbors(timestamp_node):
            node_data = self.kg.graph.nodes[neighbor]
            if node_data.get('type') == 'Measurement':
                metric = node_data.get('metric', '')
                if 'target' in metric.lower():
                    return float(node_data.get('value', 0))
        return None
    
    def train(self, data, epochs=200, lr=0.01, weight_decay=5e-4):
        """Train the GNN model"""
        
        print("\nInitializing GNN model...")
        num_features = data.x.shape[1]
        self.model = YieldGNN(num_features, self.hidden_dim, self.num_layers)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        self.model = self.model.to(device)
        data = data.to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=lr, weight_decay=weight_decay)
        
        print(f"\nTraining for {epochs} epochs...")
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print("Training complete!")
    
    def evaluate(self, data):
        """Evaluate model on test set"""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            
            test_pred = out[data.test_mask].cpu().numpy()
            test_true = data.y[data.test_mask].cpu().numpy()
            
            test_pred_original = self.scaler_y.inverse_transform(
                test_pred.reshape(-1, 1)).flatten()
            test_true_original = self.scaler_y.inverse_transform(
                test_true.reshape(-1, 1)).flatten()
            
            test_node_indices = torch.where(data.test_mask)[0].cpu().numpy()
            test_dates = []
            test_plots = []
            
            for idx in test_node_indices:
                node_id = self.idx_to_node[idx]
                node_data = self.kg.graph.nodes[node_id]
                if 'datetime' in node_data:
                    test_dates.append(pd.to_datetime(node_data['datetime']))
                if 'field' in node_data:
                    field_id = node_data['field']
                    if 'Plot_' in field_id:
                        plot_num = field_id.split('Plot_')[1].split('_')[0]
                        test_plots.append(int(plot_num))
                    else:
                        test_plots.append(0)
            
            mse = mean_squared_error(test_true_original, test_pred_original)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_true_original, test_pred_original)
            r2 = r2_score(test_true_original, test_pred_original)
            
            print("\nOverall Test Set Performance:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
            
            results_df = pd.DataFrame({
                'actual': test_true_original,
                'predicted': test_pred_original
            })
            if len(test_dates) == len(test_true_original):
                results_df['date'] = test_dates
            if len(test_plots) == len(test_true_original):
                results_df['plot'] = test_plots
            
            import os
            os.makedirs('outputs/gnn', exist_ok=True)
            os.makedirs('outputs/gnn/individual_plots', exist_ok=True)
            results_df.to_csv('outputs/gnn/test_results.csv', index=False)
            print(f"\nResults saved to: outputs/gnn/test_results.csv")
            
            if 'plot' in results_df.columns:
                unique_plots = sorted(results_df['plot'].unique())
                print(f"\nCreating individual graphs for {len(unique_plots)} plots...")
                
                for plot_num in unique_plots:
                    plot_data = results_df[results_df['plot'] == plot_num].copy()
                    plot_data = plot_data.sort_values('date') if 'date' in plot_data.columns else plot_data
                    
                    plt.figure(figsize=(10, 6))
                    
                    if 'date' in plot_data.columns:
                        plt.plot(plot_data['date'], plot_data['actual'], 
                               label='Actual', marker='o', linewidth=2, markersize=6)
                        plt.plot(plot_data['date'], plot_data['predicted'], 
                               label='Predicted', marker='s', linewidth=2, markersize=6)
                        plt.xlabel('Date', fontsize=12)
                        plt.xticks(rotation=45, ha='right')
                    else:
                        plt.plot(plot_data['actual'], label='Actual', marker='o', linewidth=2)
                        plt.plot(plot_data['predicted'], label='Predicted', marker='s', linewidth=2)
                        plt.xlabel('Sample', fontsize=12)
                    
                    plot_rmse = np.sqrt(mean_squared_error(plot_data['actual'], plot_data['predicted']))
                    plot_r2 = r2_score(plot_data['actual'], plot_data['predicted'])
                    
                    plt.ylabel('Yield', fontsize=12)
                    plt.title(f'Plot {plot_num} - Predictions vs Actual\nRMSE: {plot_rmse:.2f}, R²: {plot_r2:.3f}', 
                             fontsize=14, fontweight='bold')
                    plt.legend(fontsize=11)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    plt.savefig(f'outputs/gnn/individual_plots/plot_{plot_num}.png', 
                               dpi=150, bbox_inches='tight')
                    plt.close()
                
                print(f"Individual plot files saved to: outputs/gnn/individual_plots/")
            
            return {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': test_pred_original,
                'actuals': test_true_original,
                'dates': test_dates
            }
    
    def save_model(self, filepath='outputs/gnn/gnn_model.pt'):
        """Save trained model"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.model is None:
            print("No model to save!")
            return
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'node_to_idx': self.node_to_idx,
            'idx_to_node': self.idx_to_node,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_features': self.model.convs[0].in_channels
        }
        
        torch.save(save_dict, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath='outputs/gnn/gnn_model.pt'):
        """Load trained model"""
        checkpoint = torch.load(filepath)
        
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        self.node_to_idx = checkpoint['node_to_idx']
        self.idx_to_node = checkpoint['idx_to_node']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_layers = checkpoint['num_layers']
        
        self.model = YieldGNN(
            checkpoint['num_features'],
            self.hidden_dim,
            self.num_layers
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from: {filepath}")


def run_gnn_prediction(kg, test_year=2023):
    """Run GNN-based yield prediction"""
    
    print("="*60)
    print("GNN-BASED YIELD PREDICTION (Transductive Learning)")
    print("="*60)
    print(f"\nGraph contains ALL data including {test_year}")
    print(f"But {test_year} labels are MASKED during training")
    
    predictor = GNNYieldPredictor(kg, hidden_dim=64, num_layers=3)
    graph_data = predictor.prepare_graph_data(test_year=test_year)
    predictor.train(graph_data, epochs=200, lr=0.01)
    results = predictor.evaluate(graph_data)
    
    import os
    os.makedirs('outputs/gnn', exist_ok=True)
    predictor.save_model('outputs/gnn/gnn_model.pt')
    
    print("\n" + "="*60)
    print("GNN PREDICTION COMPLETE")
    print("="*60)
    
    return predictor, results


if __name__ == "__main__":
    print("Call from main script:")
    print("  from gnn_predictor import run_gnn_prediction")
    print("  predictor, results = run_gnn_prediction(kg, test_year=2023)")