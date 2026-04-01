import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath='processed_subreddit_weekly_enriched.csv'):
    print("Loading data...")
    df = pd.read_csv(filepath)
    df['year_week'] = pd.to_datetime(df['year_week'])
    df = df.sort_values(by=['subreddit', 'year_week']).reset_index(drop=True)
    
    # 1. Target Variable
    print("Computing target variable...")
    df['engagement'] = df['total_comments'] + df['post_count']
    y_labels = []
    for sub in df['subreddit'].unique():
        sub_data = df[df['subreddit'] == sub].copy()
        historical_peak = sub_data['engagement'].expanding().max()
        rolling_avg = sub_data['engagement'].rolling(window=4, min_periods=1).mean()
        collapsed_mask = (rolling_avg < (0.30 * historical_peak)) & (historical_peak > 50)
        sub_data['is_collapsed'] = collapsed_mask.astype(int)
        y_labels.append(sub_data)
    
    df = pd.concat(y_labels)
    
    # 2. Lag Features
    print("Engineering lag features...")
    feature_cols = ['post_count', 'total_comments', 'avg_score', 'avg_upvote_ratio',
                    'total_awards', 'total_crossposts', 'avg_subscribers']
    
    for col in feature_cols:
        for lag in [1, 2, 3, 4]:
            df[f'{col}_lag{lag}'] = df.groupby('subreddit')[col].shift(lag)
            
    df = df.dropna().copy()
    
    # Create complete wide index N nodes by T weeks
    unique_subs = sorted(df['subreddit'].unique())
    sub_to_idx = {sub: i for i, sub in enumerate(unique_subs)}
    
    unique_weeks = sorted(df['year_week'].unique())
    
    return df, sub_to_idx, unique_weeks

def build_graph(df, sub_to_idx):
    print("Building adjacency matrix based on engagement correlation...")
    # Pivot to get time series of engagement per subreddit
    pivot_df = df.pivot(index='year_week', columns='subreddit', values='engagement').fillna(0)
    
    # Calculate Pearson correlation matrix
    corr_matrix = pivot_df.corr().fillna(0)
    
    # Threshold for edges
    threshold = 0.5
    edge_index = []
    unique_subs = sorted(list(sub_to_idx.keys()))
    
    for i in range(len(unique_subs)):
        for j in range(i + 1, len(unique_subs)):
            sub_i = unique_subs[i]
            sub_j = unique_subs[j]
            if sub_i in corr_matrix.columns and sub_j in corr_matrix.columns:
                corr = corr_matrix.loc[sub_i, sub_j]
                if corr > threshold:
                    idx_i = sub_to_idx[sub_i]
                    idx_j = sub_to_idx[sub_j]
                    edge_index.append([idx_i, idx_j])
                    edge_index.append([idx_j, idx_i]) # Undirected
                    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    print(f"Graph constructed with {len(unique_subs)} nodes and {edge_index.shape[1] // 2} undirected edges.")
    return edge_index, corr_matrix

def prepare_data(df, sub_to_idx, unique_weeks, edge_index):
    print("Formatting data for PyTorch Geometric...")
    predictors = [col for col in df.columns if 'lag' in col]
    
    # We need to scale features
    scaler = StandardScaler()
    df[predictors] = scaler.fit_transform(df[predictors])
    
    data_list = []
    
    for week in unique_weeks:
        week_df = df[df['year_week'] == week]
        if len(week_df) == 0:
            continue
            
        num_nodes = len(sub_to_idx)
        num_features = len(predictors)
        x = torch.zeros((num_nodes, num_features), dtype=torch.float)
        y = torch.zeros(num_nodes, dtype=torch.long)
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        for _, row in week_df.iterrows():
            idx = sub_to_idx[row['subreddit']]
            x[idx] = torch.tensor(row[predictors].values.astype(np.float32), dtype=torch.float)
            y[idx] = int(row['is_collapsed'])
            mask[idx] = True
            
        data = Data(x=x, edge_index=edge_index, y=y)
        data.mask = mask
        data.week = week
        data_list.append(data)
        
    # Split chronologically
    split_idx = int(len(data_list) * 0.8)
    train_data = data_list[:split_idx]
    test_data = data_list[split_idx:]
    
    print(f"Prepared {len(train_data)} training weeks and {len(test_data)} test weeks.")
    return train_data, test_data

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train_model(train_data, test_data, num_features):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(in_channels=num_features, hidden_channels=32, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Class weights since collapsed is rare
    all_y = []
    for data in train_data:
        all_y.extend(data.y[data.mask].tolist())
    all_y = np.array(all_y)
    class_counts = np.bincount(all_y)
    weights = class_counts.sum() / (2.0 * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    print("Training GCN Model...")
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    final_test_preds = []
    final_test_true = []
    
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for data in train_data:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.mask], data.y[data.mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        val_loss = 0
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for data in test_data:
                data = data.to(device)
                out = model(data.x, data.edge_index)
                loss = criterion(out[data.mask], data.y[data.mask])
                val_loss += loss.item()
                preds = out[data.mask].argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(data.y[data.mask].cpu().numpy())
                
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            final_test_preds = all_preds
            final_test_true = all_true
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Train Loss: {total_loss/len(train_data):.4f}, Val Loss: {val_loss/len(test_data):.4f}")
            
    print("Training Completed.\n")
    print("Classification Report on Test Set:")
    target_names = ['Healthy (0)', 'Collapsed (1)']
    print(classification_report(final_test_true, final_test_preds, target_names=target_names, zero_division=0))

if __name__ == '__main__':
    df, sub_to_idx, unique_weeks = load_and_preprocess_data()
    edge_index, _ = build_graph(df, sub_to_idx)
    train_data, test_data = prepare_data(df, sub_to_idx, unique_weeks, edge_index)
    num_features = train_data[0].x.shape[1]
    train_model(train_data, test_data, num_features)
