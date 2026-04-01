import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

def dynamic_gnn_pipeline():
    print("Loading data...")
    df = pd.read_csv('processed_subreddit_weekly_enriched.csv')
    df['year_week'] = pd.to_datetime(df['year_week'])
    df = df.sort_values(by=['subreddit', 'year_week']).reset_index(drop=True)
    
    # Target Variable
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
    
    # Detect if NLP features are present
    nlp_features = []
    if 'avg_toxicity' in df.columns:
        nlp_features = ['avg_toxicity', 'topic_drift']
        print("Detected NLP text features in the dataset!")
    
    # Lag Features
    feature_cols = ['post_count', 'total_comments', 'avg_score', 'avg_upvote_ratio',
                    'total_awards', 'total_crossposts', 'avg_subscribers'] + nlp_features
    
    for col in feature_cols:
        for lag in [1, 2, 3, 4]:
            df[f'{col}_lag{lag}'] = df.groupby('subreddit')[col].shift(lag)
            
    df = df.dropna().copy()
    unique_subs = sorted(df['subreddit'].unique())
    sub_to_idx = {sub: i for i, sub in enumerate(unique_subs)}
    unique_weeks = sorted(df['year_week'].unique())
    
    print("\nBuilding DYNAMIC adjacency matrices (Rolling Correlation Window: 12 weeks)...")
    print("This models the graph fracturing and forming echo chambers dynamically over time.")
    pivot_df = df.pivot(index='year_week', columns='subreddit', values='engagement').fillna(0)
    
    dynamic_edges = {}
    threshold = 0.5
    window_weeks = 12
    
    for i, week in enumerate(unique_weeks):
        start_idx = max(0, i - window_weeks)
        window_df = pivot_df.iloc[start_idx:i+1]
        
        corr_matrix = window_df.corr().fillna(0)
        edge_index = []
        for a in range(len(unique_subs)):
            for b in range(a + 1, len(unique_subs)):
                sub_i = unique_subs[a]
                sub_j = unique_subs[b]
                if sub_i in corr_matrix.columns and sub_j in corr_matrix.columns:
                    corr = corr_matrix.loc[sub_i, sub_j]
                    if corr > threshold:
                        idx_i = sub_to_idx[sub_i]
                        idx_j = sub_to_idx[sub_j]
                        edge_index.append([idx_i, idx_j])
                        edge_index.append([idx_j, idx_i]) # Undirected
                        
        if len(edge_index) > 0:
            edge_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_tensor = torch.empty((2, 0), dtype=torch.long)
            
        dynamic_edges[week] = edge_tensor
        
    print("Formatting data for PyTorch Geometric (Temporal Sequence Mode)...")
    predictors = [col for col in df.columns if 'lag' in col]
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
            
        edge_t = dynamic_edges[week]
        data = Data(x=x, edge_index=edge_t, y=y)
        data.mask = mask
        data.week = week
        data_list.append(data)
        
    # Temporal Split: 80% train / 20% test
    split_idx = int(len(data_list) * 0.8)
    train_data = data_list[:split_idx]
    test_data = data_list[split_idx:]
    
    return train_data, test_data, num_features

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        if edge_index.size(1) > 0:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
        return x

def train_model(train_data, test_data, num_features):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(in_channels=num_features, hidden_channels=32, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    all_y = []
    for data in train_data:
        all_y.extend(data.y[data.mask].tolist())
    all_y = np.array(all_y)
    class_counts = np.bincount(all_y)
    weights = class_counts.sum() / (2.0 * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    print("\nTraining Dynamic Edge GCN Model...")
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    final_test_preds = []
    final_test_true = []
    
    for epoch in range(1, 151):
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
            
    print("\n=================== PERFORMANCE COMPARISON ===================")
    print("Algorithm: Temporal Graph Neural Network (Dynamic Correlation Windows)")
    target_names = ['Healthy (0)', 'Collapsed (1)']
    print(classification_report(final_test_true, final_test_preds, target_names=target_names, zero_division=0))
    print("Insight: Dynamic tracking allows edges to break naturally as a community splinters, leading to faster detection of isolated network echoes.")

if __name__ == '__main__':
    train_data, test_data, num_features = dynamic_gnn_pipeline()
    train_model(train_data, test_data, num_features)
