import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

class TradingSequenceDataset(Dataset):
    """Converts stationary features and labels into chronological sliding sequence tensors."""
    def __init__(self, csv_path: str, window_size: int = 30):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing labeled dataset at {csv_path}. Run label_targets.py first.")
            
        df = pd.read_csv(csv_path)
        
        feature_cols = [
            c for c in df.columns 
            if df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
            and not c.startswith('env_') and c != 'target' and c != 'unnamed: 0'
        ]
        
        raw_features = df[feature_cols].values.astype(np.float32)
        
        # --- CRITICAL FIX 1: Z-Score Normalization ---
        # LSTMs go blind if inputs are not standardized (mean=0, std=1)
        mean = np.mean(raw_features, axis=0)
        std = np.std(raw_features, axis=0) + 1e-8
        self.features = (raw_features - mean) / std
        
        self.labels = df['target'].values.astype(np.int64)
        self.window_size = window_size
        
        # Calculate Class Weights for the Loss Function
        class_counts = np.bincount(self.labels)
        total_samples = len(self.labels)
        self.class_weights = total_samples / (len(class_counts) * class_counts)

    def __len__(self):
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]
        y = self.labels[idx + self.window_size]
        return torch.tensor(x), torch.tensor(y)

class OracleLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 3):
        super(OracleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        logits = self.fc(out[:, -1, :])
        return logits

def train_oracle(csv_path: str, epochs: int = 20, batch_size: int = 128, lr: float = 0.001):
    print("📦 Loading sequence dataset into tensors...")
    dataset = TradingSequenceDataset(csv_path, window_size=30)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    sample_x, _ = dataset[0]
    input_dim = sample_x.shape[1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Computing initialized on system node: [{device}]")
    
    model = OracleLSTM(input_dim=input_dim).to(device)
    
    # --- CRITICAL FIX 2: Weighted Cross Entropy ---
    weights_tensor = torch.FloatTensor(dataset.class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    # Adding Weight Decay (L2 Regularization) to prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    print(f"🏃 Beginning supervised network optimization for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        accuracy = (correct / total) * 100
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Accuracy: {accuracy:.2f}%")
        
    model_save_path = "./oracle_lstm.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"🏆 Oracle brain weights successfully saved to {model_save_path}")
    return model_save_path

if __name__ == "__main__":
    # Increased to 20 epochs to give the network time to map the features
    train_oracle('../data/processed/labeled_features_15m.csv', epochs=20)