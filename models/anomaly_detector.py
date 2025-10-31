import torch
import torch.nn as nn


class AnomalyDetector(nn.Module):
    """
    3-layer fully connected network for anomaly detection.
    
    Architecture from paper:
    - FC1: 4096 -> 512 (ReLU + Dropout 0.6)
    - FC2: 512 -> 32 (ReLU + Dropout 0.6)
    - FC3: 32 -> 1 (Sigmoid)
    """
    
    def __init__(self, input_dim=4096, dropout=0.6):
        super(AnomalyDetector, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_segments, feature_dim)
               e.g., (30, 32, 4096)
        
        Returns:
            scores: (batch_size, num_segments)
                    Anomaly score for each segment (0~1)
        """
        batch_size, num_segments, feature_dim = x.shape
        
        # Reshape to process all segments at once
        x = x.view(-1, feature_dim)  # (batch_size * num_segments, 4096)
        
        # FC layers
        x = self.fc1(x)           # (batch_size * num_segments, 512)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)           # (batch_size * num_segments, 32)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)           # (batch_size * num_segments, 1)
        x = self.sigmoid(x)
        
        # Reshape back
        scores = x.view(batch_size, num_segments)  # (batch_size, 32)
        
        return scores