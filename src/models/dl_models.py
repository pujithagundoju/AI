import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    """
    Simple Feed-Forward Neural Network for text classification.
    Expects TF-IDF or BoW vectors as input.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.3):
        super(MLPClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)
