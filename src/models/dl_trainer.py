import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import scipy.sparse as sp

class DLTrainer:
    """
    Trainer for PyTorch models.
    """
    def __init__(self, model, lr=1e-3, epochs=10, batch_size=64, device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
    def _prepare_dataloader(self, X, y=None, shuffle=True):
        if sp.issparse(X):
            X_tensor = torch.FloatTensor(X.toarray())
        else:
            X_tensor = torch.FloatTensor(X)
            
        if y is not None:
            y_tensor = torch.LongTensor(y)
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)
            
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def train(self, X_train, y_train):
        loader = self._prepare_dataloader(X_train, y_train, shuffle=True)
        self.model.train()
        
        start_time = time.time()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            # print(f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss/len(loader):.4f}")
            
        end_time = time.time()
        return end_time - start_time
        
    def predict(self, X_test):
        loader = self._prepare_dataloader(X_test, shuffle=False)
        self.model.eval()
        
        all_preds = []
        start_time = time.time()
        
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                
        end_time = time.time()
        return np.array(all_preds), (end_time - start_time)
        
    def evaluate(self, y_test, preds):
        acc = accuracy_score(y_test, preds)
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
        
        return {
            "model": "MLP (PyTorch)",
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1
        }
