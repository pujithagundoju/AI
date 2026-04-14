import pickle
import os
import time
from typing import Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class ModelFactory:
    """
    Factory class to fetch different Scikit-learn classification models.
    """
    @staticmethod
    def get_model(model_name: str) -> Any:
        models = {
            'lr': LogisticRegression(max_iter=1000, random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': LinearSVC(random_state=42, dual='auto'),
            'nb': MultinomialNB()
        }
        if model_name.lower() not in models:
            raise ValueError(f"Model {model_name} not supported. Options: {list(models.keys())}")
        return models[model_name.lower()]

class ResumeClassifier:
    """
    Trainer and evaluator for resume classification.
    """
    def __init__(self, model_name: str = 'lr'):
        self.model_name = model_name
        self.model = ModelFactory.get_model(model_name)
        self.is_fitted = False
        
    def train(self, X_train, y_train) -> float:
        """
        Train the model and return the training time.
        """
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        
        self.is_fitted = True
        return end_time - start_time
        
    def predict(self, X_test) -> Tuple[Any, float]:
        """
        Predict labels and return predictions along with inference time.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        start_time = time.time()
        preds = self.model.predict(X_test)
        end_time = time.time()
        
        return preds, (end_time - start_time)
        
    def evaluate(self, y_test, preds) -> dict:
        """
        Evaluate and return metrics.
        """
        acc = accuracy_score(y_test, preds)
        
        # We use weighted average because job classes might be imbalanced
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
        
        return {
            "model": self.model_name,
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1
        }
        
    def save(self, filepath: str):
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
            self.is_fitted = True
