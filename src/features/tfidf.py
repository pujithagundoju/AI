from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from typing import Tuple
from scipy.sparse import csr_matrix

class TFIDFExtractor:
    """
    Wrapper for Scikit-learn's TfidfVectorizer.
    Handles fitting, transforming, and saving/loading the vectorizer state.
    """
    def __init__(self, max_features: int = 8000, n_gram_range: Tuple[int, int] = (1, 3)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=n_gram_range,
            sublinear_tf=True
        )
        self.is_fitted = False
        
    def fit_transform(self, texts: list[str]) -> csr_matrix:
        """
        Fit the vectorizer on the texts and return the transformed matrix.
        """
        X = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return X
        
    def transform(self, texts: list[str]) -> csr_matrix:
        """
        Transform texts using the already fitted vectorizer.
        """
        if not self.is_fitted:
            raise ValueError("The vectorizer is not fitted yet. Call fit_transform first.")
        return self.vectorizer.transform(texts)
        
    def save(self, filepath: str = "src/features/tfidf_vectorizer.pkl"):
        """
        Serialize and save the vectorizer model.
        """
        if not self.is_fitted:
            raise ValueError("The vectorizer is not fitted yet.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            
    def load(self, filepath: str = "src/features/tfidf_vectorizer.pkl"):
        """
        Load the vectorizer model from file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved vectorizer found at {filepath}")
            
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
            self.is_fitted = True
