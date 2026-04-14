from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from typing import List, Tuple

class ResumeRanker:
    """
    Ranks resumes against a given Job Description (JD).
    """
    def __init__(self, vectorizer):
        """
        Expects a fitted vectorizer (TFIDFExtractor) capable of transforming texts.
        """
        self.vectorizer = vectorizer
        
    def rank(self, jd_text: str, candidate_texts: List[str], candidate_labels: List[str], top_n: int = 10) -> pd.DataFrame:
        """
        Vectorize JD and Candidates, and rank candidates based on Cosine Similarity.
        
        Args:
            jd_text (str): Raw string of the job description.
            candidate_texts (List[str]): List of raw resume texts.
            candidate_labels (List[str]): Corresponding original candidate labels/IDs.
            top_n (int): Number of top resumes to return.
            
        Returns:
            pd.DataFrame: Ranked resumes with their similarity scores.
        """
        # Vectorize JD and candidates
        jd_vector = self.vectorizer.transform([jd_text])
        candidate_vectors = self.vectorizer.transform(candidate_texts)
        
        # Calculate cosine similarity between JD and all candidates
        # cosine_similarity output shape is (1, num_candidates)
        similarities = cosine_similarity(jd_vector, candidate_vectors).flatten()
        
        # Get ranks (sort indices descending)
        ranked_indices = np.argsort(similarities)[::-1]
        
        ranked_data = []
        for idx in ranked_indices[:top_n]:
            ranked_data.append({
                "CandidateClass": candidate_labels[idx],
                "Snippet": candidate_texts[idx][:150] + "...",
                "SimilarityScore": similarities[idx]
            })
            
        return pd.DataFrame(ranked_data)
