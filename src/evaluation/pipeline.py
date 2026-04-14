import sys
import os
import copy
import pandas as pd
from tqdm import tqdm
import time

# Adjust import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.loader import load_data
from src.data.cleaner import TextCleaner
from src.features.tfidf import TFIDFExtractor
from src.models.classifiers import ResumeClassifier
from src.models.dl_models import MLPClassifier
from src.models.dl_trainer import DLTrainer
from src.ranking.scorer import ResumeRanker

def main():
    print("=== Smart Resume Screening Pipeline ===")
    
    # 1. Load Data
    print("\n1. Data Loading...")
    train_df, test_df = load_data('data/raw')
    print(f"Loaded {len(train_df)} training and {len(test_df)} testing rows.")
    
    # 2. Text Cleaning
    print("\n2. Text Cleaning & Preprocessing (This might take a moment)...")
    cleaner = TextCleaner()
    # It's better to use tqdm for apply to show progress
    tqdm.pandas(desc="Cleaning Training Data")
    train_df['Cleaned_Text'] = train_df['text'].progress_apply(cleaner.clean_text)
    
    tqdm.pandas(desc="Cleaning Testing Data")
    test_df['Cleaned_Text'] = test_df['text'].progress_apply(cleaner.clean_text)
    
    # Extract Texts and Labels
    X_train_raw = train_df['Cleaned_Text'].tolist()
    y_train_raw = train_df['label'].tolist()
    X_test_raw = test_df['Cleaned_Text'].tolist()
    y_test_raw = test_df['label'].tolist()
    
    # Convert labels to ints implicitly via pandas categorical or manual if needed
    labels = sorted(list(set(y_train_raw)))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_train = [label_to_idx[l] for l in y_train_raw]
    y_test = [label_to_idx[l] for l in y_test_raw]
    
    # 3. Feature Extraction
    print("\n3. Feature Extraction (TF-IDF)...")
    tfidf = TFIDFExtractor(max_features=8000, n_gram_range=(1,3))
    X_train = tfidf.fit_transform(X_train_raw)
    X_test = tfidf.transform(X_test_raw)
    print(f"Extracted TF-IDF dense shape: {X_train.shape}")
    
    # Ensure vectorizer is saved
    tfidf.save()
    
    # 4. Modeling (Traditional ML)
    print("\n4. Modeling...")
    results = []
    models_to_run = ['nb', 'knn', 'lr', 'svm', 'rf', 'gb']
    
    for m in models_to_run:
        print(f" -> Training {m.upper()}...")
        clf = ResumeClassifier(m)
        t_time = clf.train(X_train, y_train)
        preds, i_time = clf.predict(X_test)
        metrics = clf.evaluate(y_test, preds)
        metrics['Compute Time (Train)'] = round(t_time, 4)
        metrics['Inference Time'] = round(i_time, 4)
        results.append(metrics)
        print(f"    Done {m.upper()}: Acc={metrics['accuracy']:.4f}, Train={t_time:.2f}s")
        
    # 5. Modeling (Deep Learning - PyTorch MLP)
    try:
        import torch
        print("\n -> Training PyTorch MLP...")
        # Since it's TF-IDF, input dim = X_train.shape[1]
        mlp = MLPClassifier(input_dim=X_train.shape[1], num_classes=len(labels))
        trainer = DLTrainer(mlp, epochs=45, batch_size=32, lr=5e-4)
        
        t_time = trainer.train(X_train, y_train)
        preds, i_time = trainer.predict(X_test)
        metrics = trainer.evaluate(y_test, preds)
        metrics['Compute Time (Train)'] = round(t_time, 4)
        metrics['Inference Time'] = round(i_time, 4)
        results.append(metrics)
        print(f"    Done DL-MLP: Acc={metrics['accuracy']:.4f}, Train={t_time:.2f}s")
    except ImportError:
        print(" -> Skipping DL Training (PyTorch not found or torch module errored)")
        
        
    # 6. Formatting Comparison Report
    print("\n=== Model Comparisons ===")
    report_df = pd.DataFrame(results)
    print(report_df.to_string(index=False))
    
    # 7. Semantic JD Matching Demo
    print("\n=== Semantic JD Matching / Ranking System ===")
    sample_jd = "Looking for a seasoned Data Scientist with strong experience in Python, Machine Learning, Deep Learning, SQL, and natural language processing. PyTorch and Scikit-Learn required."
    
    try:
        from sentence_transformers import SentenceTransformer, util
        print(" -> Loading MiniLM Transformer (Downloading if first time)...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Raw text for semantic embedding!
        jd_emb = embedder.encode(sample_jd, convert_to_tensor=True)
        # We'll rank against the first 500 Test resumes
        doc_embs = embedder.encode(test_df['text'].tolist(), convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(jd_emb, doc_embs)[0]
        top_results = torch.topk(cosine_scores, k=5)
        
        ranked_data = []
        for score, idx in zip(top_results[0], top_results[1]):
            # test_df labels and texts
            l = test_df['label'].iloc[idx.item()]
            s = test_df['text'].iloc[idx.item()][:150] + "..."
            ranked_data.append({"CandidateClass": l, "Snippet": s, "SemanticScore": round(float(score), 4)})
            
        top_candidates = pd.DataFrame(ranked_data)
        print(f"Top 5 Semantically Matched Candidates for JD: 'Data Scientist'\n")
        print(top_candidates.to_string())
    except ImportError:
        print(" sentence-transformers not found! Skipped Semantic Matching.")


if __name__ == "__main__":
    main()
