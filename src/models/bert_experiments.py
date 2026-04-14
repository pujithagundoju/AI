import pandas as pd
from src.data.loader import load_data
from src.data.cleaner import TextCleaner
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sentence_transformers import SentenceTransformer
import time

def run():
    print("Loading data...")
    train_df, test_df = load_data('data/raw')
    
    # We can actually just use the raw text for Sentence BERT, it works better without removing stops!
    # But let's just use raw text for maximum semantic power
    X_train_raw = train_df['text'].tolist()
    y_train_raw = train_df['label'].tolist()
    X_test_raw = test_df['text'].tolist()
    y_test_raw = test_df['label'].tolist()
    
    print("Loading BERT Embedder...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding Train...")
    X_train = embedder.encode(X_train_raw, show_progress_bar=True)
    print("Encoding Test...")
    X_test = embedder.encode(X_test_raw, show_progress_bar=True)
    
    models = {
        'lr': LogisticRegression(max_iter=2000, random_state=42),
        'svm': LinearSVC(random_state=42, dual='auto'),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train_raw)
        t_time = time.time() - start
        
        preds = model.predict(X_test)
        
        acc = accuracy_score(y_test_raw, preds)
        p, r, f1, _ = precision_recall_fscore_support(y_test_raw, preds, average='weighted', zero_division=0)
        
        results.append({
            "model": name.upper(),
            "acc": acc,
            "f1": f1,
            "time": t_time
        })
        
    df = pd.DataFrame(results)
    print("\n=== Sentence BERT Embedding Classification Metrics ===")
    print(df.to_string(index=False))

if __name__ == '__main__':
    run()
