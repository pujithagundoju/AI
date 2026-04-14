# 🔮 Evaluation Layer (`src/evaluation/`)

This directory contains the central orchestrator that wires every discrete model, dataset, and analyzer together.

### Key Files:
- `pipeline.py`: The master execution script. It linearly invokes:
  1. Data Loaders & NLTK Cleaners
  2. TF-IDF Extractors
  3. Every single Traditional ML baseline.
  4. The Deep Learning PyTorch Trainer.
  5. The HuggingFace `Sentence-BERT` JD Ranker.
  It computes execution times organically and structures a highly descriptive Pandas comparison table summarizing F1, Recall, Precision, and Accuracy metrics natively to the terminal output.
