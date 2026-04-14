# 🧠 Models Layer (`src/models/`)

This directory is the core artificial intelligence engine. It houses every algorithm utilized to categorize the numerical data arrays.

### Key Files:
- `classifiers.py`: A unified factory architecture executing massive traditional ML models linearly (`Naive Bayes`, `KNN`, `Logistic Regression`, `Support Vector Machine`, `Random Forest`, and `Gradient Boosting`).
- `dl_models.py`: Defines the native PyTorch Neural Network architecture (`MLPClassifier`), comprising `nn.Linear` fully connected layers separated by `ReLU` non-linearities and Dropout regularization.
- `dl_trainer.py`: Encapsulates the Deep Learning training loops, executing backpropagation via `Adam` gradients and Cross Entropy loss functions.
- `bert_experiments.py`: A sandbox automation script utilized locally to test semantic deep learning (`Sentence-BERT`) on purely mathematical classification instead of vocabulary intersection.
- `MODEL_ANALYSIS_REPORT.md`: Our highly intensive, mathematically rigorous audit detailing exactly how and why every algorithm failed or succeeded geometrically.
