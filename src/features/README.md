# 🧮 Features Layer (`src/features/`)

This module translates natural language into mathematical properties.

### Key Files:
- `tfidf.py`: A wrapper implementation around Scikit-Learn's `TfidfVectorizer`. 
  - Generates sparse positional arrays (shape arrays containing mostly zeros) representing keyword importance.
  - Custom engineered to extract `(1,3)` n-grams and restrict computational bleeding by capping at `8,000` maximum features. It also features persistence mechanisms (`.save()`, `.load()`) so vectorization logic isn't lost post-training.
