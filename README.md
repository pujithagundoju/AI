# Smart Resume Screening System

An enterprise-grade applicant tracking and resume screening pipeline. It utilizes traditional machine learning algorithms alongside modern deep learning semantic transformers (Sentence-BERT) to autonomously categorize resumes and rank them semantically against technical Job Descriptions.

## 📁 Repository Structure

```text
├── data/
│   └── raw/                   # Contains Train.csv and Test.csv dataset
├── src/
│   ├── data/                  # Data ingestion and regex/NLTK text cleaners
│   ├── evaluation/            # Master pipeline runner
│   ├── features/              # TF-IDF sparse matrix extraction mapping
│   ├── models/                # Algorithms (SVM, RF, GradientBoosting, PyTorch)
│   └── ranking/               # Information Retrieval and JD similarity module
├── tests/                     # Automated unit testing suite
├── requirements.txt           # Environment dependencies
└── README.md                  # System overview
```
*(Detailed `README.md` files are located inside each sub-directory explaining its specific architecture)*

## ⚙️ Setup Instructions

1. **Initialize Environment**
Make sure you have Python installed. Create and activate a local virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

2. **Install Dependencies**
Install all required mathematical libraries and transformer architectures:
```powershell
pip install -r requirements.txt
pip install sentence-transformers
```

## 🚀 How to Execute Everything

To run the complete end-to-end pipeline (Loading -> Cleaning -> Feature Extraction -> Modeling -> JD Semantic Ranking), execute the master evaluation script from the root directory:

```powershell
.\venv\Scripts\python -m src.evaluation.pipeline
```
*Depending on your hardware, compiling the Gradient Boosting model over 8000 dimensions and downloading the HuggingFace MiniLM may take a few minutes.*
