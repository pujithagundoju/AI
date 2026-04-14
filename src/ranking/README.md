# 🎯 Ranking Subsystem (`src/ranking/`)

This directory handles Information Retrieval (IR) tasks. Rather than definitively classifying a resume into a rigid category (like HR or Sales), this module *ranks* resumes dynamically against a continuous similarity distribution based on the provided Job Description.

### Key Files:
- `scorer.py`: Originally utilized Cosine Similarity overlaid on TF-IDF vectors to execute strict keyword exact-match analysis. Note: We have organically pushed past this constraint by integrating HuggingFace Transfomers (`all-MiniLM-L6-v2`) inside the `pipeline.py` evaluation script to capture deep conceptual and semantic equivalences rather than lexical string overlaps.
