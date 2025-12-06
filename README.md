# Aspect-Based Sentiment Analysis on the FABSA Dataset

This project performs Aspect-Based Sentiment Analysis (ABSA) on the FABSA dataset. 
This repository includes:
1. Data exploration and visualization  
2. RoBERTa-based ABSA modeling with class imbalance handling  
3. SHAP-based interpretability for aspect-level insights  

The project provides both predictive performance and insights into how language influences sentiment across aspects and industries.
## Project Files

### `nlp_final_proj_analysis_viz.py`
Exploratory data analysis and visualization.  
Includes dataset inspection, aspect/sentiment distribution, industry-level comparisons, text statistics, and flattening of multi-aspect reviews.

### `nlp_final_proj.py`
Main modeling pipeline.  
Loads the flattened dataset, builds DataLoaders, applies class-weighted loss for imbalance, fine-tunes RoBERTa for 3-class sentiment classification, evaluates performance, and saves the model and tokenizer.

### `shap_explain.py`
SHAP explainability module.  
Generates token-level SHAP values for individual samples and aggregated aspect-level explanations to understand which words influence negative, neutral, or positive predictions.

## Running the Scripts

Run EDA:
python3 nlp_final_proj_analysis_viz.py

Train the model:
python3 nlp_final_proj.py

Generate SHAP explanations:
python shap_explain.py

## Requirements
Key libraries:
torch
transformers
datasets
numpy
pandas
scikit-learn
matplotlib
seaborn
shap
tqdm




