# Face Embeddings Gender Prediction

This project explores the use of face recognition embeddings to predict gender from facial images. The analysis uses a dataset of 92 face images, each represented by 128-dimensional embeddings.

## Dataset

The dataset (`faces_embeddings.csv`) contains:
- 92 rows (images)
- 128 feature columns (X0-X127) representing face embeddings
- Gender labels (-1 for male, +1 for female)

## Project Structure

- `gender_prediction.py`: Initial implementation with basic gender prediction
- `gender_prediction_extended.py`: Enhanced version with regularization analysis and visualizations
- `requirements.txt`: Python package dependencies
- Generated visualizations:
  - `regularization_performance.png`: Model performance vs regularization strength
  - `feature_sparsity.png`: Feature sparsity analysis
  - `feature_importance_heatmap.png`: Feature importance patterns

## Results

The model achieves excellent performance:
- Perfect accuracy (100%) with optimal regularization (C=1.0)
- Identifies key features for gender prediction
- Robust performance across k-fold cross-validation

## Setup and Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the analysis:
```bash
python3 gender_prediction_extended.py
```

## Methodology

- Uses L1 regularization (Lasso) for feature selection
- Implements 5-fold cross-validation for robust error estimation
- Analyzes feature importance and regularization effects
- Visualizes results and model behavior 