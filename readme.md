# RP-PCA for BTC Market Signal Classification

## Overview

This repository explores the application of Random Projection + Principal Component Analysis (RP-PCA) for reduction of noisy financial time-series data.

The objective is to investigate whether projecting high-dimensional technical features into a lower-dimensional latent space can:

- Improve predictive stability
- Reduce noise when multiple variables are in play
- Capture major structural drivers for better prediction and analysis
- Provide anomaly detection via reconstruction error
- Improve ensemble performance

This is experimental implementation. Live trading integrations are intentionally excluded.


## Motivation

Financial market features (technical indicators, volatility metrics, sentiment signals) are often:

- Extremely correlated
- Very Noisy
- Very High dimensional

Standard PCA and any machine learning methodology can struggle in very noisy, high dimensional settings. 

To address this and mitigate signal noise, this project applies:

1. Robust scaling
2. Gaussian random projection
3. PCA in projected space
4. Reconstruction error as anomaly signal

This pipeline aims to stabilize latent factor extraction while preserving dominant variance structure.


## Architecture within the 4 files within the repo

The pipeline consists of:

- Feature matrix input (synthetic BTC-like data for demo)
- RP-PCA transformation:
  - RobustScaler
  - GaussianRandomProjection
  - PCA
  - RP reconstruction error
- Feature augmentation with RP principal components
- Soft-voting ensemble classifier:
  - Random Forest
  - LightGBM
  - CatBoost
- Time-aware train/test split (no shuffle)
- Model diagnostics:
  - Rolling accuracy
  - Confusion matrix
  - Precision / Recall / F1 per class
  - Calibration


## RP-PCA Details

Let X be the feature matrix.

Step 1:
X_scaled = RobustScaler(X)

Step 2:
Z = GaussianRandomProjection(X_scaled)

Step 3:
Z_pca = PCA(Z)

Step 4:
Reconstruction error:
|| Z - Z_hat ||

The reconstruction norm is used as an anomaly measure.

Additionally, approximate mappings to original feature space allow identification of top drivers per principal component.


## Running the Demo

Install dependencies: 

pip install -r requirements.txt (Mentioned in text file)

Run:

python train_rppca_demo.py

Some Outputs will be generated in the folder 'outputs' as 'qa':

outputs/qa/

- rolling_accuracy.png
- confusion_matrix.png
- per_class_metrics.csv
- reliability plots


## Notes

- The dataset used here is synthetic for reproducibility.
- The full live trading system and API integrations are intentionally not included.
- This repository focuses strictly on dimensionality reduction methodology and model evaluation.
- Apologies in advance for discrepancies and errors in architecture or methodology 



## Possible Integrations

- Apply RP-PCA to real MT5 data
- Integrate with macro-economic latent factor research (Possibilities exist for capital allocation decisions)