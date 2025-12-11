Evaluation instructions for ScadaIntrusion detection algorithms

What this does
- `evaluate.py` trains an unsupervised Gaussian Mixture baseline (and attempts a GMM-HMM if `hmmlearn` is installed), scores attack datasets, and prints evaluation metrics (AUC, accuracy, precision, recall, F1).

Prerequisites
- Python 3.x (tested on 3.8+)
- Install dependencies from `requirements.txt` in this folder. Example (PowerShell):

```powershell
python -m pip install -r requirements.txt
```

How to run
- From the `scada-intrusion-detection` folder run:

```powershell
python evaluate.py
```

Notes
- The script uses a simple heuristic threshold (5th percentile of training scores) to convert log-likelihoods into anomaly labels for computing Accuracy/Precision/Recall/F1. To get more robust evaluation, sweep thresholds and compute ROC/PR curves.
- `hmmlearn` installation may require a C compiler on Windows. If `hmmlearn` cannot be installed, the script will still run GMM baseline.
- This is a starting point — I can expand it to save outputs, plot ROC curves, and produce CSV metric tables per attack type.

The repository now contains `detectors.py` with additional detectors you can evaluate:

- `RandomForestDetector` (supervised)
- `XGBoostDetector` (supervised, requires `xgboost`)
- `TCNAutoencoder` (unsupervised sequence autoencoder, requires TensorFlow/Keras)
- `VAETracer` (variational autoencoder, requires TensorFlow/Keras)
- `DeepSVDDDetector` (uses encoder latent distances as a simple SVDD approximation; requires TensorFlow/Keras)

Notes on heavy dependencies

- `tensorflow` and `xgboost` can be large and may require build tools on Windows. If you want a minimal evaluation run, install only the core deps first (`numpy`, `scipy`, `scikit-learn`, `hmmlearn`, `statsmodels`).

If you'd like, I can:

- integrate the new detectors into `evaluate.py` to run them and produce per-method CSV/markdown tables, or
- add a separate `evaluate_deep.py` that focuses on deep models and generates trained model artifacts.
