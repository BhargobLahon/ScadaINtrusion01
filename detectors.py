"""
Detector wrappers for ScadaIntrusion evaluation

Includes:
- RandomForestDetector (supervised)
- XGBoostDetector (supervised, optional)
- TCNAutoencoder (unsupervised sequence autoencoder using Keras)
- VAE (unsupervised variational autoencoder)
- DeepSVDDDetector (uses encoder latent distances as anomaly score)

All deep models are optional and require TensorFlow/Keras installed. XGBoost is optional.
This module is intended as a lightweight, self-contained set of detectors that integrate
with the existing `evaluate.py` pipeline. Models are defensive when imports are missing.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, losses, optimizers
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


def create_windows(series, window_size=60, step=60):
    """Create sliding windows from a 2D time-series array.

    series: array-like shape (n_samples, n_features)
    Returns: array shape (n_windows, window_size, n_features)
    """
    arr = np.array(series)
    n = arr.shape[0]
    windows = []
    i = 0
    while i + window_size <= n:
        windows.append(arr[i:i+window_size])
        i += step
    if i < n and (n - i) > 0:
        # last partial window padded with last value
        last = arr[i:]
        pad_len = window_size - last.shape[0]
        pad = np.repeat(last[-1][np.newaxis, :], pad_len, axis=0)
        windows.append(np.vstack([last, pad]))
    return np.array(windows)


class RandomForestDetector:
    """Supervised Random Forest classifier used as anomaly detector.

    Usage:
    - Fit with flattened windows X (n_windows, window_size * n_features) and labels y (0 normal, 1 anomaly).
    - Score returns probability of anomaly for each window.
    """
    def __init__(self, **kwargs):
        self.clf = RandomForestClassifier(**kwargs)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X_windows, y):
        X = np.reshape(X_windows, (X_windows.shape[0], -1))
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        self.is_fitted = True

    def score(self, X_windows):
        if not self.is_fitted:
            raise RuntimeError('Model not fitted')
        X = np.reshape(X_windows, (X_windows.shape[0], -1))
        Xs = self.scaler.transform(X)
        # return probability of positive (anomaly) class if available
        if hasattr(self.clf, 'predict_proba'):
            return self.clf.predict_proba(Xs)[:, 1]
        else:
            return self.clf.predict(Xs)


class XGBoostDetector:
    def __init__(self, **kwargs):
        if not XGBOOST_AVAILABLE:
            raise RuntimeError('xgboost not available')
        self.model = xgb.XGBClassifier(**kwargs)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X_windows, y):
        X = np.reshape(X_windows, (X_windows.shape[0], -1))
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        self.is_fitted = True

    def score(self, X_windows):
        if not self.is_fitted:
            raise RuntimeError('Model not fitted')
        X = np.reshape(X_windows, (X_windows.shape[0], -1))
        Xs = self.scaler.transform(X)
        return self.model.predict_proba(Xs)[:, 1]


if TF_AVAILABLE:
    def _build_tcn_autoencoder(window_size, n_features, latent_dim=64):
        inp = layers.Input(shape=(window_size, n_features))
        x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inp)
        x = layers.MaxPool1D(pool_size=2)(x)
        x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.MaxPool1D(pool_size=2)(x)
        x = layers.Flatten()(x)
        z = layers.Dense(latent_dim, activation='relu', name='latent')(x)

        # decoder
        x = layers.Dense((window_size // 4) * 128, activation='relu')(z)
        x = layers.Reshape((window_size // 4, 128))(x)
        x = layers.UpSampling1D(size=2)(x)
        x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.UpSampling1D(size=2)(x)
        out = layers.Conv1D(n_features, kernel_size=3, padding='same', activation='linear')(x)

        model = models.Model(inputs=inp, outputs=out)
        encoder = models.Model(inputs=inp, outputs=z)
        model.compile(optimizer='adam', loss='mse')
        return model, encoder


    class TCNAutoencoder:
        def __init__(self, window_size, n_features, latent_dim=64):
            self.window_size = window_size
            self.n_features = n_features
            self.latent_dim = latent_dim
            self.model, self.encoder = _build_tcn_autoencoder(window_size, n_features, latent_dim)

        def fit(self, X_windows, epochs=20, batch_size=64, validation_split=0.05):
            # X_windows shape: (n_windows, window_size, n_features)
            self.model.fit(np.array(X_windows), np.array(X_windows), epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)

        def score(self, X_windows):
            recon = self.model.predict(np.array(X_windows))
            mse = np.mean(np.square(recon - np.array(X_windows)), axis=(1,2))
            return mse


    def _build_vae(window_size, n_features, latent_dim=32):
        inp = layers.Input(shape=(window_size, n_features))
        x = layers.Flatten()(inp)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        z_mean = layers.Dense(latent_dim)(x)
        z_log_var = layers.Dense(latent_dim)(x)

        def sampling(args):
            z_mean, z_log_var = args
            eps = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * eps

        z = layers.Lambda(sampling)([z_mean, z_log_var])

        # decoder
        x = layers.Dense(128, activation='relu')(z)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(window_size * n_features, activation='linear')(x)
        out = layers.Reshape((window_size, n_features))(x)

        encoder = models.Model(inp, [z_mean, z_log_var, z], name='encoder')
        vae = models.Model(inp, out, name='vae')

        # VAE loss
        mse_loss = tf.reduce_mean(tf.square(tf.reshape(inp, [-1, window_size * n_features]) - tf.reshape(out, [-1, window_size * n_features])), axis=1)
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        vae_loss = tf.reduce_mean(mse_loss + kl_loss)

        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        return vae, encoder


    class VAETracer:
        def __init__(self, window_size, n_features, latent_dim=32):
            self.vae, self.encoder = _build_vae(window_size, n_features, latent_dim)

        def fit(self, X_windows, epochs=50, batch_size=64, validation_split=0.05):
            self.vae.fit(np.array(X_windows), epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1)

        def score(self, X_windows):
            recon = self.vae.predict(np.array(X_windows))
            mse = np.mean(np.square(recon - np.array(X_windows)), axis=(1,2))
            return mse


    class DeepSVDDDetector:
        """A simple Deep-SVDD-like detector that uses an encoder and distances to center.

        Implementation: use encoder (from VAE or TCN) to compute latent vectors on training normal windows,
        set center = mean(latents), then anomaly score = squared distance to center.
        This is a lightweight approximation (not full SVDD training loop).
        """
        def __init__(self, encoder_model):
            self.encoder = encoder_model
            self.center = None

        def fit(self, X_windows):
            z = self.encoder.predict(np.array(X_windows))
            # if encoder outputs [z_mean, z_log_var, z], handle accordingly
            if isinstance(z, list):
                z = z[-1]
            self.center = np.mean(z, axis=0)

        def score(self, X_windows):
            z = self.encoder.predict(np.array(X_windows))
            if isinstance(z, list):
                z = z[-1]
            dist = np.sum(np.square(z - self.center), axis=1)
            return dist


else:
    # TensorFlow not available: define placeholders that raise informative errors
    TF_AVAILABLE = False

    class TCNAutoencoder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('TensorFlow/Keras is not available')

    class VAETracer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('TensorFlow/Keras is not available')

    class DeepSVDDDetector:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('TensorFlow/Keras is not available')


__all__ = [
    'create_windows',
    'RandomForestDetector',
    'XGBoostDetector',
    'TCNAutoencoder' if TF_AVAILABLE else None,
    'VAETracer' if TF_AVAILABLE else None,
    'DeepSVDDDetector' if TF_AVAILABLE else None,
]
