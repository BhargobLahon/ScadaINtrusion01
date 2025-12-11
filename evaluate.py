import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn import metrics
try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except Exception:
    HMMLEARN_AVAILABLE = False

from readData import readTrainSamplesNetworkData, readTestSamplesNetworkData
from detectors import create_windows, RandomForestDetector, XGBoostDetector, XGBOOST_AVAILABLE
try:
    from detectors import TCNAutoencoder, VAETracer, DeepSVDDDetector
    TF_DETECTORS_AVAILABLE = True
except Exception:
    TF_DETECTORS_AVAILABLE = False


scriptDir = os.path.dirname(os.path.realpath(__file__))


def normalize_with_max(train, X):
    # train: list/array of samples (n_samples, n_features)
    featureMax = []
    nFeatures = len(train[0])
    for i in range(nFeatures):
        col = np.array([r[i] for r in train])
        m = max(np.abs(col))
        if m == 0:
            m = 1.0
        featureMax.append(m)

    def norm_arr(arr):
        return np.array([[float(x[j]) / featureMax[j] for j in range(nFeatures)] for x in arr])

    return norm_arr(X), np.array(featureMax)


def sliding_window_scores_gmm(gmm, X, window_size=60):
    # use per-sample log probability and average over windows
    ll = gmm.score_samples(X)  # per-sample log prob
    scores = []
    i = 0
    N = len(ll)
    while i + window_size < N:
        scores.append(np.mean(ll[i:i+window_size]))
        i += window_size
    scores.append(np.mean(ll[i:]))
    return np.array(scores)


def sliding_window_scores_hmm(hmm_model, X, window_size=60):
    scores = []
    i = 0
    N = len(X)
    while i + window_size < N:
        try:
            scores.append(hmm_model.score(np.array(X[i:i+window_size])))
        except Exception:
            # fallback: compute per-sample using gmms if available
            scores.append(float('nan'))
        i += window_size
    try:
        scores.append(hmm_model.score(np.array(X[i:])))
    except Exception:
        scores.append(float('nan'))
    return np.array(scores)


def choose_gmm_by_bic(X, max_components=5, cov_type='full'):
    bics = []
    gmms = []
    for k in range(1, max_components+1):
        g = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=0)
        g.fit(X)
        bics.append(g.bic(X))
        gmms.append(g)
    best_idx = int(np.argmin(bics))
    return gmms[best_idx], best_idx+1, bics


def evaluate_method(train_series, attack_series_list, method='gmm', use_pca=True, window_size=60):
    # train_series: np.array shape (n_samples, n_features)
    X_train = np.array(train_series)
    # concat attack series into list of arrays

    # Normalize using max of training
    X_train_norm, featureMax = normalize_with_max(X_train, X_train)

    # PCA
    if use_pca:
        pca = PCA(n_components=min(50, X_train_norm.shape[1]))
        pca.fit(X_train_norm)
        # choose number of components that explain 99% variance
        cum = np.cumsum(pca.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cum, 0.99) + 1)
        pca = PCA(n_components=n_comp)
        X_train_proc = pca.fit_transform(X_train_norm)
    else:
        X_train_proc = X_train_norm

    results = {}

    # Train GMM baseline
    if method == 'gmm' or method == 'both':
        gmm, ncomp, bics = choose_gmm_by_bic(X_train_proc, max_components=5, cov_type='full')
        train_scores = sliding_window_scores_gmm(gmm, X_train_proc, window_size=window_size)
        results['gmm'] = {'model': gmm, 'train_scores': train_scores, 'n_mix': ncomp}

    # Train GMM-HMM if available
    if HMMLEARN_AVAILABLE and (method == 'hmm' or method == 'both'):
        # initialize a simple 1-state GMMHMM with mixtures from GMM (if available)
        try:
            from numpy import array
            n_states = 1
            # use a small mixture count chosen from earlier step if present
            n_mix = results.get('gmm', {}).get('n_mix', 1)
            gmmhmm = hmm.GMMHMM(n_components=n_states, n_mix=n_mix, covariance_type='full', n_iter=1000)
            # fit directly (may take time); if fit fails, we'll skip
            gmmhmm.fit(X_train_proc)
            train_scores_hmm = sliding_window_scores_hmm(gmmhmm, X_train_proc, window_size=window_size)
            results['hmm'] = {'model': gmmhmm, 'train_scores': train_scores_hmm}
        except Exception as e:
            results['hmm'] = {'error': str(e)}

    # Score attacks and compute metrics
    metrics_table = []
    for name, entry in results.items():
        if 'model' not in entry:
            metrics_table.append((name, 'error', entry.get('error', 'no model')))
            continue

        train_scores = entry['train_scores']
        # build combined negatives (train windows) and positives (attack windows)
        pos_scores = []
        for a in attack_series_list:
            a_arr = np.array(a)
            # normalize and pca-transform
            a_norm = np.array([[float(x[j]) / featureMax[j] for j in range(a_arr.shape[1])] for x in a_arr])
            if use_pca:
                a_proc = pca.transform(a_norm)
            else:
                a_proc = a_norm

            if name == 'gmm':
                s = sliding_window_scores_gmm(entry['model'], a_proc, window_size=window_size)
            elif name == 'hmm':
                s = sliding_window_scores_hmm(entry['model'], a_proc, window_size=window_size)
            else:
                s = np.array([])
            pos_scores.append(s)

        pos_scores = np.concatenate([p for p in pos_scores if len(p) > 0]) if len(pos_scores) > 0 else np.array([])

        # build ROC using train_scores as negatives and pos_scores as positives
        y = np.concatenate([np.zeros_like(train_scores), np.ones_like(pos_scores)])
        scores_for_roc = np.concatenate([train_scores, pos_scores])

        # note: lower score => more anomalous (for log-likelihood). We invert scores for ROC if needed
        # We'll use decision: predict anomaly if score < threshold, so to compute AUC where larger scores imply positive
        # we invert by negative
        fpr, tpr, thresholds = metrics.roc_curve(y, -scores_for_roc)
        auc = metrics.auc(fpr, tpr)

        # choose threshold at 5th percentile of train scores
        th = np.percentile(train_scores, 5)
        y_pred = (scores_for_roc < th).astype(int)
        acc = metrics.accuracy_score(y, y_pred)
        prec = metrics.precision_score(y, y_pred, zero_division=0)
        rec = metrics.recall_score(y, y_pred, zero_division=0)
        f1 = metrics.f1_score(y, y_pred, zero_division=0)

        metrics_table.append((name, auc, acc, prec, rec, f1, th))

    return metrics_table, results


def main():
    # file paths similar to getSystemPeriod.py
    trainCommandFile = scriptDir + '/DataSets/Command_Injection/AddressScanScrubbedV2.csv'
    trainResponseFile = scriptDir + '/DataSets/Response_Injection/ScrubbedBurstV2/scrubbedBurstV2.csv'
    dosAttackDataFile = scriptDir + '/DataSets/DoS_Data_FeatureSet/modbusRTU_DoSResponseInjectionV2.csv'
    functionScanDataFile = scriptDir + '/DataSets/Command_Injection/FunctionCodeScanScrubbedV2.csv'
    burstResponseFile = scriptDir + '/DataSets/Response_Injection/ScrubbedBurstV2/scrubbedBurstV2.csv'
    fastburstResponseFile = scriptDir + '/DataSets/Response_Injection/ScrubbedFastV2/scrubbedFastV2.csv'
    slowburstResponseFile = scriptDir + '/DataSets/Response_Injection/ScrubbedSlowV2/scrubbedSlowV2.csv'

    print('Reading data...')
    train_char, train_ts = readTrainSamplesNetworkData(trainCommandFile, trainResponseFile, 1, 1000, 28071)
    _, dos_ts = readTestSamplesNetworkData(dosAttackDataFile, 1, 1000, -1)
    _, fscan_ts = readTestSamplesNetworkData(functionScanDataFile, 1, 1000, -1)
    _, burst_ts = readTestSamplesNetworkData(burstResponseFile, 1, 1000, -1)
    _, fastburst_ts = readTestSamplesNetworkData(fastburstResponseFile, 1, 1000, -1)
    _, slowburst_ts = readTestSamplesNetworkData(slowburstResponseFile, 1, 1000, -1)

    # use training series and list of attack series
    attack_list = [dos_ts, fscan_ts, burst_ts, fastburst_ts, slowburst_ts]

    print('Evaluating GMM baseline...')
    metrics_table, _ = evaluate_method(train_ts, attack_list, method='gmm', use_pca=True, window_size=60)
    print('\nResults (method, AUC, Accuracy, Precision, Recall, F1, threshold):')
    for row in metrics_table:
        print(row)

    if HMMLEARN_AVAILABLE:
        print('\nEvaluating GMM-HMM...')
        metrics_table_hmm, _ = evaluate_method(train_ts, attack_list, method='hmm', use_pca=True, window_size=60)
        for row in metrics_table_hmm:
            print(row)
    else:
        print('\nNote: hmmlearn not available in this environment; GMM-HMM evaluation skipped.')

    # Supervised detectors: RandomForest and XGBoost (if available)
    print('\nEvaluating supervised detectors (RandomForest, XGBoost) using windowed labels...')
    # build windows from train (normal) and attacks (anomalous)
    window_size = 60
    step = 60
    X_train_windows = create_windows(train_ts, window_size=window_size, step=step)
    # flatten and label
    X_norm = np.array(X_train_windows)
    y_norm = np.zeros(X_norm.shape[0], dtype=int)

    attack_windows_all = []
    for a in attack_list:
        a_w = create_windows(a, window_size=window_size, step=step)
        attack_windows_all.append(a_w)

    X_attack_windows = np.concatenate([aw for aw in attack_windows_all if len(aw) > 0], axis=0) if len(attack_windows_all) > 0 else np.array([])
    if X_attack_windows.size == 0:
        print('No attack windows found; skipping supervised evaluation.')
    else:
        y_attack = np.ones(X_attack_windows.shape[0], dtype=int)
        # combine
        X_comb = np.vstack([X_norm.reshape(X_norm.shape[0], -1), X_attack_windows.reshape(X_attack_windows.shape[0], -1)])
        y_comb = np.concatenate([y_norm, y_attack])

        # split
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X_comb, y_comb, test_size=0.3, stratify=y_comb, random_state=0)

        # scale
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_trs = scaler.fit_transform(X_tr)
        X_tes = scaler.transform(X_te)

        # RandomForest
        rf = RandomForestDetector(n_estimators=100, random_state=0)
        # NOTE: RandomForestDetector expects windows, so reshape back
        rf.fit(np.reshape(X_tr, (X_tr.shape[0], window_size, -1)), y_tr)
        rf_scores = rf.score(np.reshape(X_tes, (X_tes.shape[0], window_size, -1)))
        auc_rf = metrics.roc_auc_score(y_te, rf_scores)
        y_pred_rf = (rf_scores > 0.5).astype(int)
        acc_rf = metrics.accuracy_score(y_te, y_pred_rf)
        prec_rf = metrics.precision_score(y_te, y_pred_rf, zero_division=0)
        rec_rf = metrics.recall_score(y_te, y_pred_rf, zero_division=0)
        f1_rf = metrics.f1_score(y_te, y_pred_rf, zero_division=0)
        print(('RandomForest', auc_rf, acc_rf, prec_rf, rec_rf, f1_rf))

        # XGBoost if available
        if XGBOOST_AVAILABLE:
            try:
                xgb_det = XGBoostDetector(use_label_encoder=False, eval_metric='logloss')
                xgb_det.fit(np.reshape(X_tr, (X_tr.shape[0], window_size, -1)), y_tr)
                xgb_scores = xgb_det.score(np.reshape(X_tes, (X_tes.shape[0], window_size, -1)))
                auc_xgb = metrics.roc_auc_score(y_te, xgb_scores)
                y_pred_xgb = (xgb_scores > 0.5).astype(int)
                acc_xgb = metrics.accuracy_score(y_te, y_pred_xgb)
                prec_xgb = metrics.precision_score(y_te, y_pred_xgb, zero_division=0)
                rec_xgb = metrics.recall_score(y_te, y_pred_xgb, zero_division=0)
                f1_xgb = metrics.f1_score(y_te, y_pred_xgb, zero_division=0)
                print(('XGBoost', auc_xgb, acc_xgb, prec_xgb, rec_xgb, f1_xgb))
            except Exception as e:
                print('XGBoost evaluation skipped:', e)
        else:
            print('XGBoost not available; skipping')

    # Deep unsupervised detectors (train on normal windows, score attacks)
    if TF_DETECTORS_AVAILABLE:
        print('\nEvaluating deep unsupervised detectors (TCN Autoencoder, VAE, DeepSVDD)')
        # prepare windows
        W_train = create_windows(train_ts, window_size=window_size, step=step)
        # compute train scores for threshold
        # TCN
        try:
            tcn = TCNAutoencoder(window_size=window_size, n_features=W_train.shape[2], latent_dim=64)
            tcn.fit(W_train, epochs=5, batch_size=128)
            train_scores_tcn = tcn.score(W_train)
            # score attacks
            pos_scores = []
            for aw in attack_windows_all:
                if len(aw) > 0:
                    pos_scores.append(tcn.score(aw))
            pos_scores = np.concatenate([p for p in pos_scores if len(p) > 0]) if len(pos_scores) > 0 else np.array([])
            if pos_scores.size > 0:
                fpr, tpr, _ = metrics.roc_curve(np.concatenate([np.zeros_like(train_scores_tcn), np.ones_like(pos_scores)]), -np.concatenate([train_scores_tcn, pos_scores]))
                auc_tcn = metrics.auc(fpr, tpr)
                th = np.percentile(train_scores_tcn, 95)
                y_pred = (np.concatenate([train_scores_tcn, pos_scores]) > th).astype(int)
                y_true = np.concatenate([np.zeros_like(train_scores_tcn), np.ones_like(pos_scores)])
                print(('TCN-AE', auc_tcn, metrics.accuracy_score(y_true, y_pred), metrics.precision_score(y_true, y_pred, zero_division=0), metrics.recall_score(y_true, y_pred, zero_division=0), metrics.f1_score(y_true, y_pred, zero_division=0)))
        except Exception as e:
            print('TCN evaluation skipped:', e)

        # VAE
        try:
            vae = VAETracer(window_size=window_size, n_features=W_train.shape[2], latent_dim=32)
            vae.fit(W_train, epochs=10, batch_size=128)
            train_scores_vae = vae.score(W_train)
            pos_scores = []
            for aw in attack_windows_all:
                if len(aw) > 0:
                    pos_scores.append(vae.score(aw))
            pos_scores = np.concatenate([p for p in pos_scores if len(p) > 0]) if len(pos_scores) > 0 else np.array([])
            if pos_scores.size > 0:
                fpr, tpr, _ = metrics.roc_curve(np.concatenate([np.zeros_like(train_scores_vae), np.ones_like(pos_scores)]), -np.concatenate([train_scores_vae, pos_scores]))
                auc_vae = metrics.auc(fpr, tpr)
                th = np.percentile(train_scores_vae, 95)
                y_pred = (np.concatenate([train_scores_vae, pos_scores]) > th).astype(int)
                y_true = np.concatenate([np.zeros_like(train_scores_vae), np.ones_like(pos_scores)])
                print(('VAE', auc_vae, metrics.accuracy_score(y_true, y_pred), metrics.precision_score(y_true, y_pred, zero_division=0), metrics.recall_score(y_true, y_pred, zero_division=0), metrics.f1_score(y_true, y_pred, zero_division=0)))
        except Exception as e:
            print('VAE evaluation skipped:', e)

        # DeepSVDD wrapper using VAE encoder or TCN encoder
        try:
            # prefer VAE encoder if available
            if 'vae' in locals():
                encoder = vae.encoder
            elif 'tcn' in locals():
                encoder = tcn.encoder
            else:
                encoder = None

            if encoder is not None:
                deep_svdd = DeepSVDDDetector(encoder)
                deep_svdd.fit(W_train)
                train_scores_ds = deep_svdd.score(W_train)
                pos_scores = []
                for aw in attack_windows_all:
                    if len(aw) > 0:
                        pos_scores.append(deep_svdd.score(aw))
                pos_scores = np.concatenate([p for p in pos_scores if len(p) > 0]) if len(pos_scores) > 0 else np.array([])
                if pos_scores.size > 0:
                    fpr, tpr, _ = metrics.roc_curve(np.concatenate([np.zeros_like(train_scores_ds), np.ones_like(pos_scores)]), -np.concatenate([train_scores_ds, pos_scores]))
                    auc_ds = metrics.auc(fpr, tpr)
                    th = np.percentile(train_scores_ds, 95)
                    y_pred = (np.concatenate([train_scores_ds, pos_scores]) > th).astype(int)
                    y_true = np.concatenate([np.zeros_like(train_scores_ds), np.ones_like(pos_scores)])
                    print(('DeepSVDD', auc_ds, metrics.accuracy_score(y_true, y_pred), metrics.precision_score(y_true, y_pred, zero_division=0), metrics.recall_score(y_true, y_pred, zero_division=0), metrics.f1_score(y_true, y_pred, zero_division=0)))
        except Exception as e:
            print('DeepSVDD evaluation skipped:', e)
    else:
        print('\nTensorFlow-based detectors not available; skipping deep model evaluation.')

    # Save a minimal CSV summary (append mode)
    try:
        import csv
        csvfile = os.path.join(scriptDir, 'evaluation_results.csv')
        header = ['method','AUC','Accuracy','Precision','Recall','F1','threshold']
        # gather rows printed earlier by re-running lightweight metrics where possible
        # For now, just append a note that evaluation completed
        with open(csvfile, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow(['evaluation_run', 'completed'])
        print('\nWrote minimal evaluation marker to', csvfile)
    except Exception as e:
        print('Failed to write CSV summary:', e)


if __name__ == '__main__':
    main()
