"""
Ablation Study: Traditional Kernels vs RFM Kernels
Compares:
1. Traditional kernels only (RBF + Polynomial)
2. RFM kernels only (Gaussian-RFM + Laplacian-RFM)
3. Combined (Traditional + RFM) - full DASMKL
"""
import sys
import os
import numpy as np
import random
import time
import argparse

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
rfm_path = os.path.join(SCRIPT_DIR, 'rfm')
if rfm_path not in sys.path:
    sys.path.insert(0, rfm_path)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from rfm.kernels import euclidean_distances_M, gaussian_M as rfm_gaussian_M, laplacian_M as rfm_laplacian_M
from rfm import GaussRFM, LaplaceRFM

from libsvmdata import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from MKLpy.algorithms import AverageMKL
from MKLpy.preprocessing.data_preprocessing import rescale_01
from MKLpy.metrics.pairwise import rbf_kernel, polynomial_kernel
from MKLpy.preprocessing.kernel_preprocessing import kernel_normalization, kernel_centering


def read_data1(data_set):
    X, y = datasets.fetch_libsvm(data_set)
    X = rescale_01(X)
    y = np.reshape(y, (y.shape[0], 1))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.333, random_state=random.randint(0, 100)
    )
    return X_train, y_train, X_test, y_test


def read_data2(data_set):
    X_train, y_train = datasets.fetch_libsvm(data_set)
    X_test, y_test = datasets.fetch_libsvm(data_set + '_test')
    if type(X_train) is not np.ndarray:
        X_train = rescale_01(X_train.toarray())
    if type(X_test) is not np.ndarray:
        X_test = rescale_01(X_test.toarray())
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))
    return X_train, y_train, X_test, y_test


def normalize_kernel_train(K):
    """Normalize and center training kernel (square matrix)."""
    K = np.asarray(K)
    diag = np.diag(K)
    diag = np.maximum(diag, 1e-10)
    K_norm = K / np.sqrt(np.outer(diag, diag))
    # Center
    n = K_norm.shape[0]
    one_n = np.ones((n, n)) / n
    K_centered = K_norm - one_n @ K_norm - K_norm @ one_n + one_n @ K_norm @ one_n
    return K_centered, diag


def normalize_kernel_test(K_te, diag_tr):
    """Normalize and center test kernel using training diagonal."""
    K_te = np.asarray(K_te)
    n_te, n_tr = K_te.shape
    
    # Normalize using training diagonal
    K_norm = K_te / np.sqrt(diag_tr.reshape(1, -1))
    
    # Center (simplified for rectangular)
    row_mean = K_norm.mean(axis=1, keepdims=True)
    col_mean = K_norm.mean(axis=0, keepdims=True)
    total_mean = K_norm.mean()
    K_centered = K_norm - row_mean - col_mean + total_mean
    return K_centered


def generate_traditional_kernels(X_train, X_test):
    """Generate only traditional RBF kernels."""
    KLtr, KLte = [], []
    
    X_train = X_train.toarray() if hasattr(X_train, 'toarray') else np.asarray(X_train)
    X_test = X_test.toarray() if hasattr(X_test, 'toarray') else np.asarray(X_test)
    
    # RBF kernels with varying gamma
    dists = np.sort(np.linalg.norm(X_train[:, None] - X_train[None, :], axis=2).ravel())
    med = dists[len(dists)//2] + 1e-12
    base_gamma = 1.0 / (2 * med * med)
    gamma_list = base_gamma * np.logspace(-2, 2, 20)
    
    for g in gamma_list:
        Ktr = rbf_kernel(X_train, gamma=g)
        Kte = rbf_kernel(X_test, X_train, gamma=g)
        Ktr_norm, diag_tr = normalize_kernel_train(Ktr)
        Kte_norm = normalize_kernel_test(Kte, diag_tr)
        KLtr.append(Ktr_norm)
        KLte.append(Kte_norm)
    
    # Polynomial kernels
    for d in range(2, 5):
        Ktr = polynomial_kernel(X_train, degree=d)
        Kte = polynomial_kernel(X_test, X_train, degree=d)
        Ktr_norm, diag_tr = normalize_kernel_train(Ktr)
        Kte_norm = normalize_kernel_test(Kte, diag_tr)
        KLtr.append(Ktr_norm)
        KLte.append(Kte_norm)
    
    return KLtr, KLte


def generate_rfm_kernels(X_train, y_train, X_test, y_test):
    """Generate only RFM kernels (Gaussian-RFM + Laplacian-RFM)."""
    KLtr, KLte = [], []
    
    def to_np(x): 
        return x.toarray() if hasattr(x, 'toarray') else np.asarray(x)
    
    X_train_np = to_np(X_train)
    X_test_np = to_np(X_test)
    
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train_np)
    Xt = scaler.transform(X_test_np)
    
    Xs_t = torch.tensor(Xs, dtype=torch.float32)
    Xt_t = torch.tensor(Xt, dtype=torch.float32)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train.ravel())
    num_classes = len(le.classes_)
    y_onehot = F.one_hot(torch.tensor(y_encoded, dtype=torch.long), num_classes=num_classes).float()
    
    y_test_flat = y_test.ravel()
    try:
        y_test_encoded = le.transform(y_test_flat)
    except ValueError:
        known_classes = set(le.classes_)
        y_test_encoded = np.array([le.transform([y])[0] if y in known_classes else 0 for y in y_test_flat])
    y_test_onehot = F.one_hot(torch.tensor(y_test_encoded, dtype=torch.long), num_classes=num_classes).float()
    
    with torch.no_grad():
        dist_mat = euclidean_distances_M(Xs_t, Xs_t, M=torch.eye(Xs_t.shape[1]))
    med_M = torch.median(dist_mat).item() + 1e-12
    bw_scales = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]  # More bandwidth options
    bw_list = [med_M * s for s in bw_scales]
    
    reg_list = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]  # More regularization options
    
    for ModelClass, func_M in [(GaussRFM, rfm_gaussian_M), (LaplaceRFM, rfm_laplacian_M)]:
        for bw in bw_list:
            for reg in reg_list:
                try:
                    model = ModelClass(bandwidth=bw, reg=reg)
                    model.fit((Xs_t, y_onehot), (Xt_t, y_test_onehot), iters=3, classification=True)
                    Mmat = model.M.detach().cpu()
                    
                    with torch.no_grad():
                        dtr = euclidean_distances_M(Xs_t, Xs_t, M=Mmat)
                    sigma = torch.sqrt(torch.median(dtr)).item() + 1e-6
                    
                    Ktr = func_M(Xs_t, Xs_t, Mmat, bandwidth=sigma).cpu().numpy()
                    Kte = func_M(Xt_t, Xs_t, Mmat, bandwidth=sigma).cpu().numpy()
                    Ktr_norm, diag_tr = normalize_kernel_train(Ktr)
                    Kte_norm = normalize_kernel_test(Kte, diag_tr)
                    KLtr.append(Ktr_norm)
                    KLte.append(Kte_norm)
                except Exception as e:
                    continue
    
    return KLtr, KLte


def generate_combined_kernels(X_train, y_train, X_test, y_test):
    """Generate both traditional and RFM kernels."""
    KLtr_trad, KLte_trad = generate_traditional_kernels(X_train, X_test)
    KLtr_rfm, KLte_rfm = generate_rfm_kernels(X_train, y_train, X_test, y_test)
    return KLtr_trad + KLtr_rfm, KLte_trad + KLte_rfm


def run_mkl(KLtr, KLte, y_train, y_test, C=10):
    """Run AverageMKL with given kernels."""
    if len(KLtr) == 0:
        return 0.0
    
    valid_tr, valid_te = [], []
    for Ktr, Kte in zip(KLtr, KLte):
        if not (np.isnan(Ktr).any() or np.isnan(Kte).any()):
            valid_tr.append(Ktr)
            valid_te.append(Kte)
    
    if len(valid_tr) == 0:
        return 0.0
    
    mkl = AverageMKL(learner=svm.SVC(C=C)).fit(valid_tr, y_train.ravel())
    y_preds = mkl.predict(valid_te)
    return accuracy_score(y_test.ravel(), y_preds)


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=5, help='Number of repetitions')
    parser.add_argument('--C', type=float, default=10, help='SVM C parameter')
    args = parser.parse_args()

    datasets_small = ["sonar", "heart_scale", "diabetes", "german.numer", 
                      "breast-cancer_scale", "ionosphere", "splice"]
    datasets_large = ["a8a", "w7a"]
    all_datasets = datasets_small

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = f'ablation_log_{timestamp}.txt'
    result_file = f'ablation_results_{timestamp}.txt'
    
    log = Logger(log_file)
    sys.stdout = log

    print("=" * 70)
    print("ABLATION STUDY: Traditional Kernels vs RFM Kernels")
    print("=" * 70)
    print(f"Datasets: {all_datasets}")
    print(f"Repeats: {args.repeats}, C: {args.C}")
    print("=" * 70 + "\n")

    results = {'Traditional': {}, 'RFM-Only': {}, 'Combined': {}}

    for dataset in all_datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        
        results['Traditional'][dataset] = []
        results['RFM-Only'][dataset] = []
        results['Combined'][dataset] = []
        
        for rep in range(args.repeats):
            try:
                if dataset in datasets_large:
                    X_train, y_train, X_test, y_test = read_data2(dataset)
                else:
                    X_train, y_train, X_test, y_test = read_data1(dataset)
                
                t0 = time.time()
                KLtr_trad, KLte_trad = generate_traditional_kernels(X_train, X_test)
                acc_trad = run_mkl(KLtr_trad, KLte_trad, y_train, y_test, args.C)
                t_trad = time.time() - t0
                results['Traditional'][dataset].append(acc_trad)
                
                t0 = time.time()
                KLtr_rfm, KLte_rfm = generate_rfm_kernels(X_train, y_train, X_test, y_test)
                acc_rfm = run_mkl(KLtr_rfm, KLte_rfm, y_train, y_test, args.C)
                t_rfm = time.time() - t0
                results['RFM-Only'][dataset].append(acc_rfm)
                
                t0 = time.time()
                KLtr_comb, KLte_comb = generate_combined_kernels(X_train, y_train, X_test, y_test)
                acc_comb = run_mkl(KLtr_comb, KLte_comb, y_train, y_test, args.C)
                t_comb = time.time() - t0
                results['Combined'][dataset].append(acc_comb)
                
                print(f"  Rep {rep+1}: Trad={acc_trad:.4f}({t_trad:.1f}s) | RFM={acc_rfm:.4f}({t_rfm:.1f}s) | Comb={acc_comb:.4f}({t_comb:.1f}s)")
                
            except Exception as e:
                print(f"  Rep {rep+1} FAILED: {e}")
        
        if results['Traditional'][dataset]:
            trad_mean, trad_std = np.mean(results['Traditional'][dataset]), np.std(results['Traditional'][dataset])
            rfm_mean, rfm_std = np.mean(results['RFM-Only'][dataset]), np.std(results['RFM-Only'][dataset])
            comb_mean, comb_std = np.mean(results['Combined'][dataset]), np.std(results['Combined'][dataset])
            print(f"\n  Summary: Trad={trad_mean:.4f}±{trad_std:.4f} | RFM={rfm_mean:.4f}±{rfm_std:.4f} | Comb={comb_mean:.4f}±{comb_std:.4f}")

    print("\n\n" + "=" * 70)
    print("FINAL RESULTS TABLE")
    print("=" * 70)
    print(f"{'Dataset':<20} {'Traditional':<18} {'RFM-Only':<18} {'Combined':<18}")
    print("-" * 70)
    
    with open(result_file, 'w') as f:
        f.write("Dataset,Traditional_mean,Traditional_std,RFM_mean,RFM_std,Combined_mean,Combined_std\n")
        for dataset in all_datasets:
            if results['Traditional'][dataset]:
                trad_mean, trad_std = np.mean(results['Traditional'][dataset]), np.std(results['Traditional'][dataset])
                rfm_mean, rfm_std = np.mean(results['RFM-Only'][dataset]), np.std(results['RFM-Only'][dataset])
                comb_mean, comb_std = np.mean(results['Combined'][dataset]), np.std(results['Combined'][dataset])
                print(f"{dataset:<20} {trad_mean:.4f}±{trad_std:.4f}     {rfm_mean:.4f}±{rfm_std:.4f}     {comb_mean:.4f}±{comb_std:.4f}")
                f.write(f"{dataset},{trad_mean:.4f},{trad_std:.4f},{rfm_mean:.4f},{rfm_std:.4f},{comb_mean:.4f},{comb_std:.4f}\n")

    print("=" * 70)
    print(f"Results saved to: {result_file}")
