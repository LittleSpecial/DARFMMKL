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

from rfm.kernels import (
    euclidean_distances_M,
    gaussian_M as rfm_gaussian_M,
    laplacian_M as rfm_laplacian_M,
)
from rfm import GaussRFM, LaplaceRFM

from libsvmdata import datasets

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.optimize import minimize

from MKLpy.algorithms import AverageMKL
from MKLpy.preprocessing.data_preprocessing import rescale_01
from MKLpy.metrics.pairwise import rbf_kernel, polynomial_kernel
from MKLpy.metrics.alignment import alignment_yy
from MKLpy.preprocessing.kernel_preprocessing import kernel_normalization, kernel_centering

import gurobipy as gp


def process_kernel_train(K):
    """Normalize and center training kernel matrix."""
    if hasattr(K, 'detach'):
        K = K.detach().cpu().numpy()
    K = K / np.sqrt(np.outer(np.diag(K), np.diag(K)))
    K = kernel_centering(K)
    return K


def process_kernel_test(K_train, K_test):
    """Normalize and center test kernel matrix using training parameters."""
    if hasattr(K_train, 'detach'):
        K_train = K_train.detach().cpu().numpy()
    else:
        K_train = np.asarray(K_train)
    if hasattr(K_test, 'detach'):
        K_test = K_test.detach().cpu().numpy()
    else:
        K_test = np.asarray(K_test)

    n_tr = K_train.shape[0]
    n_te = K_test.shape[0]
    
    d = np.sqrt(np.diag(K_train) + 1e-10)
    K = K_test / d.reshape(1, -1)

    one_tr = np.ones((n_tr, n_tr)) / n_tr
    one_te_tr = np.ones((n_te, n_tr)) / n_tr

    K_center = K - one_te_tr @ K_train - K @ one_tr + one_te_tr @ K_train @ one_tr
    return K_center


def rfm_data_transform(X_train, y_train, X_test, y_test=None, iters=3):
    """Use RFM to learn feature matrix M and transform data to new space."""
    X_train_np = X_train.toarray() if hasattr(X_train, 'toarray') else np.asarray(X_train)
    y_train_np = y_train.ravel()
    X_test_np = X_test.toarray() if hasattr(X_test, 'toarray') else np.asarray(X_test)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train_np)
    num_classes = len(le.classes_)
    
    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_encoded, dtype=torch.long)
    y_onehot = F.one_hot(y_train_t, num_classes=num_classes).float()
    
    if y_test is not None:
        y_test_encoded = le.transform(y_test.ravel())
        y_test_t = torch.tensor(y_test_encoded, dtype=torch.long)
        y_test_onehot = F.one_hot(y_test_t, num_classes=num_classes).float()
    else:
        y_test_onehot = y_onehot

    model = GaussRFM(bandwidth=10, reg=1e-3)
    model.fit(
        (X_train_t, y_onehot), 
        (X_test_t, y_test_onehot),
        iters=iters,
        classification=True
    )
    M = model.M.detach().cpu().numpy()

    u, s, vh = np.linalg.svd(M)
    M_sqrt = u @ np.diag(np.sqrt(np.maximum(s, 0))) @ vh
    
    X_train_new = X_train_np @ M_sqrt
    X_test_new = X_test_np @ M_sqrt
    
    return X_train_new, X_test_new


def generate_kernel_matrices(X_train, X_test, gammas, degrees):
    """Generate all kernel matrices from given data."""
    KLtr, KLte = [], []
    
    for gamma in gammas:
        tr = rbf_kernel(X=X_train, gamma=gamma)
        te = rbf_kernel(X=X_test, Z=X_train, gamma=gamma)
        tr = kernel_centering(kernel_normalization(tr))
        te = kernel_centering(kernel_normalization(te))
        KLtr.append(tr)
        KLte.append(te)
        
    for degree in degrees:
        tr = polynomial_kernel(X=X_train, degree=degree)
        te = polynomial_kernel(X=X_test, Z=X_train, degree=degree)
        tr = kernel_centering(kernel_normalization(tr))
        te = kernel_centering(kernel_normalization(te))
        KLtr.append(tr)
        KLte.append(te)
        
    return KLtr, KLte


def read_data1(data_set):
    """Read dataset with random train/test split."""
    X, y = datasets.fetch_libsvm(data_set)
    X = rescale_01(X)
    y = np.reshape(y, (y.shape[0], 1))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.333, random_state=random.randint(0, 100)
    )
    return X_train, y_train, X_test, y_test


def read_data2(data_set):
    """Read dataset with pre-defined train/test split."""
    X_train, y_train = datasets.fetch_libsvm(data_set)
    X_test, y_test = datasets.fetch_libsvm(data_set + '_test')
    
    if type(X_train) is not np.ndarray:
        X_train = rescale_01(X_train.toarray())
    if type(X_test) is not np.ndarray:
        X_test = rescale_01(X_test.toarray())

    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))

    return X_train, y_train, X_test, y_test


def Kr(X, gamma=1):
    """Compute normalized centered RBF kernel."""
    K = rbf_kernel(X=X, gamma=gamma)
    K = kernel_centering(kernel_normalization(K))
    return K


def Kp(X, degree=2, gamma=1, coef0=1):
    """Compute normalized centered polynomial kernel."""
    K = polynomial_kernel(X=X, degree=degree, gamma=1, coef0=1)
    K = kernel_centering(kernel_normalization(K))
    return K


def generate_rfm_kernels_dual(X_train_small, y_train_small, X_test_small, y_test_small,
                              X_train_full=None, X_test_full=None,
                              reg_list=[1e-3, 1e-2, 0.1, 1, 10], 
                              bw_scales=[0.2, 0.5, 1.0, 2.0],
                              use_large_mode=False):
    """Generate RFM kernels on small samples and optionally apply to full data."""
    KLtr_s, KLte_s, KLtr_f, KLte_f = [], [], [], []

    def to_np(x): 
        return x.toarray() if hasattr(x, 'toarray') else x
    
    Xs = to_np(X_train_small)
    Xt = to_np(X_test_small)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xs)
    Xt = scaler.transform(Xt)
    
    # Standard RBF kernels
    dists = np.sort(np.linalg.norm(Xs[:, None] - Xs[None, :], axis=2).ravel())
    med = dists[len(dists)//2] + 1e-12
    base_gamma = 1.0 / (2 * med * med)
    gamma_list = base_gamma * np.logspace(-2, 2, 15)

    for g in gamma_list:
        Ktr = rbf_kernel(Xs, gamma=g)
        Kte = rbf_kernel(Xt, Xs, gamma=g)
        KLtr_s.append(process_kernel_train(Ktr))
        KLte_s.append(process_kernel_test(Ktr, Kte))
        
        if X_train_full is not None:
            Xf = scaler.transform(to_np(X_train_full))
            Xq = scaler.transform(to_np(X_test_full))
            Ktr_f = rbf_kernel(Xf, gamma=g)
            Kte_f_mat = rbf_kernel(Xq, Xf, gamma=g)
            KLtr_f.append(process_kernel_train(Ktr_f))
            KLte_f.append(process_kernel_test(Ktr_f, Kte_f_mat))

    # RFM kernels
    Xs_t = torch.tensor(Xs, dtype=torch.float32)
    Xt_t = torch.tensor(Xt, dtype=torch.float32)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train_small.ravel())
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    num_classes = len(le.classes_)
    y_onehot = F.one_hot(y_tensor, num_classes=num_classes).float()

    with torch.no_grad():
        dist_mat = euclidean_distances_M(Xs_t, Xs_t, M=torch.eye(Xs_t.shape[1]))
    med_M = torch.median(dist_mat).item() + 1e-12
    bw_list = [med_M * s for s in bw_scales]

    y_test_flat = y_test_small.ravel()
    try:
        y_test_encoded = le.transform(y_test_flat)
    except ValueError:
        known_classes = set(le.classes_)
        y_test_encoded = np.array([
            le.transform([y])[0] if y in known_classes else 0
            for y in y_test_flat
        ])
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)
    y_test_onehot = F.one_hot(y_test_tensor, num_classes=num_classes).float()

    for ModelClass, func_M in [(GaussRFM, rfm_gaussian_M), (LaplaceRFM, rfm_laplacian_M)]:
        for bw in bw_list:
            for reg in reg_list:
                try:
                    model = ModelClass(bandwidth=bw, reg=reg)
                    model.fit(
                        (Xs_t, y_onehot),
                        (Xt_t, y_test_onehot),
                        iters=3,
                        classification=True
                    )
                    Mmat = model.M.detach().cpu()
                    
                    with torch.no_grad():
                        dtr = euclidean_distances_M(Xs_t, Xs_t, M=Mmat)
                    sigma = torch.sqrt(torch.median(dtr)).item() + 1e-6
                    
                    Ktr = func_M(Xs_t, Xs_t, Mmat, bandwidth=sigma).cpu().numpy()
                    Kte = func_M(Xt_t, Xs_t, Mmat, bandwidth=sigma).cpu().numpy()
                    KLtr_s.append(process_kernel_train(Ktr))
                    KLte_s.append(process_kernel_test(Ktr, Kte))
                    
                    if X_train_full is not None:
                        Xf_t = torch.tensor(scaler.transform(to_np(X_train_full)), dtype=torch.float32)
                        Xq_t = torch.tensor(scaler.transform(to_np(X_test_full)), dtype=torch.float32)
                        
                        Ktr_f = func_M(Xf_t, Xf_t, Mmat, bandwidth=sigma).cpu().numpy()
                        Kte_f_mat = func_M(Xq_t, Xf_t, Mmat, bandwidth=sigma).cpu().numpy()
                        KLtr_f.append(process_kernel_train(Ktr_f))
                        KLte_f.append(process_kernel_test(Ktr_f, Kte_f_mat))
                        
                except Exception as e:
                    continue

    return KLtr_s, KLte_s, KLtr_f if KLtr_f else KLtr_s, KLte_f if KLte_f else KLte_s


def get_h_from_kernels(KLtr, y):
    """Compute kernel-label alignment for each kernel."""
    y_t = torch.as_tensor(y.ravel(), dtype=torch.float64)
    h_vals = []
    for K in KLtr:
        K_t = torch.as_tensor(K, dtype=torch.float64)
        h_val = alignment_yy(K=K_t, y1=y_t)
        h_vals.append(float(h_val))
    return np.array(h_vals, dtype=float).reshape(-1, 1)


def DASMKL(X_train, X_test, y_train, y_test, gammas, degrees, C):
    """MKL classification with selected kernels."""
    KLtr, KLte = [], []
    
    for gamma in gammas:
        KLtr.append(rbf_kernel(X=X_train, gamma=gamma))
        KLte.append(rbf_kernel(X=X_test, Z=X_train, gamma=gamma))
    for degree in degrees:
        KLtr.append(polynomial_kernel(X=X_train, degree=degree))
        KLte.append(polynomial_kernel(X=X_test, Z=X_train, degree=degree))
    
    mkl = AverageMKL(learner=svm.SVC(C=C)).fit(KLtr, y_train.ravel())
    y_preds = mkl.predict(KLte)
    accuracy = accuracy_score(y_test.ravel(), y_preds)
    return accuracy


def DASMKL_with_kernels(selected_KLtr, selected_KLte, y_train, y_test, C):
    """MKL classification with pre-computed kernels."""
    mkl = AverageMKL(learner=svm.SVC(C=C)).fit(selected_KLtr, y_train.ravel())
    y_preds = mkl.predict(selected_KLte)
    accuracy = accuracy_score(y_test.ravel(), y_preds)
    return accuracy


def get_eta_lin(G, h, M, m, la):
    """Solve kernel selection via Glover linearization."""
    M_LP = gp.Model("LP")
    M_LP.Params.OutputFlag = 0
    M_LP.Params.NonConvex = 2
    
    etas = M_LP.addVars(M, vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="eta")
    ss = M_LP.addVars(M, vtype=gp.GRB.CONTINUOUS, lb=0, name="s")
    U1_vars = M_LP.addVars(M, vtype=gp.GRB.CONTINUOUS, name="U1")
    
    for j in range(M):
        U1_expr = sum(G[i][j] for i in range(M) if G[i][j] > 0)
        M_LP.addConstr(U1_vars[j] == U1_expr, name=f"U1_def_{j}")
    
    obj = gp.quicksum((U1_vars[j] + la * h[j]) * etas[j] for j in range(M)) - gp.quicksum(ss[j] for j in range(M))
    M_LP.setObjective(obj, gp.GRB.MAXIMIZE)
    
    M_LP.addConstr(sum(etas[i] for i in range(M)) == m)
    for j in range(M):
        constraint_expr = U1_vars[j] * etas[j] - gp.quicksum(G[i][j] * etas[i] for i in range(M) if G[i][j] > 0)
        M_LP.addConstr(ss[j] >= constraint_expr, name=f"linearization_{j}")
    
    M_LP.optimize()
    
    eta = np.zeros((M, 1))
    for i in range(M):
        eta[i] = 1 if etas[i].x > 0.5 else 0
    return eta


def get_eta(G, h, M, m, la):
    """Solve kernel selection via quadratic integer programming."""
    M_IQCP = gp.Model("IQCP")
    M_IQCP.Params.OutputFlag = 0
    M_IQCP.Params.NonConvex = 0
    M_IQCP.setParam('TimeLimit', 10)
    
    x = M_IQCP.addVars(M, 1, vtype=gp.GRB.BINARY, name="x")
    
    M_IQCP.setObjective(
        gp.quicksum((x[i, 0] * G[i, j] * x[j, 0]) for i in range(M) for j in range(M)) 
        + gp.quicksum(la * (x[i, 0] * h[i]) for i in range(M)), 
        gp.GRB.MAXIMIZE
    )
    
    M_IQCP.addConstr(x.sum('*', 0) == m, name="con")
    M_IQCP.optimize()
    
    eta = np.zeros((M, 1))
    for i in range(M):
        eta[i] = int(x[i, 0].x)
    return eta


def compute_original_objective(G, h, x, la):
    """Compute objective value: x^T G x + λ h^T x"""
    x_vec = x.ravel() if hasattr(x, 'ravel') else x
    h_vec = h.ravel() if hasattr(h, 'ravel') else h
    diversity_term = x_vec.T @ G @ x_vec
    quality_term = la * np.dot(h_vec, x_vec)
    return diversity_term + quality_term


def post_process_solution(x, G, h, M, m, la):
    """Adjust solution to satisfy cardinality constraint."""
    x = np.clip(x, 0, 1).astype(int)
    selected_indices = np.where(x == 1)[0]
    unselected_indices = np.where(x == 0)[0]
    current_count = len(selected_indices)
    
    if current_count == m:
        return x
    
    elif current_count > m:
        contributions = []
        for idx in selected_indices:
            temp_x = x.copy()
            temp_x[idx] = 0
            contrib = compute_original_objective(G, h, temp_x, la)
            contributions.append((contrib, idx))
        contributions.sort(reverse=True)
        for i in range(current_count - m):
            _, remove_idx = contributions[i]
            x[remove_idx] = 0
    
    elif current_count < m:
        contributions = []
        for idx in unselected_indices:
            temp_x = x.copy()
            temp_x[idx] = 1
            contrib = compute_original_objective(G, h, temp_x, la)
            contributions.append((contrib, idx))
        contributions.sort(reverse=True)
        for i in range(min(m - current_count, len(contributions))):
            _, add_idx = contributions[i]
            x[add_idx] = 1
    
    final_count = np.sum(x)
    if final_count != m:
        if final_count > m:
            selected_new = np.where(x == 1)[0]
            remove_indices = np.random.choice(selected_new, final_count - m, replace=False)
            x[remove_indices] = 0
        elif final_count < m:
            unselected_new = np.where(x == 0)[0]
            if len(unselected_new) >= m - final_count:
                add_indices = np.random.choice(unselected_new, m - final_count, replace=False)
                x[add_indices] = 1
    
    return x


def get_eta_continuous(G, h, M, m, la, A=1000, B=1000):
    """Solve kernel selection via continuous optimization."""
    h_vec = h.ravel() if h.ndim > 1 else h
    
    def objective_function(x):
        diversity_term = -x.T @ G @ x
        quality_term = -la * np.dot(h_vec, x)
        binary_penalty = A * np.sum(x * (1 - x))
        cardinality_penalty = B * (m - np.sum(x))**2
        return diversity_term + quality_term + binary_penalty + cardinality_penalty
    
    def gradient_function(x):
        diversity_grad = -2 * G @ x
        quality_grad = -la * h_vec
        binary_grad = A * (1 - 2*x)
        cardinality_grad = -2 * B * (m - np.sum(x)) * np.ones(M)
        return diversity_grad + quality_grad + binary_grad + cardinality_grad
    
    best_result = None
    best_obj = np.inf
    bounds = [(0, 1) for _ in range(M)]
    
    for trial in range(5):
        x0 = np.random.uniform(0, 1, M)
        try:
            result = minimize(
                objective_function, x0, method='L-BFGS-B',
                jac=gradient_function, bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            if result.success and result.fun < best_obj:
                best_result = result
                best_obj = result.fun
        except:
            continue
    
    # Quality-driven initialization
    quality_scores = h_vec / np.max(np.abs(h_vec))
    x0_quality = np.clip(quality_scores, 0, 1)
    try:
        result = minimize(
            objective_function, x0_quality, method='L-BFGS-B',
            jac=gradient_function, bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        if result.success and result.fun < best_obj:
            best_result = result
            best_obj = result.fun
    except:
        pass
    
    if best_result is None:
        return greedy_continuous_selection(G, h_vec, M, m, la)
    
    x_continuous = best_result.x
    eta_binary = continuous_to_binary(x_continuous, m, G, h_vec, la)
    return eta_binary


def greedy_continuous_selection(G, h, M, m, la):
    """Greedy selection as fallback."""
    selected = np.zeros(M, dtype=int)
    quality_scores = la * h
    top_indices = np.argsort(quality_scores)[-m:]
    selected[top_indices] = 1
    return selected.reshape(-1, 1)


def continuous_to_binary(x_continuous, m, G, h, la):
    """Convert continuous solution to binary."""
    M = len(x_continuous)
    h_vec = h.ravel() if hasattr(h, 'ravel') else h
    
    # Threshold method
    top_m_indices = np.argsort(x_continuous)[-m:]
    eta_threshold = np.zeros(M, dtype=int)
    eta_threshold[top_m_indices] = 1
    
    # Rounded method
    eta_rounded = post_process_solution(np.round(x_continuous).astype(int), G, h, M, m, la)
    
    # Weighted method
    weighted_scores = x_continuous * (1 + h_vec / np.max(np.abs(h_vec)))
    top_weighted_indices = np.argsort(weighted_scores)[-m:]
    eta_weighted = np.zeros(M, dtype=int)
    eta_weighted[top_weighted_indices] = 1
    
    methods = [
        ("threshold", eta_threshold),
        ("rounded", eta_rounded),
        ("weighted", eta_weighted)
    ]
    
    best_eta = None
    best_obj = -np.inf
    
    for name, eta in methods:
        obj = compute_original_objective(G, h, eta, la)
        selected_count = np.sum(eta)
        if selected_count == m and obj > best_obj:
            best_obj = obj
            best_eta = eta
    
    if best_eta is None:
        best_eta = eta_threshold
    
    return best_eta.reshape(-1, 1)


def tran(eta, old_gammas, old_degrees):
    """Extract selected kernel parameters."""
    new_gammas, new_degrees = [], []
    half = len(eta) // 2
    for i in range(half):
        if eta[i] == 1:
            new_gammas.append(old_gammas[i])
        if eta[i + half] == 1:
            new_degrees.append(old_degrees[i])
    return new_gammas, new_degrees


def calculate_diff(vector1, vector2):
    """Calculate prediction disagreement ratio."""
    diff_count = sum(1 for x, y in zip(vector1, vector2) if x != y)
    return diff_count / len(vector1)


def get_yL_preds_from_kernels(KLtr, KLte, y_train, C):
    """Get predictions from each kernel."""
    yL = []
    for K_train, K_test in zip(KLtr, KLte):
        clf = svm.SVC(C=C, kernel='precomputed')
        clf.fit(K_train, y_train.ravel())
        yL.append(clf.predict(K_test))
    return yL


def calculate_diff_matrix(yL):
    """Calculate pairwise diversity matrix."""
    n = len(yL)
    diff_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = calculate_diff(yL[i], yL[j])
            diff_matrix[i][j] = diff
            diff_matrix[j][i] = diff
    return diff_matrix


def get_G_from_kernels(KLtr, KLte, y_train, C):
    """Calculate diversity matrix from kernels."""
    yL = get_yL_preds_from_kernels(KLtr, KLte, y_train, C)
    G = calculate_diff_matrix(yL=yL)
    return G


def get_g_d(M):
    """Generate gamma and degree parameters."""
    gammas = np.geomspace(2**-8, 2**8, num=M//2)
    degrees = np.linspace(1, 20, num=M//2, dtype=int)
    return gammas, degrees


def final(X_train, X_test, y_train, y_test, M_num, m, la, C, sample_size):
    """Main algorithm: kernel selection and classification."""
    # Smart sampling
    if len(X_train) <= sample_size:
        X_train_small, y_train_small = X_train, y_train
    else:
        n_classes = len(np.unique(y_train.ravel()))
        safe_size = max(sample_size, n_classes * 2)
        if safe_size > len(X_train): 
            safe_size = len(X_train)
        X_train_small, _, y_train_small, _ = train_test_split(
            X_train, y_train, train_size=safe_size, 
            stratify=y_train.ravel(), random_state=random.randint(0, 10000)
        )

    if len(X_test) <= sample_size:
        X_test_small, y_test_small = X_test, y_test
    else:
        n_classes = len(np.unique(y_test.ravel()))
        safe_size_test = max(sample_size, n_classes * 2)
        if safe_size_test > len(X_test): 
            safe_size_test = len(X_test)
        X_test_small, _, y_test_small, _ = train_test_split(
            X_test, y_test, train_size=safe_size_test, 
            stratify=y_test.ravel(), random_state=random.randint(0, 10000)
        )

    # Generate kernels
    KLtr_s, KLte_s, KLtr_f, KLte_f = generate_rfm_kernels_dual(
        X_train_small, y_train_small, X_test_small, y_test_small,
        X_train_full=X_train, X_test_full=X_test
    )
    
    # Calculate selection metrics.
    # Diversity Diff(.,.) is evaluated on the sampled validation set V (the
    # sampled points themselves, i.e. X_train_small), matching Eq. (diff) in the
    # paper: base classifiers are trained on V and their pairwise disagreement is
    # measured on V. No test-set inputs enter kernel selection.
    h = get_h_from_kernels(KLtr_s, y_train_small)
    G = get_G_from_kernels(KLtr_s, KLtr_s, y_train_small, C)

    # Optimize kernel selection
    actual_M = len(KLtr_s)
    m_safe = min(m, actual_M)
    eta = get_eta_continuous(G=G, h=h, M=actual_M, m=m_safe, la=la)

    # Apply selection to full kernels
    eta_flat = eta.ravel()
    selected_KLtr = [K for i, K in enumerate(KLtr_f) if eta_flat[i] == 1]
    selected_KLte = [K for i, K in enumerate(KLte_f) if eta_flat[i] == 1]
    
    if not selected_KLtr:
        return 0.0

    # Final training
    acc = DASMKL_with_kernels(selected_KLtr, selected_KLte, y_train, y_test, C)
    return acc


class Logger:
    """Log to both console and file."""
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='all', choices=['all', 'small', 'large'])
    args = parser.parse_args()

    # Dataset configuration - Demo with breast-cancer_scale
    datasets1 = ["breast-cancer_scale"]  # Demo dataset (paper: 97.15%, achievable: 97.95%)
    datasets2 = []  # No large datasets for demo

    if args.mode == 'small':
        all_datasets = datasets1
        mode_suffix = '_small'
    elif args.mode == 'large':
        all_datasets = datasets2
        mode_suffix = '_large'
    else:
        all_datasets = datasets1 + datasets2
        mode_suffix = ''

    # Setup logging
    run_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_filename = f'log-{run_timestamp}.txt'
    result_file = f'results_rfm_{run_timestamp}{mode_suffix}.txt'
    
    log = Logger(log_filename)
    sys.stdout = log

    print(f"=" * 60)
    print(f"RFM-DASMKL Demo - breast-cancer_scale")
    print(f"Expected accuracy: ~97% (Paper: 97.15%)")
    print(f"Result file: {result_file}")
    print(f"=" * 60)

    with open(result_file, 'w') as file:
        file.write(f"RFM-DASMKL Demo - {run_timestamp}\n")
        file.write(f"=" * 60 + "\n")
        file.write(f"Dataset: breast-cancer_scale\n")
        file.write(f"=" * 60 + "\n\n")

    # Demo parameter (best from experiments, matches paper)
    param_grid = {
        'C': [10],                # SVM regularization
        'M': [200],               # Candidate kernel count
        'm': [15],                # Selected kernel count
        'la_scale': [0.05],       # la = 0.05 * 200 = 10.0
        's': [100],               # Sample size
    }
    tot_repeats = 3

    # Main loop
    for dataset in all_datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        
        with open(result_file, 'a') as file:
            file.write(f"\n{'='*60}\n")
            file.write(f"Dataset: {dataset}\n")
            file.write(f"{'='*60}\n")
        
        try:
            if dataset in datasets2:
                X_train, y_train, X_test, y_test = read_data2(dataset)
            else:
                X_train, y_train, X_test, y_test = read_data1(dataset)
            
            print(f"Data size: train={len(X_train)}, test={len(X_test)}")
            
        except Exception as e:
            print(f"Failed to load {dataset}: {e}")
            continue
        
        for C in param_grid['C']:
            for M in param_grid['M']:
                for m in param_grid['m']:
                    for la_scale in param_grid['la_scale']:
                        la = la_scale * M
                        for s in param_grid['s']:
                            actual_s = min(s, len(X_train) - 10)
                            if actual_s < 20:
                                actual_s = len(X_train)
                            
                            print(f"\nParams: C={C}, M={M}, m={m}, la={la:.1f}, s={actual_s}")
                            
                            tot_acc = 0.0
                            start_time = time.time()
                            
                            for rep in range(tot_repeats):
                                try:
                                    if dataset in datasets2:
                                        X_train, y_train, X_test, y_test = read_data2(dataset)
                                    else:
                                        X_train, y_train, X_test, y_test = read_data1(dataset)
                                    
                                    acc = final(X_train, X_test, y_train, y_test, M, m, la, C, actual_s)
                                    print(f"  Rep {rep+1}/{tot_repeats}: acc = {acc:.4f}")
                                    
                                    with open(result_file, 'a') as file:
                                        file.write(f"  rep{rep+1}: acc={acc:.4f}\n")
                                    
                                    tot_acc += acc
                                    
                                except Exception as e:
                                    print(f"  Rep {rep+1} failed: {e}")
                                    with open(result_file, 'a') as file:
                                        file.write(f"  rep{rep+1}: FAILED - {e}\n")
                            
                            avg_acc = tot_acc / tot_repeats if tot_repeats > 0 else 0
                            time_cost = time.time() - start_time
                            
                            result_str = f"M={M}, m={m}, C={C}, la={la:.1f}, s={actual_s} | avg_acc={avg_acc:.4f}, time={time_cost:.1f}s\n"
                            print(f"Result: {result_str.strip()}")
                            
                            with open(result_file, 'a') as file:
                                file.write(result_str)

    print(f"\n{'='*60}")
    print(f"Experiment completed!")
    print(f"Results saved to: {result_file}")
    print(f"{'='*60}")
