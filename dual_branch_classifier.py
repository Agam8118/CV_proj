"""
╔══════════════════════════════════════════════════════════════════╗
║         DUAL-BRANCH DEFECT CLASSIFIER — NEU-DET Dataset          ║
║  Structural Geometry + LBP Texture | Random Forest + K-Fold CV   ║
╚══════════════════════════════════════════════════════════════════╝

Folder structure expected:
  NEU-DET/
    train/images/
      clean_metal/   
      patches/       
      scratches/     
    validation/images/
      clean_metal/
      patches/       
      scratches/     
"""

import os, sys, warnings, argparse, time, random
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
    accuracy_score, precision_recall_fscore_support,
)
import joblib

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
CFG = dict(
    base_dir    = ".",
    output_dir  = "output",
    img_size    = 128,

    # Branch A: Structural & Edge Tuning
    canny_low   = 60,    
    canny_high  = 140,   
    hough_thresh     = 35,
    hough_min_length = 25,   
    hough_max_gap    = 15,

    # Branch B: Texture Tuning (Strict Uniform Method)
    lbp_radius  = 2,     
    lbp_points  = 16,    
    lbp_bins    = 18,    # MUST be lbp_points + 2

    # PCA Compression
    pca_variance = 0.85, 

    # Random Forest Hyperparameters
    rf_estimators = 100,
    rf_max_depth  = 10,
)

LABEL_MAP   = {"scratches": 0, "patches": 1, "clean_metal": 2}
LABEL_NAMES = {0: "Scratch", 1: "Patch", 2: "Clean"}
CLASS_COLORS = {0: "#E74C3C", 1: "#3498DB", 2: "#2ECC71"}
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ══════════════════════════════════════════════════════════════════
#  PHASE 1 — STANDARDIZATION & AUGMENTATION
# ══════════════════════════════════════════════════════════════════
def standardize(img_path: str) -> np.ndarray:
    """Grayscale → 128×128 → Histogram Equalization."""
    raw = cv2.imread(img_path)
    if raw is None:
        raise FileNotFoundError(f"Cannot open: {img_path}")
    gray    = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw
    resized = cv2.resize(gray, (CFG["img_size"], CFG["img_size"]),
                         interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(resized)

def augment_image(img: np.ndarray) -> np.ndarray:
    """Randomly apply physical alterations to simulate factory conditions."""
    aug_img = img.copy()
    if random.choice([True, False]): aug_img = cv2.flip(aug_img, 1) # Horizontal
    if random.choice([True, False]): aug_img = cv2.flip(aug_img, 0) # Vertical

    # Brightness Adjustment (+/- 15%)
    brightness_shift = random.uniform(-0.15, 0.15)
    aug_img = aug_img.astype(np.float32) * (1.0 + brightness_shift)
    aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)

    # Gaussian Noise
    if random.choice([True, False]):
        noise = np.random.normal(0, 5, aug_img.shape).astype(np.float32)
        aug_img = np.clip(aug_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return aug_img


# ══════════════════════════════════════════════════════════════════
#  PHASE 2 — DUAL-BRANCH FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════
def branch_a(img: np.ndarray) -> np.ndarray:
    """Branch A — Structural & GLCM Features → array of 6 floats."""
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, CFG["canny_low"], CFG["canny_high"])
    
    # 1. Hough Lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, 
                            threshold=CFG["hough_thresh"], 
                            minLineLength=CFG["hough_min_length"], maxLineGap=CFG["hough_max_gap"])
    lc = int(len(lines)) if lines is not None else 0
    
    # 2. Edge Density
    edge_density = np.sum(edges > 0) / edges.size
    
    # 3. Max Aspect Ratio
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_ar = 0.0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 0 and h > 0:
            max_ar = max(max_ar, w/h, h/w)
                
    # 4 & 5. GLCM Homogeneity & Contrast
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    # 6. Global Pixel Variance
    pixel_variance = np.std(img)
    
    return np.array([lc, edge_density, max_ar, homogeneity, contrast, pixel_variance])


def branch_b(img: np.ndarray) -> np.ndarray:
    """Branch B — Patch Detector (LBP) → 18-dim normalised histogram."""
    lbp = local_binary_pattern(img, P=CFG["lbp_points"], R=CFG["lbp_radius"], method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=CFG["lbp_bins"], range=(0, CFG["lbp_bins"]))
    hist = hist.astype(np.float64)
    hist /= hist.sum() + 1e-9
    return hist


def extract_features(img: np.ndarray):
    return branch_a(img), branch_b(img)


# ══════════════════════════════════════════════════════════════════
#  DATASET BUILDER
# ══════════════════════════════════════════════════════════════════
def build_dataset(data_dir: str, label_map: dict, augment: bool = False):
    struct_feats, lbp_vecs, labels, paths = [], [], [], []
    for cls_name, cls_id in label_map.items():
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"    ⚠  Skipped (not found): {cls_dir}")
            continue
        files = [f for f in sorted(os.listdir(cls_dir)) if os.path.splitext(f)[1].lower() in VALID_EXT]
        print(f"    [{LABEL_NAMES[cls_id]:^11}] {len(files):>4} images  →  {cls_dir}")
        ok = err = 0
        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            try:
                img = standardize(fpath)
                if augment: img = augment_image(img)
                s_feat, lbp = extract_features(img)
                struct_feats.append(s_feat)
                lbp_vecs.append(lbp)
                labels.append(cls_id)
                paths.append(fpath)
                ok += 1
            except Exception as e:
                print(f"      ✗ {fname}: {e}")
                err += 1
    return (np.array(struct_feats), np.array(lbp_vecs), np.array(labels), paths)


# ══════════════════════════════════════════════════════════════════
#  PHASE 3 & 4 — PCA COMPRESSION & CSV GENERATION
# ══════════════════════════════════════════════════════════════════
def fit_pca(train_lbp, val_lbp=None):
    pca = PCA(n_components=CFG["pca_variance"], random_state=42)
    train_pca = pca.fit_transform(train_lbp)
    val_pca = pca.transform(val_lbp) if val_lbp is not None else None
    print(f"    PCA: {CFG['lbp_bins']}D → {train_pca.shape[1]}D  |  Variance retained: {pca.explained_variance_ratio_.sum() * 100:.2f}%")
    return pca, train_pca, val_pca

def save_csv(struct_feats, pca_feats, lbls, fpath):
    n_pca = pca_feats.shape[1]
    cols = ["Hough_Line_Count", "Edge_Density", "Max_Aspect_Ratio", 
            "GLCM_Homogeneity", "GLCM_Contrast", "Pixel_Variance"] + \
           [f"PCA_Texture_{i+1}" for i in range(n_pca)] + ["Target_Label"]
    data = np.hstack([struct_feats, pca_feats, lbls.reshape(-1, 1)])
    df = pd.DataFrame(data, columns=cols)
    df["Target_Label"] = df["Target_Label"].astype(int)
    df.to_csv(fpath, index=False)
    print(f"    CSV → {fpath}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    return df

def assemble_X(struct_feats, pca_feats):
    return np.hstack([struct_feats, pca_feats])


# ══════════════════════════════════════════════════════════════════
#  PHASE 5 — RANDOM FOREST TRAINING
# ══════════════════════════════════════════════════════════════════
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(
        n_estimators=CFG["rf_estimators"],
        max_depth=CFG["rf_max_depth"],
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    return scaler, model


# ══════════════════════════════════════════════════════════════════
#  PLOTS (Dashboard, ROC, Confusion Matrix)
# ══════════════════════════════════════════════════════════════════
plt.rcParams.update({"font.family": "monospace", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--"})

def plot_confusion(y_true, y_pred, cls_ids, title, out):
    cm = confusion_matrix(y_true, y_pred, labels=cls_ids)
    labels = [LABEL_NAMES[i] for i in cls_ids]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12); ax.set_yticklabels(labels, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=14, fontweight="bold", color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

def plot_metrics_dashboard(y_true, y_pred, y_prob_full, n_classes, present_ids, split_name, out):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=present_ids, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    per_auc = [auc(*roc_curve(y_bin[:, cid], y_prob_full[:, cid])[:2]) for cid in present_ids]
    
    labels = [LABEL_NAMES[i] for i in present_ids]
    x, width = np.arange(len(labels)), 0.2
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Metrics Dashboard — {split_name}  (Acc={acc:.3f})", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.bar(x - width, precision, width, label="Precision", color="#3498DB")
    ax.bar(x, recall, width, label="Recall", color="#E74C3C")
    ax.bar(x + width, f1, width, label="F1-Score", color="#2ECC71")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12); ax.set_ylim([0, 1.12])
    ax.legend(fontsize=11)

    ax2 = axes[1]
    bars = ax2.bar(labels, per_auc, color=[CLASS_COLORS[i] for i in present_ids], width=0.45)
    ax2.set_ylim([0, 1.12]); ax2.set_title("ROC AUC per Class", fontweight="bold")
    for bar, val in zip(bars, per_auc): ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════
#  INFERENCE ON EXTERNAL IMAGES
# ══════════════════════════════════════════════════════════════════
def predict_image(img_path: str, model_bundle: dict):
    scaler, model, pca = model_bundle["scaler"], model_bundle["model"], model_bundle["pca"]
    img = standardize(img_path)
    struct_feats, lbp = extract_features(img)
    lbp_pca = pca.transform(lbp.reshape(1, -1))
    
    feat = np.hstack([struct_feats.reshape(1, -1), lbp_pca])
    feat_sc = scaler.transform(feat)
    
    prob_vec = model.predict_proba(feat_sc)[0]
    prob_full = np.zeros(len(LABEL_MAP))
    for i, cls in enumerate(model.classes_): prob_full[cls] = prob_vec[i]
        
    pred_id = int(np.argmax(prob_full))

    print(f"\n  Image      : {img_path}")
    print(f"  Prediction : {LABEL_NAMES[pred_id]}  (class {pred_id})")
    print(f"  Confidence : {prob_full[pred_id]*100:.1f}%")
    print(f"  Hough Lines: {struct_feats[0]:.0f}")
    print(f"  Edge Dens. : {struct_feats[1]:.4f}")
    print(f"  Pixel Var. : {struct_feats[5]:.2f}")
    for cls_id, name in LABEL_NAMES.items():
        print(f"  P({name:^7}): {prob_full[cls_id]*100:.2f}%")
    return pred_id, prob_full


# ══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE (K-FOLD)
# ══════════════════════════════════════════════════════════════════
def _header(text):
    print(f"\n{'─'*60}\n  {text}\n{'─'*60}")

def main(predict_path=None):
    t0 = time.time()
    os.makedirs(CFG["output_dir"], exist_ok=True)
    out = CFG["output_dir"]
    N = len(LABEL_MAP)

    # ── PREDICTION MODE ──
    model_file = os.path.join(out, "dual_branch_rf.pkl")
    if predict_path:
        if not os.path.exists(model_file): sys.exit("ERROR: Model not found. Train first.")
        predict_image(predict_path, joblib.load(model_file))
        return

    # ── EXTRACTION ──
    _header("PHASE 1+2 — Standardisation + Feature Extraction")
    train_dir = os.path.join(CFG["base_dir"], "train", "images")
    val_dir   = os.path.join(CFG["base_dir"], "validation", "images")

    print("\n  [LOADING ALL DATA FOR K-FOLD]")
    tr_s, tr_lbp, tr_lbl, tr_paths = build_dataset(train_dir, LABEL_MAP, augment=False)
    
    # Pass LABEL_MAP to ensure clean_metal is included in validation load
    va_s, va_lbp, va_lbl, va_paths = build_dataset(val_dir, LABEL_MAP, augment=False)

    all_s = np.concatenate([tr_s, va_s])
    all_lbp = np.concatenate([tr_lbp, va_lbp])
    all_labels = np.concatenate([tr_lbl, va_lbl])
    print(f"\n  Total combined samples: {len(all_labels)}")

    # ── PCA ──
    _header("PHASE 3 — PCA Compression")
    pca, all_pca, _ = fit_pca(all_lbp, None)
    X_all = assemble_X(all_s, all_pca)

    # ── K-FOLD CV ──
    _header("PHASE 4+5 — Random Forest K-Fold Cross-Validation")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []
    best_acc, best_model, best_scaler = 0.0, None, None
    best_y_test, best_y_pred, best_y_prob = None, None, None

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, all_labels)):
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = all_labels[train_idx], all_labels[test_idx]
        
        scaler, model = train_model(X_tr, y_tr)
        
        X_te_sc = scaler.transform(X_te)
        y_pred = model.predict(X_te_sc)
        
        acc = accuracy_score(y_te, y_pred)
        fold_accs.append(acc)
        print(f"  Fold {fold_idx + 1} Accuracy: {acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc, best_model, best_scaler = acc, model, scaler
            best_y_test, best_y_pred = y_te, y_pred
            prob_raw = model.predict_proba(X_te_sc)
            best_y_prob = np.zeros((len(prob_raw), N))
            for i, cls in enumerate(model.classes_): best_y_prob[:, cls] = prob_raw[:, i]

    print(f"\n  >> Average K-Fold Accuracy: {np.mean(fold_accs)*100:.2f}%")

    # ── PLOTS & SAVING ──
    _header("GENERATING PLOTS (Best Fold)")
    
    # We pull the IDs from LABEL_MAP so it always expects [0, 1, 2]
    present_ids = sorted(LABEL_MAP.values())
    
    plot_confusion(best_y_test, best_y_pred, present_ids, "Confusion Matrix (Best Fold)", os.path.join(out, "cm_best_fold.png"))
    plot_metrics_dashboard(best_y_test, best_y_pred, best_y_prob, N, present_ids, "Best Fold", os.path.join(out, "dashboard_best_fold.png"))

    joblib.dump({"scaler": best_scaler, "model": best_model, "pca": pca, "label_map": LABEL_NAMES}, model_file)
    print(f"\n    Model saved → {model_file}")
    print(f"  Total runtime: {time.time() - t0:.1f}s\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", type=str, default=None)
    main(predict_path=parser.parse_args().predict)