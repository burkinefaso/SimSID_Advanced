import os
import numpy as np
import torch
import lpips
import shutil
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import importlib
import json
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.special import expit   # sigmoid için
from sklearn.cluster import KMeans
import umap

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# --- Sabitler ---
NORMAL_ROOT = r"C:\Users\BURKINEFASO\LPIPS_SIMSID\data\karma4\test\DIGER\ALL_NORMAL"
ANOMALY_ROOT = r"C:\Users\BURKINEFASO\LPIPS_SIMSID\data\karma4\test\DIGER\ALL_ANOMALY"
TEST_NORMAL_DIR = r"C:\Users\BURKINEFASO\LPIPS_SIMSID\data\karma4\test\NORMAL"
TEST_ANOMALY_DIR = r"C:\Users\BURKINEFASO\LPIPS_SIMSID\data\karma4\test\ANOMALY"
CHECKPOINT_PATH = r"C:\Users\BURKINEFASO\LPIPS_SIMSID\checkpoints\karma_17_experiment\model_epoch800.pth"
CONFIG_PATH = "karma4_dev"
TRAIN_STATS_PATH = "train_score_stats.npz"
VAL_METRICS_PATH = r"C:\Users\BURKINEFASO\LPIPS_SIMSID\VAL_LPIPS_LPIPS_ONLY\metrics_sigmoid.json"
RESULTS_BASE = r"C:\Users\BURKINEFASO\LPIPS_SIMSID\FRIKKO_TEST_RESULTS"
NORM_PER_SET = 1000
ANOM_PER_SET = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(RESULTS_BASE, exist_ok=True)

CONFIG = importlib.import_module(f'configs.{CONFIG_PATH}').Config()
model = importlib.import_module('models.squid').AE(CONFIG, 32, level=CONFIG.level).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE), strict=False)
model.eval()
lpips_fn = lpips.LPIPS(net='alex').to(DEVICE).eval()

def load_train_stats(stats_path):
    stats = dict(np.load(stats_path))
    stats = {k: float(v) for k, v in stats.items()}
    return stats
train_stats = load_train_stats(TRAIN_STATS_PATH)

def load_validation_threshold(path):
    with open(path, "r") as f:
        metrics = json.load(f)
    return metrics["threshold"]

def pick_random_images(src_folder, n):
    files = [os.path.join(src_folder, f) for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    selected = random.sample(files, min(n, len(files)))
    return selected

def clear_and_copy(files, dest_folder):
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    os.makedirs(dest_folder)
    for f in files:
        shutil.copy(f, dest_folder)

def eval_scores_lpips(model, lpips_fn, stats, device, loader):
    model.eval()
    lpips_list = []
    labels_list = []
    features_list = []
    for imgs, labels in tqdm(loader, desc="Test verisi inferans"):
        imgs = imgs.to(device)
        labels = labels.cpu().numpy().flatten()
        with torch.no_grad():
            out = model(imgs)
            recons = out["recon"]
            lp = lpips_fn(recons, imgs).view(-1).detach().cpu().numpy()
            lpips_list.extend(lp)
            features = out["embeddings"][-1].view(imgs.size(0), -1).cpu().numpy()
            features_list.append(features)
        labels_list.extend(labels)
    lpips_norm = (np.array(lpips_list) - stats["lpips_mean"]) / (stats["lpips_std"] + 1e-8)
    features_all = np.concatenate(features_list, axis=0)
    return lpips_norm, np.array(labels_list).flatten(), features_all

# ----------- YENİ: GÖRSELLEŞTİRME FONKSİYONLARI -----------
def plot_histogram_kde(scores, labels, save_dir):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(scores[labels == 0], color='b', label='Normal', kde=False, stat='density', bins=30, alpha=0.7)
    sns.histplot(scores[labels == 1], color='r', label='Anomali', kde=False, stat='density', bins=30, alpha=0.7)
    plt.title("(a) Anomali Skoru Histogramı")
    plt.xlabel("Skor"); plt.ylabel("Yoğunluk")
    plt.legend()
    plt.subplot(1,2,2)
    sns.kdeplot(scores[labels==0], color="blue", label="Normal", fill=True)
    sns.kdeplot(scores[labels==1], color="red", label="Anomali", fill=True)
    plt.title("(b) KDE")
    plt.xlabel("Skor"); plt.ylabel("Yoğunluk")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "histogram_kde_combo.png"))
    plt.close()

def plot_roc_prc_cm(y_true, scores, save_dir, threshold):
    preds = (scores >= threshold).astype(int)
    # ROC
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    # PRC
    precision, recall, _ = precision_recall_curve(y_true, scores)
    prc_auc = auc(recall, precision)
    # CM
    cm = confusion_matrix(y_true, preds)
    fig, axs = plt.subplots(1,3,figsize=(16,5))
    axs[0].plot(fpr, tpr, color="darkred", linewidth=2)
    axs[0].plot([0,1],[0,1],"--k")
    axs[0].set_title("(a) ROC\nAUC={:.2f}".format(roc_auc)); axs[0].set_xlabel("FPR"); axs[0].set_ylabel("TPR")
    axs[1].plot(recall, precision, color="green", linewidth=2)
    axs[1].set_title("(b) PRC\nAUC={:.2f}".format(prc_auc)); axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Anomali"], yticklabels=["Normal", "Anomali"], ax=axs[2])
    axs[2].set_title("(c) Karışıklık Matrisi")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_prc_cm_combo.png"))
    plt.close()

def plot_umap_kmeans_pca_tsne(features, labels, save_dir):
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(features)
    kmeans = KMeans(n_clusters=2, random_state=42).fit(X_umap)
    kmeans_labels = kmeans.labels_
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features)
    tsne = TSNE(n_components=2, perplexity=40, n_iter=3000, learning_rate=300, random_state=42)
    X_tsne = tsne.fit_transform(features)
    fig, axs = plt.subplots(2,2,figsize=(14,12))
    axs[0,0].scatter(X_umap[labels==0,0], X_umap[labels==0,1], s=10, c="blue", alpha=0.6, label="Normal")
    axs[0,0].scatter(X_umap[labels==1,0], X_umap[labels==1,1], s=10, c="red", alpha=0.6, label="Anomali")
    axs[0,0].set_title("(a) UMAP", fontsize=13)
    axs[0,0].legend()
    axs[0,1].scatter(X_umap[:,0], X_umap[:,1], c=kmeans_labels, cmap="coolwarm", alpha=0.7, s=10)
    axs[0,1].set_title("(b) KMeans (UMAP)", fontsize=13)
    axs[1,0].scatter(X_pca[labels==0,0], X_pca[labels==0,1], s=10, c="blue", alpha=0.6, label="Normal")
    axs[1,0].scatter(X_pca[labels==1,0], X_pca[labels==1,1], s=10, c="red", alpha=0.6, label="Anomali")
    axs[1,0].set_title("(c) PCA", fontsize=13)
    axs[1,0].legend()
    axs[1,1].scatter(X_tsne[labels==0,0], X_tsne[labels==0,1], s=10, c="blue", alpha=0.6, label="Normal")
    axs[1,1].scatter(X_tsne[labels==1,0], X_tsne[labels==1,1], s=10, c="red", alpha=0.6, label="Anomali")
    axs[1,1].set_title("(d) t-SNE", fontsize=13)
    axs[1,1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "umap_kmeans_pca_tsne_combo.png"))
    plt.close()
    # Ayrıca hepsini ayrı ayrı da kaydet
    plt.figure(figsize=(6,5))
    plt.scatter(X_umap[labels==0,0], X_umap[labels==0,1], s=10, c="blue", alpha=0.6, label="Normal")
    plt.scatter(X_umap[labels==1,0], X_umap[labels==1,1], s=10, c="red", alpha=0.6, label="Anomali")
    plt.title("UMAP")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "umap.png")); plt.close()
    plt.figure(figsize=(6,5))
    plt.scatter(X_umap[:,0], X_umap[:,1], c=kmeans_labels, cmap="coolwarm", alpha=0.7, s=10)
    plt.title("KMeans (UMAP)"); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "kmeans_umap.png")); plt.close()
    plt.figure(figsize=(6,5))
    plt.scatter(X_pca[labels==0,0], X_pca[labels==0,1], s=10, c="blue", alpha=0.6, label="Normal")
    plt.scatter(X_pca[labels==1,0], X_pca[labels==1,1], s=10, c="red", alpha=0.6, label="Anomali")
    plt.title("PCA")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca.png")); plt.close()
    plt.figure(figsize=(6,5))
    plt.scatter(X_tsne[labels==0,0], X_tsne[labels==0,1], s=10, c="blue", alpha=0.6, label="Normal")
    plt.scatter(X_tsne[labels==1,0], X_tsne[labels==1,1], s=10, c="red", alpha=0.6, label="Anomali")
    plt.title("t-SNE")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "tsne.png")); plt.close()

# ----------- MAIN FONKSİYON -----------
def main():
    threshold = load_validation_threshold(VAL_METRICS_PATH)
    normal_folders = sorted([os.path.join(NORMAL_ROOT, x) for x in os.listdir(NORMAL_ROOT) if os.path.isdir(os.path.join(NORMAL_ROOT, x))])
    anom_folders = sorted([os.path.join(ANOMALY_ROOT, x) for x in os.listdir(ANOMALY_ROOT) if os.path.isdir(os.path.join(ANOMALY_ROOT, x))])
    total_combos = len(normal_folders) * len(anom_folders)
    combo_results = []
    print(f"Toplam {len(normal_folders)} normal x {len(anom_folders)} anomalili = {total_combos} kombinasyon denenecek.\n")
    combo_id = 1
    for norm_path in normal_folders:
        norm_name = os.path.basename(norm_path)
        norm_imgs = pick_random_images(norm_path, NORM_PER_SET)
        for anom_path in anom_folders:
            anom_name = os.path.basename(anom_path)
            anom_imgs = pick_random_images(anom_path, ANOM_PER_SET)
            print(f"\n[{combo_id}/{total_combos}] Kombinasyon: NORMAL={norm_name} | ANOMALİ={anom_name}")
            clear_and_copy(norm_imgs, TEST_NORMAL_DIR)
            clear_and_copy(anom_imgs, TEST_ANOMALY_DIR)
            config_mod = importlib.reload(importlib.import_module(f'configs.{CONFIG_PATH}'))
            test_loader = config_mod.Config().test_loader
            scores, labels, features = eval_scores_lpips(model, lpips_fn, train_stats, DEVICE, test_loader)
            preds = (scores >= threshold).astype(int)
            auc_ = roc_auc_score(labels, scores)
            prc = average_precision_score(labels, scores)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds)
            prec = precision_score(labels, preds, zero_division=0)
            rec = recall_score(labels, preds)
            cm = confusion_matrix(labels, preds)
            out_dir = os.path.join(RESULTS_BASE, f"{norm_name}__{anom_name}")
            os.makedirs(out_dir, exist_ok=True)
            metrics = {
                "normal": norm_name, "anomaly": anom_name,
                "auc_roc": float(auc_), "auc_prc": float(prc),
                "accuracy": float(acc), "f1": float(f1),
                "precision": float(prec), "recall": float(rec),
                "threshold": float(threshold),
                "confusion_matrix": cm.tolist(),
                "TP": int(cm[1, 1]), "TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0])
            }
            combo_results.append(metrics)
            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            df = pd.DataFrame({
                "label": np.array(labels).flatten(),
                "score": np.array(scores).flatten()
            })
            df.to_csv(os.path.join(out_dir, "test_scores.csv"), index=False)
            # --- YENİ EKLENEN GÖRSELLEŞTİRME ÇAĞRILARI ---
            plot_histogram_kde(scores, labels, out_dir)
            plot_roc_prc_cm(labels, scores, out_dir, threshold)
            plot_umap_kmeans_pca_tsne(features, labels, out_dir)
            print(f"   -> [OK] AUC: {auc_:.3f} | ACC: {acc:.3f} | F1: {f1:.3f}")
            combo_id += 1
    pd.DataFrame(combo_results).to_csv(os.path.join(RESULTS_BASE, "bulk_test_summary.csv"), index=False)
    print(f"\nTüm {total_combos} kombinasyon tamamlandı! Sonuçlar: {RESULTS_BASE}")

if __name__ == "__main__":
    main()
