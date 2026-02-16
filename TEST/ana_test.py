import os
import numpy as np
import torch
import lpips
import shutil
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import importlib
import json
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.special import expit   # sigmoid için

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
RESULTS_BASE = r"C:\Users\BURKINEFASO\LPIPS_SIMSID\BULK_TEST_RESULTS"
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

def plot_histogram(scores, labels, save_dir):
    plt.figure(figsize=(8, 6))
    sns.histplot(scores[labels == 0], color='b', label='Normal', kde=True, stat='density')
    sns.histplot(scores[labels == 1], color='r', label='Anomali', kde=True, stat='density')
    plt.legend()
    plt.title("Anomali Skoru Dağılımı")
    plt.xlabel("Skor")
    plt.ylabel("Yoğunluk")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "histogram.png"))
    plt.close()

def plot_sigmoid_histogram(scores, labels, save_dir):
    scores_sigmoid = expit(scores)
    plt.figure(figsize=(8, 6))
    sns.histplot(scores_sigmoid[labels == 0], color='b', label='Normal', kde=True, stat='density')
    sns.histplot(scores_sigmoid[labels == 1], color='r', label='Anomali', kde=True, stat='density')
    plt.legend()
    plt.title("Anomali Skoru Dağılımı")
    plt.xlabel(" Anomali Skoru")
    plt.ylabel("Yoğunluk")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sigmoid_histogram.png"))
    plt.close()

def plot_roc(y_true, scores, save_dir):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc_ = roc_auc_score(y_true, scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc_:.2f}", color='darkred', linewidth=2)
    plt.plot([0, 1], [0, 1], "--k")
    plt.xlabel("Yanlış Pozitif Oranı (1 - Özgüllük)", fontsize=12)
    plt.ylabel("Doğru Pozitif Oranı (Duyarlılık)", fontsize=12)
    plt.legend()
    plt.title("ROC Eğrisi", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc.png"))
    plt.close()

def plot_prc(y_true, scores, save_dir):
    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, _ = precision_recall_curve(y_true, scores)
    prc_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f"PRC AUC={prc_auc:.2f}", color='g')
    plt.xlabel("Duyarlılık")
    plt.ylabel("Kesinlik")
    plt.legend()
    plt.title("PRC Eğrisi")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prc.png"))
    plt.close()

def plot_cm(y_true, preds, save_dir):
    cm = confusion_matrix(y_true, preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Anomali"], yticklabels=["Normal", "Anomali"])
    plt.title("Karışıklık Matrisi")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cm.png"))
    plt.close()

def plot_kde(scores, labels, save_dir):
    plt.figure()
    sns.kdeplot(scores[labels==0], color="blue", label="Normal")
    sns.kdeplot(scores[labels==1], color="red", label="Anomali")
    plt.legend()
    plt.title("KDE Dağılımı")
    plt.xlabel("Skor")
    plt.ylabel("Yoğunluk")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "kde.png"))
    plt.close()

def plot_dim_red(features, labels, save_dir):
    # tSNE
    tsne = TSNE(n_components=2, perplexity=40, n_iter=3000, learning_rate=300, random_state=42)
    X_tsne = tsne.fit_transform(features)
    plt.figure()
    plt.scatter(X_tsne[labels==0,0], X_tsne[labels==0,1], color='blue', label='Normal', alpha=0.5)
    plt.scatter(X_tsne[labels==1,0], X_tsne[labels==1,1], color='red', label='Anomali', alpha=0.5)
    plt.xlabel("Bileşen 1")
    plt.ylabel("Bileşen 2")
    plt.title("t-SNE Dağılımı")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"tsne.png"))
    plt.close()

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features)
    plt.figure()
    plt.scatter(X_pca[labels==0,0], X_pca[labels==0,1], color='blue', label='Normal', alpha=0.5)
    plt.scatter(X_pca[labels==1,0], X_pca[labels==1,1], color='red', label='Anomali', alpha=0.5)
    plt.xlabel("Bileşen 1")
    plt.ylabel("Bileşen 2")
    plt.title("PCA Dağılımı")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"pca.png"))
    plt.close()

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
            # Dataloader'ı yeniden başlat!
            config_mod = importlib.reload(importlib.import_module(f'configs.{CONFIG_PATH}'))
            test_loader = config_mod.Config().test_loader
            # Test ve skor
            scores, labels, features = eval_scores_lpips(model, lpips_fn, train_stats, DEVICE, test_loader)
            preds = (scores >= threshold).astype(int)
            auc = roc_auc_score(labels, scores)
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
                "auc_roc": float(auc), "auc_prc": float(prc),
                "accuracy": float(acc), "f1": float(f1),
                "precision": float(prec), "recall": float(rec),
                "threshold": float(threshold),
                "confusion_matrix": cm.tolist(),
                "TP": int(cm[1, 1]), "TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0])
            }
            combo_results.append(metrics)
            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            plot_histogram(scores, labels, out_dir)
            plot_sigmoid_histogram(scores, labels, out_dir)  # <-- Sigmoid histogram eklendi!
            plot_kde(scores, labels, out_dir)
            plot_roc(labels, scores, out_dir)
            plot_prc(labels, scores, out_dir)
            plot_cm(labels, preds, out_dir)
            plot_dim_red(features, labels, out_dir)
            df = pd.DataFrame({
                "label": np.array(labels).flatten(),
                "score": np.array(scores).flatten()
            })
            df.to_csv(os.path.join(out_dir, "test_scores.csv"), index=False)
            print(f"   -> [OK] AUC: {auc:.3f} | ACC: {acc:.3f} | F1: {f1:.3f}")
            combo_id += 1

    pd.DataFrame(combo_results).to_csv(os.path.join(RESULTS_BASE, "bulk_test_summary.csv"), index=False)
    print(f"\nTüm {total_combos} kombinasyon tamamlandı! Sonuçlar: {RESULTS_BASE}")

if __name__ == "__main__":
    main()
