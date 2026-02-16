import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import lpips
import sys
import os
from tools import build_disc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "models")))
import squid
from dataloader.dataloader_karma4 import KarmaDataset
from configs.karma4_dev import Config

def anomaly_score(mse, lpips_s, disc_s, recon_w, lpips_w, d_w):
    return recon_w * mse + lpips_w * lpips_s + d_w * (1.0 - disc_s)

def main():
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    model_path = "checkpoints/karma_16_experiment/model_epoch1250.pth"
    disc_path = "checkpoints/karma_16_experiment/discriminator_epoch1250.pth"
    out_dir = Path("eval_full_output"); out_dir.mkdir(exist_ok=True)

    # Model
    model = squid.AE(cfg, features_root=32, level=cfg.level).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Discriminator
    disc = build_disc(cfg).to(device)
    disc.load_state_dict(torch.load(disc_path, map_location=device, weights_only=True))
    disc.eval()

    # LPIPS
    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()

    # DataLoader
    test_set = KarmaDataset(root='data/karma4/test', train=False, img_size=(128,128))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Değişkenler
    all_scores, all_labels, all_inputs, all_recons, all_diffs, all_overlays, all_anomaps = [], [], [], [], [], [], []

    for img, label in test_loader:
        img = img.to(device)
        img_norm = img * 2.0 - 1.0
        with torch.no_grad():
            out = model(img_norm)
            recon = out['recon']
            disc_score = disc(recon).view(-1)

        mse = (img_norm - recon).pow(2).mean(dim=[1,2,3])
        lpips_score = lpips_fn(recon, img_norm).view(-1)
        score = anomaly_score(
            mse.cpu().detach().numpy(),
            lpips_score.cpu().detach().numpy(),
            disc_score.cpu().detach().numpy(),
            recon_w=cfg.recon_w,
            lpips_w=cfg.lpips_w,
            d_w=cfg.d_w
        )

        all_scores.append(score[0])
        all_labels.append(label.item())

        diff = (img_norm - recon).abs()
        overlay = 0.7 * img_norm + 0.3 * diff
        anomap = diff.squeeze().cpu().numpy()

        all_inputs.append(img_norm[0].cpu())
        all_recons.append(recon[0].cpu())
        all_diffs.append(diff[0].cpu())
        all_overlays.append(overlay[0].cpu())
        all_anomaps.append(anomap)

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    all_inputs = torch.stack(all_inputs)
    all_recons = torch.stack(all_recons)
    all_diffs = torch.stack(all_diffs)
    all_overlays = torch.stack(all_overlays)
    all_anomaps = np.stack(all_anomaps)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(labels, scores)
    prc_auc = auc(rec, prec)
    youden_idx = np.argmax(tpr - fpr)
    best_thr = thresholds[youden_idx] if len(thresholds) > youden_idx else np.inf
    preds = (scores >= best_thr).astype(int)
    cm = confusion_matrix(labels, preds)

    # ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}"); plt.legend(); plt.title("ROC Curve")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.savefig(out_dir / "roc.png"); plt.close()

    # PRC
    plt.figure()
    plt.step(rec, prec, label=f"AUC={prc_auc:.3f}"); plt.legend(); plt.title("PRC Curve")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.savefig(out_dir / "prc.png"); plt.close()

    # Confusion Matrix
    plt.figure()
    plt.imshow(cm, cmap=plt.cm.Blues)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.xticks([0,1],["Normal","Anomali"]); plt.yticks([0,1],["Normal","Anomali"])
    plt.xlabel("Tahmin"); plt.ylabel("Gerçek"); plt.title("Confusion Matrix")
    plt.savefig(out_dir/"confmat.png"); plt.close()

    metrikler = {
        "roc_auc": float(roc_auc),
        "prc_auc": float(prc_auc),
        "youden_threshold": float(best_thr),
        "accuracy": float((cm[0,0] + cm[1,1]) / cm.sum()),
        "precision": float(cm[1,1] / (cm[0,1] + cm[1,1] + 1e-8)),
        "recall": float(cm[1,1] / (cm[1,0] + cm[1,1] + 1e-8)),
        "f1_score": float(2 * cm[1,1] / (2 * cm[1,1] + cm[0,1] + cm[1,0] + 1e-8)),
        "confusion_matrix": cm.tolist()
    }
    import json
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrikler, f, indent=2)
    print(json.dumps(metrikler, indent=2))

if __name__ == "__main__":
    main()
