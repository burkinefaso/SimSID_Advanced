import torch
torch.set_printoptions(10)
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
import time
from torch.utils.tensorboard import SummaryWriter
from models import squid
from models import hierarchical_memory
from models.memory import MemoryQueue
import random
import importlib
from tqdm import tqdm
from tools import parse_args, build_disc, log, log_loss, save_image, backup_files
from alert import GanAlert
import lpips  # LPIPS için
from sklearn.metrics import roc_auc_score
import json
from tools import save_tensor_as_image
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

args = parse_args()
CONFIG = importlib.import_module('configs.'+args.config).Config()

if not os.path.exists(os.path.join('checkpoints', args.exp)):
    os.makedirs(os.path.join('checkpoints', args.exp), exist_ok=True)
if not os.path.exists(os.path.join('checkpoints', args.exp, 'test_images')):
    os.makedirs(os.path.join('checkpoints', args.exp, 'test_images'), exist_ok=True)
save_path = os.path.join('checkpoints', args.exp, 'test_images')
log_file = open(os.path.join('checkpoints', args.exp, 'log.txt'), 'w')
writer = SummaryWriter(log_dir=os.path.join('checkpoints', args.exp))
backup_files(args)

model = squid.AE(CONFIG, 32, level=CONFIG.level).cuda()
opt = CONFIG.opt(model.parameters(), lr=CONFIG.lr, eps=1e-7, betas=(0.5, 0.999), weight_decay=0.00001)
scheduler = CONFIG.scheduler(opt, **CONFIG.scheduler_args)

discriminator = None
opt_d = None
if (CONFIG.enbale_gan is not None and CONFIG.enbale_gan >= 0):
    discriminator = build_disc(CONFIG).cuda()
    opt_d = CONFIG.opt(discriminator.parameters(), betas=(0.5, 0.999), lr=CONFIG.gan_lr)

ce = nn.BCEWithLogitsLoss().cuda()
recon_criterion = torch.nn.MSELoss(reduction='mean').cuda()
lpips_fn = lpips.LPIPS(net='alex').cuda()
alert = GanAlert(discriminator=discriminator, args=args, CONFIG=CONFIG, generator=model)
def main():
    best_auc = -1
    best_epoch = 0

    # Varsayılan olarak 1. epoch'tan başla
    resume_epoch = 1
    resume_model_path = os.path.join("checkpoints", args.exp, "model.pth")

    # Eğer daha önce kayıtlı bir model varsa, ordan devam et
    if os.path.exists(resume_model_path):
        print(f"[INFO] Kayıtlı model bulundu. Kaldığı yerden devam ediliyor: {resume_model_path}")
        checkpoint = torch.load(resume_model_path)
        model.load_state_dict(checkpoint)
        # epoch bilgisini log dosyasından ya da JSON'dan çekmek daha iyi ama burada elle güncelleyebilirsin.
        # Örneğin son epoch bilgisini bir json ile kaydetmiş olabilirsin.
        with open(os.path.join("checkpoints", args.exp, "validation_auc_log.json"), "r") as f:
            lines = f.readlines()
            if lines:
                last_entry = json.loads(lines[-1])
                resume_epoch = last_entry["epoch"] + 1

    if CONFIG.enbale_gan is not None and discriminator is not None:
        disc_path = os.path.join("checkpoints", args.exp, "discriminator.pth")
        if os.path.exists(disc_path):
            print(f"[INFO] Discriminator yükleniyor: {disc_path}")
            discriminator.load_state_dict(torch.load(disc_path))

    # Ana eğitim döngüsü
    for epoch in range(resume_epoch, CONFIG.epochs + 1):
        print(f"\n[INFO] Epoch {epoch} başlıyor...")

        if CONFIG.enbale_gan is None or epoch < CONFIG.enbale_gan:
            train_loss = train(CONFIG.train_loader, epoch)
        else:
            train_loss = gan_train(CONFIG.train_loader, epoch, writer)

        scheduler.step()

        # Validation ve görseller sadece her 5 epoch'ta bir yapılacak
        if epoch % 5 == 0:
            model.eval()
            all_scores, all_labels = [], []
            saved_anom = saved_normal = 0
            input_anom, recon_anom = [], []
            input_normal, recon_normal = [], []

            with torch.no_grad():
                for img, label in tqdm(CONFIG.val_loader, desc=f"Validation {epoch}"):
                    img = img.cuda()
                    label = label.cuda()
                    recon = model(img)["recon"]

                    mse = (img - recon).pow(2).mean(dim=[1, 2, 3])
                    lpips_score = lpips_fn(recon, img).view(-1) if CONFIG.lpips_w > 0 else torch.zeros_like(mse)
                    score = CONFIG.recon_w * mse + CONFIG.lpips_w * lpips_score
                    all_scores.extend(score.cpu().numpy())
                    all_labels.extend(label.cpu().numpy())

                    for i in range(img.shape[0]):
                        if label[i] == 1 and saved_anom < 50:  # anomali
                            input_anom.append(img[i].cpu())
                            recon_anom.append(recon[i].cpu())
                            saved_anom += 1
                        elif label[i] == 0 and saved_normal < 50:  # normal
                            input_normal.append(img[i].cpu())
                            recon_normal.append(recon[i].cpu())
                            saved_normal += 1

            # Görsel kaydı
            save_dir = os.path.join("checkpoints", args.exp, "test_images", f"epoch_{epoch:03d}")
            os.makedirs(save_dir, exist_ok=True)
            save_image(save_dir, input_normal, prefix="input_normal")
            save_image(save_dir, recon_normal, prefix="recon_normal")
            save_image(save_dir, input_anom, prefix="input_anom")
            save_image(save_dir, recon_anom, prefix="recon_anom")

            # AUC hesapla
            try:
                auc = roc_auc_score(all_labels, all_scores)
                print(f"[INFO] Epoch {epoch} AUC: {auc:.4f}")

                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join("checkpoints", args.exp, "model.pth"))
                    if CONFIG.enbale_gan is not None and discriminator is not None:
                        torch.save(discriminator.state_dict(), os.path.join("checkpoints", args.exp, "discriminator.pth"))
                    print("[INFO] Yeni en iyi model kaydedildi.")

                with open(os.path.join("checkpoints", args.exp, "validation_auc_log.json"), "a") as f:
                    json.dump({"epoch": epoch, "auc": auc}, f)
                    f.write("\n")

            except:
                print("[WARN] AUC hesaplanamadı. Muhtemelen validation etiketleri eksik.")

            # Checkpoint yedekle
            if epoch % 50 == 0:
                torch.save(model.state_dict(), os.path.join("checkpoints", args.exp, f"model_epoch{epoch}.pth"))
                if CONFIG.enbale_gan is not None and discriminator is not None:
                    torch.save(discriminator.state_dict(), os.path.join("checkpoints", args.exp, f"discriminator_epoch{epoch}.pth"))

    print(f"[INFO] Eğitim tamamlandı. En iyi epoch: {best_epoch}, AUC: {best_auc:.4f}")
    log_file.close()
    writer.close()
def train(dataloader, epoch):
    model.train()
    batches_done = 0
    tot_loss = {'recon_loss': 0., 'lpips_loss': 0., 'g_loss': 0., 'd_loss': 0., 't_recon_loss': 0., 'dist_loss': 0.}
    if CONFIG.limit is None:
        limit = len(dataloader) - len(dataloader) % CONFIG.n_critic
    else:
        limit = CONFIG.limit

    for i, (img, label) in enumerate(tqdm(dataloader, desc=f"val(): Epoch {epoch}", leave=True)):
        if i > limit:
            break
        batches_done += 1
        img = img.to(CONFIG.device)
        label = label.to(CONFIG.device)
        opt.zero_grad()
        out = model(img)

        if CONFIG.alert is not None:
            CONFIG.alert.record(out['recon'].detach(), img)

        mse_loss = recon_criterion(out["recon"], img)
        lpips_loss = lpips_fn(out["recon"], img).mean()
        loss_all = CONFIG.recon_w * mse_loss + CONFIG.lpips_w * lpips_loss
        tot_loss['recon_loss'] += mse_loss.item()
        tot_loss['lpips_loss'] = lpips_loss.item()

        if  CONFIG.dist and 'dist_loss' in out and torch.is_tensor(out['dist_loss']):
            dist_loss = CONFIG.dist_w  * out["dist_loss"]
            loss_all = loss_all + dist_loss
            tot_loss['dist_loss'] += dist_loss.item()

        loss_all.backward()
        opt.step()

        for module in model.modules():
            if isinstance(module, MemoryQueue):
                module.update()

    for k, v in tot_loss.items():
        tot_loss[k] /= batches_done
    return tot_loss

def gan_train(dataloader, epoch, writer):
    model.train()
    batches_done = 0
    tot_loss = {'loss': 0., 'recon_loss': 0., 'lpips_loss': 0., 'g_loss': 0., 'd_loss': 0., 't_recon_loss': 0., 'dist_loss': 0.}
    if CONFIG.limit is None:
        limit = len(dataloader) - len(dataloader) % CONFIG.n_critic
    else:
        limit = CONFIG.limit

    fadein_weights = [0.0 for _ in range(CONFIG.level)]
    if epoch > 200:
        fadein_weights[-1] += min((epoch - 200) / (400 - 200), 1.0)
    if epoch > 400:
        fadein_weights[-2] += min((epoch - 400) / (600 - 400), 1.0)
    fadein_weights = [0.0, 0.0, 1.0, 1.0]
    print(f'Epoch {epoch}, fadein weights {fadein_weights}')

    for i, (img, label) in enumerate(tqdm(dataloader, disable=False)):
        if i > limit:
            break
        batches_done += 1
        iter_start = time.time()
        img = img.to(CONFIG.device)
        label = label.to(CONFIG.device)
        opt_d.zero_grad()
        out = model(img, fadein_weights=fadein_weights)
        ae_time = time.time()
        real_validity = discriminator(img)
        fake_validity = discriminator(out["recon"].detach())
        disc_time = time.time()
        d_loss = ce(real_validity, torch.ones_like(real_validity))
        d_loss += ce(fake_validity, torch.zeros_like(fake_validity))
        d_loss *= CONFIG.d_w
        d_loss.backward()
        summary_grads(discriminator, 'discriminator', writer, epoch)
        opt_d.step()
        tot_loss['d_loss'] += d_loss.item()
        disc_bw_time = time.time()

        if i % CONFIG.n_critic == 0:
            if CONFIG.alert is not None:
                CONFIG.alert.record(out['recon'].detach(), img)
            mse_loss = recon_criterion(out["recon"], img)
            lpips_loss = lpips_fn(out["recon"], img).mean()
            recon_loss = CONFIG.recon_w * mse_loss + CONFIG.lpips_w * lpips_loss
            tot_loss['recon_loss'] += mse_loss.item()
            tot_loss['lpips_loss'] += lpips_loss.item()
            loss_all = recon_loss
            fake_validity = discriminator(out["recon"])
            g_loss = CONFIG.g_w * ce(fake_validity, torch.ones_like(fake_validity))
            tot_loss['g_loss'] += g_loss.item()
            loss_all = loss_all + g_loss
            if  CONFIG.dist and 'dist_loss' in out and torch.is_tensor(out['dist_loss']):
                dist_loss = CONFIG.dist_w * out["dist_loss"]
                tot_loss['dist_loss'] += dist_loss.item()
                loss_all = loss_all + dist_loss
            tot_loss['loss'] += loss_all.item()
            opt.zero_grad()
            loss_all.backward()
            summary_grads(model, 'generator', writer, epoch)
            opt.step()
            for module in model.modules():
                if isinstance(module, MemoryQueue):
                    module.update()
            del loss_all, recon_loss, g_loss, fake_validity
            if CONFIG.dist:
                del dist_loss
        del out

    for k, v in tot_loss.items():
        tot_loss[k] /= batches_done
    return tot_loss

def summary_grads(model, model_name, writer, epoch):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(float(p.grad.mean()))
    grads = np.array(grads)
    writer.add_scalar(f'Debug/{model_name} gradient mean', grads.mean(), epoch)
    writer.add_scalar(f'Debug/{model_name} gradient max', grads.max(), epoch)

if __name__ == '__main__':
    main()
