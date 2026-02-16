import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class KarmaDataset(Dataset):
    def __init__(self, root, train=True, img_size=(256, 256), 
                 normalize=False, normalize_tanh=False, 
                 enable_transform=True, full=True, positive_ratio=1.0):

        self.data = []
        self.fnames = []
        self.train = train
        self.root = root
        self.img_size = img_size
        self.normalize = normalize
        self.mean = 0.1307
        self.std = 0.3081
        self.full = full
        self.positive_ratio = positive_ratio

        # --- Dönüşüm fonksiyonu (augmentasyon sadece eğitimde)
        transform_list = []
        if train and enable_transform:
            transform_list.extend([
                transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ])
        transform_list.append(transforms.ToTensor())
        if normalize_tanh:
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        self.transform = transforms.Compose(transform_list)

        # --- Veriyi RAM'e yükle
        self._load_and_preprocess_all()

    def _load_and_preprocess_all(self):
        self.data = []
        self.fnames = []

        def _load_folder(folder, label):
            folder_path = os.path.join(self.root, folder)
            if not os.path.exists(folder_path):
                return
            for fname in sorted(os.listdir(folder_path)):
                fpath = os.path.join(folder_path, fname)
                try:
                    img = Image.open(fpath).convert('L').resize(self.img_size)
                    tensor_img = self.transform(img)
                    self.data.append((tensor_img, label))
                    self.fnames.append(fname)
                except Exception as e:
                    print(f"Problem with {fpath}: {e}")

        if not self.train:
            _load_folder('NORMAL', 0)
            _load_folder('ANOMALY', 1)
            print(f'[TEST] {len(self.data)} data loaded from: {self.root}')
            return

        pos_items = sorted(os.listdir(os.path.join(self.root, 'NORMAL')))
        neg_path = os.path.join(self.root, 'ANOMALY')
        neg_items = sorted(os.listdir(neg_path)) if os.path.exists(neg_path) else []

        num_pos = int(len(pos_items) * self.positive_ratio)
        if not neg_items:
            num_neg = 0
            print("[TRAIN] ANOMALY klasörü boş, yalnızca NORMAL verilerle eğitim yapılacak.")
        else:
            num_neg = len(neg_items) if self.full else len(pos_items) - num_pos

        # NORMAL
        for fname in pos_items[:num_pos]:
            fpath = os.path.join(self.root, 'NORMAL', fname)
            try:
                img = Image.open(fpath).convert('L').resize(self.img_size)
                tensor_img = self.transform(img)
                self.data.append((tensor_img, 0))
                self.fnames.append(fname)
            except Exception as e:
                print(f"Problem with {fpath}: {e}")

        # ANOMALY
        for fname in neg_items[:num_neg]:
            fpath = os.path.join(self.root, 'ANOMALY', fname)
            try:
                img = Image.open(fpath).convert('L').resize(self.img_size)
                tensor_img = self.transform(img)
                self.data.append((tensor_img, 1))
                self.fnames.append(fname)
            except Exception as e:
                print(f"Problem with {fpath}: {e}")

        print(f'[TRAIN] {len(self.data)} data loaded from: {self.root}, positive rate: {self.positive_ratio:.2f}')

    def __getitem__(self, index):
        img, label = self.data[index]
        if self.normalize:
            img = (img - self.mean) / self.std
        return img, torch.tensor([label]).long()

    def __len__(self):
        return len(self.data)
