import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class KarmaDataset(Dataset):
    def __init__(self, root, train=True, img_size=(256, 256), normalize=False, normalize_tanh=False, enable_transform=True, full=True):
        """
        root: Veri seti ana dizini (örneğin 'karma/train' veya 'karma/val')
        train: Eğitim veya doğrulama/test için mi kullanılacak?
        img_size: Görüntü boyutları (örneğin (256, 256))
        normalize: Normalize işlemi yapılacak mı?
        normalize_tanh: -1 ile 1 aralığına normalize edilecek mi?
        enable_transform: Veri artırma işlemi uygulanacak mı?
        full: Tüm veri mi yoksa sınırlı bir alt küme mi kullanılacak?
        """
        self.data = []
        self.train = train
        self.root = root
        self.normalize = normalize
        self.img_size = img_size
        self.mean = 0.1307
        self.std = 0.3081
        self.full = full

        if train:
            if enable_transform:
                self.transforms = [
                    transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                    transforms.ToTensor()
                ]
            else:
                self.transforms = [transforms.ToTensor()]
        else:
            self.transforms = [transforms.ToTensor()]
        if normalize_tanh:
            self.transforms.append(transforms.Normalize((0.5,), (0.5,)))
        self.transforms = transforms.Compose(self.transforms)

        self.load_data()

    def load_data(self):
        """
        Klasörleri dolaşarak görüntüleri ve etiketleri yükler.
        Train: Sadece 'NORMAL'
        Val/Test: Hem 'NORMAL' hem de 'ANOMALY'
        """
        self.fnames = []
        if self.train:
            # Sadece NORMAL verileri yükle
            items = os.listdir(os.path.join(self.root, 'NORMAL'))
            for item in items:
                image = Image.open(os.path.join(self.root, 'NORMAL', item)).resize(self.img_size)
                self.data.append((image, 0))  # NORMAL -> Label 0
                self.fnames.append(item)
        else:
            # NORMAL verileri yükle
            items = os.listdir(os.path.join(self.root, 'NORMAL'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                image = Image.open(os.path.join(self.root, 'NORMAL', item)).resize(self.img_size)
                self.data.append((image, 0))  # NORMAL -> Label 0
                self.fnames.append(item)

            # ANOMALY verilerini yükle
            items = os.listdir(os.path.join(self.root, 'ANOMALY'))
            for idx, item in enumerate(items):
                if not self.full and idx > 9:
                    break
                image = Image.open(os.path.join(self.root, 'ANOMALY', item)).resize(self.img_size)
                self.data.append((image, 1))  # ANOMALY -> Label 1
                self.fnames.append(item)

            print(f"{len(self.data)} normal ve anomalili veri yüklendi: {self.root}")

    def __getitem__(self, index):
        """
        Bir görüntü ve onun etiketini döner.
        """
        img, label = self.data[index]
        img = self.transforms(img)[[0]]
        if self.normalize:
            img -= self.mean
            img /= self.std
        return img, (torch.zeros((1,)) + label).long()

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    # Veri setini test et
    dataset = KarmaDataset(r'C:\Users\BURKINEFASO\SIMSID_PLUS\data\karma4\val', train=False)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for i, (img, label) in enumerate(trainloader):
        if img.shape[1] == 3:
            import matplotlib.pyplot as plt
            plt.imshow(img[0, 1], cmap='gray')
            plt.show()
        break
