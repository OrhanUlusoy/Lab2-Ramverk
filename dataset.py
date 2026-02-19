"""Dataset-wrapper för CIFAR-10.

Syfte:
- Hålla data-loading isolerad från träningslogik.
- Ge ett enhetligt PyTorch `Dataset`-gränssnitt som kan matas till `DataLoader`.
"""

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFAR10Wrapper(Dataset):
    def __init__(self, root="data/cifar10", train=True, download=False):
        # Minimal preprocessing: konvertera till tensor (skalar till [0,1] och CHW-format).
        # Hålls medvetet enkel för labben.
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.ds = CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=self.transform,
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Returnerar (bild_tensor, label_int)
        return self.ds[idx]
