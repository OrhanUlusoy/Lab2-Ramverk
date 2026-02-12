from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset


class CIFAR10Wrapper(Dataset):
    def __init__(self, root="data/cifar10", train=True):
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # converts to [0,1] and CHW
        ])
        self.ds = CIFAR10(root=root, train=train, download=True, transform=self.transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]  # returns (image_tensor, label_int)
