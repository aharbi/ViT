from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class MNISTDataset(Dataset):
    def __init__(self, patch_size: int = 14, train: bool = True, path: str = "data/"):
        self.patch_size = patch_size

        self.train = train
        self.path = path

        self.data = MNIST(
            self.path, train=self.train, download=True, transform=ToTensor()
        )

        self.C, self.H, self.W = self.data[0][0].shape

        # Compute number of patches
        self.Nx = self.W // self.patch_size
        self.Ny = self.H // self.patch_size

        self.N = self.Nx * self.Ny

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        # x shape [C, H, W]
        x, y = self.data[idx]

        # x_p shape [C, Nx, Ny, patch_size, patch_size]
        x_p = x.unfold(1, self.patch_size, self.patch_size).unfold(
            2, self.patch_size, self.patch_size
        )

        # x_p shape [C, N, patch_size, patch_size]
        x_p = x_p.contiguous().view(self.C, self.N, self.patch_size, self.patch_size)

        # x_p shape [N, C, patch_size, patch_size]
        x_p = x_p.permute(1, 0, 2, 3)

        # x_p shape [N, C * patch_size * patch_size]
        x_p = x_p.contiguous().view(-1, self.C * self.patch_size**2)

        return x_p, y
