import torch

from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from vit import ViT
from dataset import MNISTDataset


class MNISTViT:
    def __init__(
        self,
        patch_size: int = 7,
        num_classes: int = 10,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_size: int = 3072,
        dropout_rate: float = 0.1,
    ):

        self.input_size = patch_size**2
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_size = mlp_size
        self.dropout_rate = dropout_rate

        self.vit = ViT(
            input_size=self.input_size,
            num_classes=self.num_classes,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_size=self.mlp_size,
            dropout_rate=self.dropout_rate,
        )

    def initialize_dataset(self, batch_size):

        train_dataset = MNISTDataset(patch_size=self.patch_size, train=True)
        test_dataset = MNISTDataset(patch_size=self.patch_size, train=False)

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    def train(self, num_epochs: int, batch_size: int, lr: float, device: str):

        train_dataloader, test_dataloader = self.initialize_dataset(batch_size)

        optim = torch.optim.Adam(self.vit.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()

        self.vit.to(device)
        self.vit.train()

        for epoch in range(num_epochs):

            optim.param_groups[0]["lr"] = lr * (1 - epoch / num_epochs)

            pbar = tqdm(train_dataloader)

            loss_ema = None
            for x, y in pbar:
                optim.zero_grad()

                x = x.to(device)
                y = y.to(device)

                y_hat = self.vit(x)

                output = loss(y_hat, y)

                output.backward()

                loss_ema = output.item()

                if loss_ema is None:
                    loss_ema = output.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * output.item()

                pbar.set_description("Loss: {:.4f} - Epoch: {}".format(loss_ema, epoch))
                optim.step()

            print(torch.argmax(y_hat, 1))
            print(y)


if __name__ == "__main__":
    mnist_vit = MNISTViT(
        patch_size=14,
        num_classes=10,
        hidden_size=256,
        num_heads=4,
        num_layers=6,
        mlp_size=256,
        dropout_rate=0.1,
    )

    mnist_vit.train(5, 8, 3e-5, "cpu")
