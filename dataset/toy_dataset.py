"""

Intellectual Property Notice

- This file (DataModule.py) contains code written by [Seyedsaman Emami].
- Copyright [Seyedsaman Emami] [2023]
- Licensed under the [LGPL-2.1 license].

"""

import math
import torch
import numpy as np
from typing import Callable
from typing import Callable, Tuple
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def custom_polynomial(x, noise=True):
    """Define a custom polynomial function."""
    result = 7 * np.sin(x)
    if noise:
        result += 3 * np.abs(np.cos(x / 2)) * np.random.randn()
    return result


class HeteroscedasticData(LightningDataModule):
    """Implementation of a Toy Dataset with Heteroscedastic Noise."""

    def __init__(
        self,
        n_train: int = 400,
        n_true: int = 50,
        batch_size: int = 128,
        generate_y: Callable = custom_polynomial,
    ):
        super().__init__()

        np.random.seed(1)
        X_train = np.zeros(n_train)
        Y_train = np.zeros(n_train)

        for i in range(n_train):
            rnd = np.random.rand()
            if rnd < 1 / 3.0:
                X_train[i] = np.random.normal(loc=-4, scale=2.0 / 5.0)
            else:
                if rnd < 2.0 / 3.0:
                    X_train[i] = np.random.normal(loc=0.0, scale=0.9)
                else:
                    X_train[i] = np.random.normal(loc=4.0, scale=2.0 / 5.0)

            Y_train[i] = generate_y(X_train[i])

        mean_X, std_X = np.mean(X_train), np.std(X_train)
        mean_Y, std_Y = np.mean(Y_train), np.std(Y_train)

        X_train_n = (X_train - mean_X) / std_X
        Y_train_n = (Y_train - mean_Y) / std_Y

        self.X_train = torch.from_numpy(X_train_n).unsqueeze(-1).type(torch.float32)
        self.y_train = torch.from_numpy(Y_train_n).unsqueeze(-1).type(torch.float32)

        X_true = np.linspace(X_train.min(), X_train.max(), n_true)
        Y_true = X_true * 0.0

        for i in range(len(X_true)):
            Y_true[i] = generate_y(X_true[i], noise=False)

        X_true_n = (X_true - mean_X) / std_X
        Y_true_n = (Y_true - mean_Y) / std_Y

        self.X_true = torch.from_numpy(X_true_n).unsqueeze(-1).type(torch.float32)
        self.y_true = torch.from_numpy(Y_true_n).unsqueeze(-1).type(torch.float32)

        self.batch_size = batch_size

    def prepare_data(self):
        """Prepare and split the data."""
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.1, random_state=42
        )
        self.train_data = TensorDataset(X_train, y_train)
        self.val_data = TensorDataset(X_val, y_val)

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            TensorDataset(self.X_true, self.y_true), batch_size=self.batch_size
        )

    def get_train_loader(self) -> DataLoader:
        """Return train loader."""
        return self.train_dataloader()

    def get_test_loader(self) -> DataLoader:
        """Return test loader."""
        return self.test_dataloader()

    def get_train_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return training data."""
        return self.X_train, self.y_train

    def get_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return testing data."""
        return self.X_true, self.y_true


def circle_data(num_samples=1000, test_split=0.2, noise_factor=0.1):
    angles = torch.rand(num_samples) * 2 * math.pi

    x = torch.cos(angles)[:, np.newaxis]
    y = torch.sin(angles)[:, np.newaxis]

    # Heteroscedastic noise
    noise_x = torch.abs(x) * torch.randn_like(x) * noise_factor
    noise_y = torch.abs(y) * torch.randn_like(y) * noise_factor

    x += noise_x
    y += noise_y

    labels = (x**2 + y**2 <= 1).float()

    split_idx = int(num_samples * (1 - test_split))
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    labels_train, labels_test = labels[:split_idx], labels[split_idx:]

    return x_train, x_test, y_train, y_test, labels_train, labels_test
