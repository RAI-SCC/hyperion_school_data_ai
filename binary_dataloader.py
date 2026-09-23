"""
DataLoaders for a real binary classification dataset.

The breast-cancer dataset is provided by scikit-learn. It contains
569 samples, 30 numerical features, and two target classes.

The data are standardized using statistics from the training split only.
"""

import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def get_dataloaders(batch_size=32, test_size=0.2, random_state=42):
    dataset = load_breast_cancer()

    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.target,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    y_train = torch.tensor(
        y_train,
        dtype=torch.float32
    ).reshape(-1, 1)

    y_test = torch.tensor(
        y_test,
        dtype=torch.float32
    ).reshape(-1, 1)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader, x_train.shape[1]