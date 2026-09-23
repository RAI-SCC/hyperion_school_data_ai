"""
Train logistic regression with PyTorch on the breast-cancer dataset.

The DataLoader is defined in binary_dataloader.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from binary_dataloader import get_dataloaders


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"


def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            probabilities = model(x_batch)
            predictions = (probabilities >= 0.5).float()

            correct += (predictions == y_batch).sum().item()
            total += y_batch.numel()

    return correct / total


def main():
    train_loader, test_loader, n_features = get_dataloaders(
        batch_size=32
    )

    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, 1),
        torch.nn.Sigmoid(),
    )

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01
    )

    losses = []

    for epoch in range(100):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in train_loader:
            probabilities = model(x_batch)
            loss = loss_fn(probabilities, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)

        epoch_loss /= len(train_loader.dataset)
        losses.append(epoch_loss)

        if (epoch + 1) % 10 == 0:
            accuracy = evaluate(model, test_loader)

            print(
                f"Epoch {epoch + 1:3d} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Test accuracy: {accuracy:.3f}"
            )

    FIGURE_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title("Logistic regression training")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "logistic_regression_loss.png",
        dpi=150,
    )
    plt.close()


if __name__ == "__main__":
    main()