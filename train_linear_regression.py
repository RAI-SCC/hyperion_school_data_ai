"""
Train a linear regression model with PyTorch on the generated housing data.

Run create_data.py first.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "housing.csv"
FIGURE_DIR = ROOT / "figures"


def main():
    data = pd.read_csv(DATA_FILE)

    x = torch.tensor(
        data["size_m2"].values,
        dtype=torch.float32,
    ).reshape(-1, 1)

    y = torch.tensor(
        data["price"].values,
        dtype=torch.float32,
    ).reshape(-1, 1)

    model = torch.nn.Linear(1, 1)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

    losses = []

    for epoch in range(1000):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Weight: {model.weight.item():.4f}")
    print(f"Bias: {model.bias.item():.4f}")

    FIGURE_DIR.mkdir(exist_ok=True)

    with torch.no_grad():
        predictions = model(x)

    plt.figure(figsize=(6, 4))
    plt.scatter(x.numpy(), y.numpy(), label="Data")
    plt.plot(
        x.numpy(),
        predictions.numpy(),
        label="Linear regression",
    )
    plt.xlabel("Size (m²)")
    plt.ylabel("Price")
    plt.title("PyTorch linear regression")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "linear_regression.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()