"""
Create a small synthetic house-price dataset.

Outputs:
    data/housing.csv
    figures/housing_data.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"


def main():
    rng = np.random.default_rng(42)

    data = pd.DataFrame(
        {
            "size_m2": [45, 55, 60, 70, 80, 90, 100, 120, 140],
            "rooms": [1, 2, 2, 3, 3, 4, 4, 5, 5],
        }
    )

    # Synthetic house prices with a little random noise.
    data["price"] = (
        80
        + 3.2 * data["size_m2"]
        + 18 * data["rooms"]
        + rng.normal(0, 25, len(data))
    )

    DATA_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)

    data.to_csv(DATA_DIR / "housing.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.scatter(data["size_m2"], data["price"])
    plt.xlabel("Size (m²)")
    plt.ylabel("Price")
    plt.title("Synthetic house-price data")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "housing_data.png", dpi=150)
    plt.close()

    print(f"Saved dataset to {DATA_DIR / 'housing.csv'}")
    print(f"Saved plot to {FIGURE_DIR / 'housing_data.png'}")


if __name__ == "__main__":
    main()