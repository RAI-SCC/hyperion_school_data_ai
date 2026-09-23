# Linear and Logistic Regression with PyTorch

This projects aim is to  let students play around with data analysis and machine learning concepts.

## Files

- `create_data.py`  
  Creates a small synthetic house-price dataset and saves it as CSV.
  It also creates a scatter plot as PNG.

- `train_linear_regression.py`  
  Reads the generated CSV and trains a linear regression model using
  PyTorch.

- `binary_dataloader.py`  
  Provides PyTorch `DataLoader`s for the real breast-cancer binary
  classification dataset from scikit-learn.

- `train_logistic_regression.py`  
  Trains a logistic regression model using the `DataLoader`s and saves
  the training-loss plot.

## Requirements

Python 3.10+ is recommended.

Install the required packages in a virtual environment:

```bash
python -m venv teaching
source teaching/bin/activate

pip install torch numpy pandas matplotlib scikit-learn
```

## Running the examples

First create the synthetic regression data:

```bash
python create_data.py
```

Then train linear regression:

```bash
python train_linear_regression.py
```

Finally, train logistic regression on the binary classification dataset:

```bash
python train_logistic_regression.py
```

The generated files are placed in:

```text
data/
    housing.csv

figures/
    housing_data.png
    linear_regression.png
    logistic_regression_loss.png
```

## Dataset

The logistic-regression example uses the breast-cancer dataset provided
by scikit-learn.

