# CARD - wrapper

This repository presents a streamlined wrapper
implementation for the CARD
(Classification and Regression Diffusion Models) model [1].

## Disclaimer and Acknowledgement

All written code in this repository is a revision of the reference [1].
All rights belong to the original authors [1] of the original code,
and this repository serves as a wrapper to facilitate easier
replication of the study outlined in the referenced paper [1].

# Usage

To utilize the CARD wrapper, you can follow the example below:

To utilize the CARD wrapper, follow these steps:

- Data Preparation: Prepare your data for training and testing.
  You can either use your own dataset or generate synthetic
  data using [dataset](dataset).

- Initialization: Import the necessary modules and initialize the CARD model.

- Setting Parameters: Set the desired parameters for the CARD model.

- Model Fitting: Fit the model to your training data.


```Python

from model.train import train
from argparse import Namespace
from Libs.config import get_config
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, make_classification


def data(clf):
    if not clf:
        X, y = make_regression()
    else:
        X, y = make_classification()

    x_train, x_test, y_train, y_test = train_test_split(X, y)
    return x_train, x_test, y_train, y_test, clf


x_train, x_test, y_train, y_test, clf = data(False)
card = train(config=get_config())

params = {
    "config": Namespace(
        seed=111,
        verbose=False,
        batch_size=128,
        shuffle=True,
        n_steps=100,
        n_z_samples=100,
        epochs_mlp=200,
        eta_mlp=1e-1,
        dropout=1e-1,
        patience_mlp=False,
        epochs_diff=200,
        eta_diff=1e-1,
        patience_diff=20,
        classification=clf,
    )
}

card.set_params(**params)
print(card.get_params())
card.fit(x_train, y_train)

```

# Contribution

Contributions are welcome! If you find any issues or 
have suggestions for improvements, 
please feel free to open an issue or submit a pull request.

# Reference

- [1] Han, Xizewen, Huangjie Zheng, and Mingyuan Zhou. "Card:
  Classification and regression diffusion models."
  Advances in Neural Information Processing Systems 35 (2022): 18100-18115.
