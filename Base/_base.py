import os
import time
import torch
import logging
import numpy as np
from ._utils import *
import torch.nn as nn
from ._params import Params
from abc import abstractmethod
from torch.utils.data import DataLoader
from Libs._logging import FileHandler, StreamHandler
from sklearn.model_selection import train_test_split
from ._deterministics import MLPRegressor, MLPClassifier
from ._diffusion import ConditionalRegressor, ConditionalClassifier_ResNet18, ConditionalClassifier


class Base(Params):

    @abstractmethod
    def __init__(self, config):

        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def _softmax(self, proba):
        "Returns the predicted values for classification."

    def _check_array(self, arr, clf=False):
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        if not self.config.classification:
            if arr.dim() == 1:
                arr = arr.unsqueeze(1)
        if self.config.classification and clf:
            arr = arr.squeeze().long()
        else:
            arr = arr.to(torch.float32)
        return arr.to(self.device)

    def _dataloader(self, X, y):

        def _data_loader(X, y):
            dataset = torch.utils.data.TensorDataset(X, y)
            data_loader = DataLoader(
                dataset, batch_size=self.config.batch_size, shuffle=self.config.shuffle
            )
            return data_loader

        x_train, x_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=1e-1,
            random_state=self.config.seed,
            shuffle=self.config.shuffle,
        )

        train_loader, val_loader = _data_loader(x_train, y_train), _data_loader(
            x_val, y_val
        )
        return train_loader, val_loader

    def _perturbation_gen(
        self, beta_schedule="linear", beta_start=0.0001, beta_end=0.02
    ):

        betas = make_beta_schedule(
            beta_schedule, self.config.n_steps, beta_start, beta_end
        ).to(self.device)

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)

        return alphas_bar_sqrt, one_minus_alphas_bar_sqrt, alphas

    def _mlp(self, X, y):

        if not self.config.classification:

            cond_pred_model = MLPRegressor(
                input_dim=X.shape[1],
                output_dim=1 if y.ndim == 1 else y.shape[1],
                dropout_rate=self.config.dropout,
                use_batchnorm=True,
                negative_slope=1e-1,
            )

            loss = nn.MSELoss()

        else:

            cond_pred_model = MLPClassifier(
                input_dim=X.shape[1],
                output_dim=len(torch.unique(y)),
                dropout_rate=self.config.dropout,
            )

            loss = nn.CrossEntropyLoss()

        cond_pred_model.to(self.device)
        return cond_pred_model, loss

    def _diff(self, X, y):

        ResNet = False

        if self.config.classification:

            if not ResNet:

                diff_model = ConditionalClassifier(
                    n_steps=self.config.n_steps,
                    x_dim=X.shape[1],
                    y_dim=len(torch.unique(y)),
                    n_hidden=[64, 64],
                    cat_x=True,
                    cat_y_pred=True,
                )
            else:

                diff_model = ConditionalClassifier_ResNet18(
                    n_steps=self.config.n_steps,
                    x_dim=X.shape[1],
                    y_dim=len(torch.unique(y)),
                    n_hidden=[64, 64],
                    cat_x=True,
                    cat_y_pred=True,
                )

        else:
            diff_model = ConditionalRegressor(
                n_steps=self.config.n_steps,
                cat_x=True,
                cat_y_pred=True,
                x_dim=X.shape[1],
                y_dim=y.shape[1],
            )

        diff_model.to(self.device)
        return diff_model

    def _clear_session(self):
        torch.cuda.empty_cache()
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        globals().update(
            {name: obj for name, obj in globals().items() if torch.is_tensor(obj)}
        )
        torch.cuda.empty_cache()

    def _set_seed(self):
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        np.random.seed(self.config.seed)

        np.set_printoptions(precision=4, suppress=True)
        torch.set_float32_matmul_precision("medium")

    def _check_params(self):

        self._clear_session()
        self._set_seed()
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        os.chdir(path)
        current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        current_path = os.path.join(path, f"logs_{current_time}")
        os.mkdir(f"logs_{current_time}")
        os.chdir(current_path)

        self.log_fh = FileHandler(current_path)
        self.log_sh = StreamHandler()

        if self.config.verbose == True:
            StreamHandler.logger.setLevel(logging.INFO)
        else:
            StreamHandler.logger.setLevel(logging.WARNING)

        self.log_fh.info(f"logs created on {current_time}")
