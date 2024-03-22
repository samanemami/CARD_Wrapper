import torch
import numpy as np
from Base import Base
from tqdm import trange
from Base._utils import *
from torch.optim import Adam


class train(Base):
    def __init__(self, config):
        super().__init__(config)

    def _softmax(self, proba):
        softmax_output = torch.softmax(proba, dim=1)
        pred = torch.argmax(softmax_output, dim=1)
        return pred.unsqueeze(1)

    def fit(self, X, y):

        self._check_params()

        clf = True if self.config.classification else False
        X = self._check_array(X, False)
        y = self._check_array(y, clf)

        train_loader, validation_loader = self._dataloader(X, y)
        X_batch, y_batch = next(iter(train_loader))
        self.mlp, aux_cost_fn = self._mlp(X=X_batch, y=y_batch)
        self.diff = self._diff(X=X_batch, y=y_batch)

        self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, self.alpha = (
            self._perturbation_gen()
        )

        best_valid_loss = np.inf
        counter = 0
        self.mlp.train()
        aux_optimizer = Adam(self.mlp.parameters(), lr=self.config.eta_mlp)

        self.log_sh.info(f"----Pre-training Feed Forward model----")
        self.log_fh.info(f"----Pre-training Feed Forward model----")
        for epoch in trange(self.config.epochs_mlp, leave=True):
            # Training
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                target = target.squeeze()
                aux_optimizer.zero_grad()
                y_pred = self.mlp(data)
                aux_cost = aux_cost_fn(y_pred, target)
                aux_cost.backward()
                aux_optimizer.step()
                self.log_sh.info(f"\t Training loss - Epoch {epoch}: {aux_cost.item()}")
                self.log_fh.info(f"\t Training loss - Epoch {epoch}: {aux_cost.item()}")

            self.mlp.eval()
            with torch.no_grad():
                num_batches = 0
                valid_loss = 0
                for val_x, val_y in validation_loader:
                    val_x, val_y = val_x.to(self.device), val_y.to(self.device)
                    val_y = val_y.squeeze()
                    y_pred = self.mlp(val_x)
                    aux_cost = aux_cost_fn(y_pred, val_y)
                    valid_loss += aux_cost.item()
                    self.log_sh.info(
                        f"\t Validation loss - Epoch {epoch}: {aux_cost.item()}"
                    )
                    self.log_fh.info(
                        f"\t Validation loss - Epoch {epoch}: {aux_cost.item()}"
                    )
                    num_batches += 1
            valid_loss /= len(validation_loader)

            # Applying Early stooping
            if self.config.patience_mlp:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    counter = 0
                else:
                    counter += 1

                if counter > self.config.patience_mlp:
                    self.log_sh.info(f"Early stopping at epoch {epoch}.")
                    self.log_fh.info(f"Early stopping at epoch {epoch}.")
                    break

        diff_optimizer = Adam(self.diff.parameters(), lr=self.config.eta_diff)
        diff_bar = trange(self.config.epochs_diff, leave=True)
        self.diff.train()
        best_loss = float("inf")
        early_stop_counter = 0

        self.log_sh.info(f"----diffusion process----")
        self.log_fh.info(f"----diffusion process----")
        self.diff_loss_train = []
        self.diff_loss_val = []
        for epoch in diff_bar:
            training_loss = 0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.shape[0]

                ant_samples_t = torch.randint(
                    low=0, high=self.config.n_steps, size=(batch_size // 2 + 1,)
                ).to(self.device)
                ant_samples_t = torch.cat(
                    [ant_samples_t, self.config.n_steps - 1 - ant_samples_t], dim=0
                )[:batch_size]

                y_0_hat = self.mlp(data)

                m_target = target.unsqueeze(1) if self.config.classification else target

                e = torch.randn_like(
                    m_target.float() if self.config.classification else m_target
                )

                y_t_sample = q_sample(
                    m_target,
                    y_0_hat,
                    self.alphas_bar_sqrt,
                    self.one_minus_alphas_bar_sqrt,
                    ant_samples_t,
                    noise=e,
                )

                model_output = self.diff(data, y_t_sample, y_0_hat, ant_samples_t)

                loss = (e - model_output).square().mean()
                training_loss += loss.item()

                diff_optimizer.zero_grad()
                loss.backward()
                diff_optimizer.step()

                self.log_sh.info(f"\t Training loss - Epoch {epoch}: {loss.item()}")
                self.log_fh.info(f"\t Training loss - Epoch {epoch}: {loss.item()}")

                aux_cost = aux_cost_fn(self.mlp(data), target)
                aux_optimizer.zero_grad()
                aux_cost.backward()
                aux_optimizer.step()

                diff_bar.set_description(f"Loss: {loss.item()}", refresh=True)
            training_loss /= len(train_loader)
            self.diff_loss_train.append(training_loss)

            # Validation
            self.mlp.eval()
            self.diff.eval()
            with torch.no_grad():
                validation_loss = 0
                num_batches = 0
                for val_x, val_y in validation_loader:
                    val_x, val_y = val_x.to(self.device), val_y.to(self.device)
                    y_0_hat_val = self.mlp(val_x)

                    batch_size = val_x.shape[0]
                    ant_samples_t = torch.randint(
                        low=0, high=self.config.n_steps, size=(batch_size // 2 + 1,)
                    ).to(self.device)
                    ant_samples_t = torch.cat(
                        [ant_samples_t, self.config.n_steps - 1 - ant_samples_t], dim=0
                    )[:batch_size]

                    m_val_y = (
                        val_y.unsqueeze(1) if self.config.classification else val_y
                    )

                    e = torch.randn_like(
                        m_val_y.float() if self.config.classification else m_val_y
                    )
                    y_t_sample_val = q_sample(
                        m_val_y,
                        y_0_hat_val,
                        self.alphas_bar_sqrt,
                        self.one_minus_alphas_bar_sqrt,
                        ant_samples_t,
                        noise=e,
                    )
                    output_val = self.diff(
                        val_x, y_t_sample_val, y_0_hat_val, ant_samples_t
                    )

                    loss_val = (e - output_val).square().mean()

                    validation_loss += loss_val.item()

                    self.log_sh.info(
                        f"\t Validation loss - Epoch {epoch}: {loss_val.item()}"
                    )
                    self.log_fh.info(
                        f"\t Validation loss - Epoch {epoch}: {loss_val.item()}"
                    )
                    num_batches += 1
                validation_loss /= num_batches
                self.diff_loss_val.append(validation_loss)

            # Applying Early stooping
            if self.config.patience_diff:
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter > self.config.patience_mlp:
                    self.log_sh.info(f"Early stopping at epoch {epoch}.")
                    self.log_fh.info(f"Early stopping at epoch {epoch}.")
                    break

    def predict(self, X):
        X = self._check_array(X)
        y_0_hat = self.mlp(X.to(device))
        pred_dict = predict_step(
            X,
            y_0_hat,
            self.alpha,
            self.one_minus_alphas_bar_sqrt,
            self.config.n_z_samples,
            self.config.n_steps,
            self.diff,
        )
        self.__pred = pred_dict["pred"]
        if self.config.classification:
            self.__pred = self._softmax(self.__pred)
        self.__pred = self.__pred.detach().numpy().squeeze()
        self.__pred_uct = pred_dict["pred_uct"].detach().numpy().squeeze()
        self._save_records()
        return self.__pred

    def _save_records(self):
        np.savetxt("training_loss.csv", self.diff_loss_train, delimiter=",")
        np.savetxt("validation_loss.csv", self.diff_loss_val, delimiter=",")
        np.savetxt("prediction.csv", self.__pred, delimiter=",")
        np.savetxt("uncertainty.csv", self.__pred_uct, delimiter=",")
