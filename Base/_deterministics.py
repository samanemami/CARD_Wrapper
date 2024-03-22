import torch.nn as nn
from abc import abstractmethod


class BaseNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, dropout_rate):

        super(BaseNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

    @abstractmethod
    def forward(self, x):
        pass


class MLPRegressor(BaseNetwork):
    def __init__(
        self,
        input_dim,
        output_dim,
        dropout_rate,
        use_batchnorm,
        negative_slope,
    ):
        super(MLPRegressor, self).__init__(input_dim, output_dim, dropout_rate)
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        self.hidden_layers = [100, 50]
        self.layer_dims = [input_dim] + list(self.hidden_layers)
        self.network = self._build_network()

    def _build_network(self):
        layers = []
        for in_dim, out_dim in zip(self.layer_dims, self.layer_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
            layers.append(nn.Dropout(p=self.dropout_rate))
        layers.pop()  # Remove dropout layer after the last hidden layer
        layers.append(nn.Linear(self.layer_dims[-1], self.output_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPClassifier(BaseNetwork):

    def __init__(
        self,
        input_dim,
        output_dim,
        dropout_rate=0,
    ):
        super(MLPClassifier, self).__init__(input_dim, output_dim, dropout_rate)

        n_hidden = [50, 50]
        layers = []
        layer_sizes = [input_dim] + n_hidden
        for idx in range(1, len(layer_sizes)):
            layers += [
                nn.Linear(layer_sizes[idx - 1], layer_sizes[idx]),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
        layers += [nn.Linear(layer_sizes[-1], output_dim)]
        self.model = nn.Sequential(*layers)
        self.n_outputs = output_dim

    def forward(self, x):
        return self.model(x)
