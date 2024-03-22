# ---------------------------------------------------------------------------------
# Revised from: https://github.com/XzwHan/CARD
# ---------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalRegressor(nn.Module):
    def __init__(
        self,
        n_steps: int,
        cat_x: bool,
        cat_y_pred: bool,
        x_dim: int,
        y_dim: int,
    ):
        super(ConditionalRegressor, self).__init__()
        self.cat_x = cat_x
        self.cat_y_pred = cat_y_pred
        data_dim = y_dim
        if self.cat_x:
            data_dim += x_dim
        if self.cat_y_pred:
            data_dim += y_dim
        self.lin1 = ConditionalLinear(data_dim, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, 1)

    def forward(self, x, y_t, y_0_hat, t):
        if self.cat_x:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat, x), dim=1)
            else:
                eps_pred = torch.cat((y_t, x), dim=1)
        else:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat), dim=1)
            else:
                eps_pred = y_t
        eps_pred = F.softplus(self.lin1(eps_pred, t))
        eps_pred = F.softplus(self.lin2(eps_pred, t))
        eps_pred = F.softplus(self.lin3(eps_pred, t))
        return self.lin4(eps_pred)


class DiffusionSequential(nn.Sequential):

    def forward(self, input, t):
        for module in self._modules.values():
            if isinstance(module, ConditionalLinear):
                input = module(input, t)
            else:
                input = module(input)
        return input


class ConditionalClassifier_ResNet18(nn.Module):

    def __init__(
        self,
        n_steps,
        x_dim,
        y_dim,
        n_hidden,
        cat_x,
        cat_y_pred,
    ):

        super().__init__()

        activation_fn = nn.Softplus()
        self.n_steps = n_steps
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.cat_x = cat_x
        self.cat_y_pred = cat_y_pred
        data_dim = y_dim
        if self.cat_x:
            data_dim += x_dim
        if self.cat_y_pred:
            data_dim += y_dim
        layer_sizes = [data_dim] + n_hidden
        layers = []
        for idx in range(1, len(layer_sizes)):
            layers += [
                ConditionalLinear(layer_sizes[idx - 1], layer_sizes[idx], n_steps),
                activation_fn,
            ]
        layers += [nn.Linear(layer_sizes[-1], y_dim)]
        self.model = DiffusionSequential(*layers)

    def forward(self, x, y_t, y_0_hat, t):

        if self.cat_x:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat, x), dim=1)
            else:
                eps_pred = torch.cat((y_t, x), dim=1)
        else:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat), dim=1)
            else:
                eps_pred = y_t
        return self.model(eps_pred, t)


class ConditionalClassifier_ResNet18(nn.Module):
    def __init__(
        self,
        n_steps,
        x_dim,
        y_dim,
        n_hidden,
        cat_x,
        cat_y_pred,
    ):
        super().__init__()

        activation_fn = nn.Softplus()
        self.n_steps = n_steps
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.cat_x = cat_x
        self.cat_y_pred = cat_y_pred
        data_dim = y_dim
        if self.cat_x:
            data_dim += x_dim
        if self.cat_y_pred:
            data_dim += y_dim

        self.model = resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, y_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x, y_t, y_0_hat, t):
        if self.cat_x:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat, x), dim=1)
            else:
                eps_pred = torch.cat((y_t, x), dim=1)
        else:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat), dim=1)
            else:
                eps_pred = y_t
        return self.model(eps_pred)
