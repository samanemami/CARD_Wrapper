# ---------------------------------------------------------------------------------
# Revised from: https://github.com/XzwHan/CARD
# ---------------------------------------------------------------------------------

import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def _softmax(proba):
    softmax_output = torch.softmax(proba, dim=1)
    pred = torch.argmax(softmax_output, dim=1)
    return pred.unsqueeze(1)


def p_sample(
    x, y, y_0_hat, y_T_mean, t: int, alphas, one_minus_alphas_bar_sqrt, guidance_model
):

    try:
        z = torch.randn_like(y)
    except:
        z = torch.randn_like(y.float())
    t = torch.tensor([t]).to(device)
    alpha_t = extract(alphas, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t_m_1 = extract(one_minus_alphas_bar_sqrt, t - 1, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()
    # y_t_m_1 posterior mean component coefficients
    gamma_0 = (
        (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
    )
    gamma_1 = (
        (sqrt_one_minus_alpha_bar_t_m_1.square())
        * (alpha_t.sqrt())
        / (sqrt_one_minus_alpha_bar_t.square())
    )
    gamma_2 = 1 + (sqrt_alpha_bar_t - 1) * (alpha_t.sqrt() + sqrt_alpha_bar_t_m_1) / (
        sqrt_one_minus_alpha_bar_t.square()
    )
    eps_theta = guidance_model(x, y, y_0_hat, t).detach()
    # y_0 reparameterization
    y_0_reparam = (
        1
        / sqrt_alpha_bar_t
        * (
            y
            - (1 - sqrt_alpha_bar_t) * y_T_mean
            - eps_theta * sqrt_one_minus_alpha_bar_t
        )
    )
    y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y + gamma_2 * y_T_mean

    beta_t_hat = (
        (sqrt_one_minus_alpha_bar_t_m_1.square())
        / (sqrt_one_minus_alpha_bar_t.square())
        * (1 - alpha_t)
    )
    y_t_m_1 = y_t_m_1_hat.to(device) + beta_t_hat.sqrt().to(device) * z.to(device)
    return y_t_m_1


def p_sample_loop(
    x,
    y_0_hat,
    y_T_mean,
    n_steps,
    alphas,
    one_minus_alphas_bar_sqrt,
    only_last_sample,
    guidance_model,
):
    num_t, y_p_seq = None, None
    try:
        z = torch.randn_like(y_T_mean).to(device)
    except:
        z = torch.randn_like(y_T_mean.float()).to(device)
    cur_y = z + y_T_mean  # sampled y_T
    if only_last_sample:
        num_t = 1
    else:
        y_p_seq = [cur_y]
    for t in reversed(range(1, n_steps)):
        y_t = cur_y
        cur_y = p_sample(
            x,
            y_t,
            y_0_hat,
            y_T_mean,
            t,
            alphas,
            one_minus_alphas_bar_sqrt,
            guidance_model,
        )  # y_{t-1}
        if only_last_sample:
            num_t += 1
        else:
            y_p_seq.append(cur_y)
    if only_last_sample:
        assert num_t == n_steps
        y_0 = p_sample_t_1to0(
            x, cur_y, y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt, guidance_model
        )
        return y_0
    else:
        assert len(y_p_seq) == n_steps
        y_0 = p_sample_t_1to0(
            x, y_p_seq[-1], y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt, guidance_model
        )
        y_p_seq.append(y_0)
        return y_p_seq


def p_sample_t_1to0(x, y, y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt, guidance_model):
    # corresponding to timestep 1 (i.e., t=1 in diffusion models)
    t = torch.tensor([0]).to(device)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    eps_theta = guidance_model(x, y, y_0_hat, t).detach()
    # y_0 reparameterization
    y_0_reparam = (
        1
        / sqrt_alpha_bar_t
        * (
            y
            - (1 - sqrt_alpha_bar_t) * y_T_mean
            - eps_theta * sqrt_one_minus_alpha_bar_t
        )
    )
    y_t_m_1 = y_0_reparam.to(device)
    return y_t_m_1


# Reverse function
def predict_step(
    x,
    y_0_hat,
    alphas,
    one_minus_alphas_bar_sqrt,
    n_z_samples,
    n_steps,
    guidance_model,
):
    with torch.no_grad():

        y_0_hat_tile = torch.tile(y_0_hat, (n_z_samples, 1)).to(device)
        test_x_tile = torch.tile(x, (n_z_samples, 1)).to(device)

        try:
            z = torch.randn_like(y_0_hat_tile).to(device)
        except:
            z = torch.randn_like(y_0_hat_tile.float()).to(device)

        y_t = y_0_hat_tile + z

        # generate samples from all time steps for the mini-batch
        y_tile_seq = p_sample_loop(
            test_x_tile,
            y_0_hat_tile,
            y_0_hat_tile,
            n_steps,
            alphas.to(device),
            one_minus_alphas_bar_sqrt.to(device),
            False,
            guidance_model,
        )

        # put in shape [n_z_samples, batch_size, output_dimension]
        y_tile_seq = [
            arr.reshape(n_z_samples, x.shape[0], y_t.shape[-1]) for arr in y_tile_seq
        ]

        final_recoverd = y_tile_seq[-1]

        mean_pred = final_recoverd.mean(dim=0).detach().cpu().squeeze()
        std_pred = final_recoverd.std(dim=0).detach().cpu().squeeze()

        return {
            "pred": mean_pred,
            "pred_uct": std_pred,
            "aleatoric_uct": std_pred,
            "samples": y_tile_seq,
        }


# Forward functions
def q_sample(y, y_0_hat, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise=None):
    """
    y_0_hat: prediction of pre-trained guidance model; can be extended to represent
        any prior mean setting at timestep T.
    """
    if noise is None:
        noise = torch.randn_like(y).to(y.device)
    sqrt_alpha_bar_t = extract(alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    # q(y_t | y_0, x)
    y_t = (
        sqrt_alpha_bar_t * y
        + (1 - sqrt_alpha_bar_t) * y_0_hat
        + sqrt_one_minus_alpha_bar_t * noise
    )
    return y_t


def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start**0.5, end**0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [
                min(
                    1
                    - (
                        math.cos(
                            ((i + 1) / num_timesteps + cosine_s)
                            / (1 + cosine_s)
                            * math.pi
                            / 2
                        )
                        ** 2
                    )
                    / (
                        math.cos(
                            (i / num_timesteps + cosine_s)
                            / (1 + cosine_s)
                            * math.pi
                            / 2
                        )
                        ** 2
                    ),
                    max_beta,
                )
                for i in range(num_timesteps)
            ]
        )
        if schedule == "cosine_reverse":
            betas = betas.flip(0)  # starts at max_beta then decreases fast
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [
                start
                + 0.5
                * (end - start)
                * (1 - math.cos(t / (num_timesteps - 1) * math.pi))
                for t in range(num_timesteps)
            ]
        )
    return betas
