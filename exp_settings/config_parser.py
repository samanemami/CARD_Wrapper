# ---------------------------------------------------------------------------------
# Intellectual property of the current class notice:
""" Base class for the Dynamic Sparse Diffusion model """
# Author: Seyedsaman Emami (https://samanemami.github.io/)
# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)
# ---------------------------------------------------------------------------------
import yaml
from argparse import Namespace


def int_or_bool(value):
    try:
        return int(value)
    except ValueError:
        if value.lower in ("true", "false"):
            return value.lower() == "true"
        else:
            raise ValueError(
                f"Invalid value: {value}. Value must be an integer or a boolean."
            )


def extract_config_values(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    params = {
        "config": Namespace(
            seed=int(config["general"]["seed"]),
            verbose=int_or_bool(config["general"]["verbose"]),
            classification=bool(config["general"]["classification"]),
            batch_size=int(config["general"]["batch_size"]),
            shuffle=bool(config["general"]["shuffle"]),
            n_steps=int(config["general"]["n_steps"]),
            n_z_samples=int(config["general"]["n_z_samples"]),
            epochs_mlp=int(config["mlp"]["epochs"]),
            eta_mlp=float(config["mlp"]["eta"]),
            dropout=float(config["mlp"]["dropout"]),
            patience_mlp=int_or_bool(config["mlp"]["patience"]),
            epochs_diff=int(config["diff"]["epochs"]),
            eta_diff=float(config["diff"]["eta"]),
            patience_diff=int_or_bool(config["diff"]["patience"]),
            prune_rate=float(config["diff"]["prune_rate"]),
            sparsity=str(config["diff"]["sparsity"]),
        )
    }

    return params
