import argparse


def add_arg_group(name):
    arg = parser.add_argument_group(name)
    arg_list.append(arg)
    return arg


def int_or_bool(value):
    try:
        return int(value)
    except ValueError:
        if value.lower in ("true", "false"):
            return value.lower() == "true"
        else:
            argparse.ArgumentTypeError(
                f"Invalid value: {value}. Value must be an integer or a boolean."
            )


if "parser" not in globals():
    arg_list = []
    parser = argparse.ArgumentParser()
    # General
    general_arg = add_arg_group("General")
    general_arg.add_argument("--seed", type=int, default=111)
    general_arg.add_argument("--batch_size", type=int, default=128)
    general_arg.add_argument("--shuffle", type=bool, default=True)
    general_arg.add_argument("--n_steps", type=int, default=100)
    general_arg.add_argument("--n_z_samples", type=int, default=100)
    general_arg.add_argument("--verbose", type=bool, default=False)
    general_arg.add_argument("--classification", type=bool, default=False)

    # MLP arguments
    mlp_arg = add_arg_group("FFNN")
    ffnn_defaults = {
        "epochs_mlp": 100,
        "dropout": 0,
        "eta_mlp": 0.1,
        "patience_mlp": False,
    }
    mlp_arg.add_argument("--epochs_mlp", type=int, default=ffnn_defaults["epochs_mlp"])
    mlp_arg.add_argument("--dropout", type=float, default=ffnn_defaults["dropout"])
    mlp_arg.add_argument("--eta_mlp", type=float, default=ffnn_defaults["eta_mlp"])
    mlp_arg.add_argument(
        "--patience_mlp", type=int_or_bool, default=ffnn_defaults["patience_mlp"]
    )

    # Diff arguments
    diff_arg = add_arg_group("Diff")
    diff_defaults = {
        "epochs_diff": 100,
        "eta_diff": 0.1,
        "patience_diff": 5,
    }
    diff_arg.add_argument(
        "--epochs_diff", type=int, default=diff_defaults["epochs_diff"]
    )
    diff_arg.add_argument("--eta_diff", type=float, default=diff_defaults["eta_diff"])
    diff_arg.add_argument(
        "--patience_diff", type=int_or_bool, default=diff_defaults["patience_diff"]
    )


def get_config():
    args, _ = parser.parse_known_args()
    return args
