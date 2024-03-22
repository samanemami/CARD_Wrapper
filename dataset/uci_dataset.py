import os
import torch
import numpy as np
import pandas as pd
import sklearn.datasets as dt


path = r"D:\Ph.D\Programming\Datasets\Regression"


def concrete():
    df = pd.read_csv(os.path.join(path, f"{concrete.__name__}.data"), header=None)
    X = (df.iloc[:, :-1]).values
    y = (df.iloc[:, -1]).values

    return X, y


def energy(target):
    cl = [
        "relative_compactness",
        "surface_area",
        "wall_area",
        "roof_area",
        "overall_height",
        "orientation",
        "glazing_area",
        "glazing_area_distribution",
        "heating_load",
        "cooling_load",
    ]
    df = pd.read_csv(os.path.join(path, f"{energy.__name__}.data"), names=cl)
    X = df.drop(["heating_load", "cooling_load"], axis=1).values
    y = (df[target]).values

    return X, y


def energy_heating():
    X, y = energy("heating_load")
    return X, y


def energy_cooling():
    X, y = energy("cooling_load")
    return X, y


def power():
    data = np.genfromtxt(
        os.path.join(path, f"{power.__name__}.data"),
        delimiter=",",
    )

    X = data[:, :-1]
    y = data[:, -1]

    return X, y


def boston():
    df = pd.read_csv(
        filepath_or_buffer="http://lib.stat.cmu.edu/datasets/boston",
        delim_whitespace=True,
        skiprows=21,
        header=None,
    )

    columns = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
        "MEDV",
    ]

    values_w_nulls = df.values.flatten()
    all_values = values_w_nulls[~np.isnan(values_w_nulls)]
    df = pd.DataFrame(
        data=all_values.reshape(-1, len(columns)),
        columns=columns,
    )
    X = (df.drop(columns=["MEDV"], axis=1)).values
    y = (df["MEDV"]).values

    return X, y


def wine():
    X, y = dt.load_wine(return_X_y=True)

    return X, y


def yacht():
    df = pd.read_csv(os.path.join(path, f"{yacht.__name__}.csv"))
    X = (df.drop(["Rr"], axis=1)).values
    y = (df["Rr"]).values

    return X, y


def protein():
    df = pd.read_csv(os.path.join(path, f"{protein.__name__}.csv"))
    X = (df.drop(columns=["RMSD"], axis=1)).values
    y = (df["RMSD"]).values

    return X, y
