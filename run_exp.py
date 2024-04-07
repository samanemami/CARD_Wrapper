# %%
import warnings
from model.train import train
import matplotlib.pylab as plt
from Libs.config import get_config
from dataset_.toy_dataset import *
from dataset_.uci_dataset import *
from sklearn.model_selection import train_test_split
from exp_settings.config_parser import extract_config_values
from sklearn.metrics import r2_score, accuracy_score, root_mean_squared_error


def run(dataset):

    warnings.simplefilter("ignore")

    X, y = dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1
    )

    params = extract_config_values(rf"exp_settings\{dataset.__name__}.yml")

    model = train(config=get_config())

    model.set_params(**params)
    model.get_params()

    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    try:
        print("R^2", r2_score(y_test, pred, multioutput="raw_values"))
        print(
            "RMSE",
            root_mean_squared_error(y_test, pred, multioutput="raw_values"),
        )
        print("RMSE", root_mean_squared_error(y_test, pred))
    except:
        print(accuracy_score(y_test, pred))

if __name__ == "__main__":
    run(concrete)
