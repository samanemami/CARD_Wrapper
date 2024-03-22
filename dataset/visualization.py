import matplotlib.pyplot as plt


def plot(x_train, y_train, x_test, y_test, prediction_set):
    fig, axs = plt.subplots(1, figsize=(10, 6))

    plt.scatter(
        x_test,
        y_test,
        color="gray",
        label="ground truth",
        alpha=0.5,
    )
    plt.scatter(
        x_train,
        y_train,
        color="blue",
        label="train_data",
        alpha=0.5,
    )
    plt.scatter(
        x_test,
        prediction_set["pred"].detach(),
        color="orange",
        label="predictions",
        alpha=0.5,
    )

    plt.fill_between(
        x_test,
        prediction_set["pred"].detach() - prediction_set["pred_uct"].detach(),
        prediction_set["pred"].detach() + prediction_set["pred_uct"].detach(),
        alpha=0.2,
        color="yellowgreen",
        label=r"Uncertainty Range",
    )

    plt.legend()
    plt.show()
