
import matplotlib.pyplot as plt
from utils import load_json_file, load_universal_hyperparameters

universal_hyperparameters = load_universal_hyperparameters()

linear_regression_results = load_json_file(universal_hyperparameters['linear_regression_results_filepath'])
simple_average_results = load_json_file(universal_hyperparameters['simple_average_results_filepath'])
very_simple_average_results = load_json_file(universal_hyperparameters['very_simple_average_results_filepath'])
lstm_results = load_json_file(universal_hyperparameters['lstm_results_filepath'])[0]['results']
# ema_results = load_json_file(universal_hyperparameters['ema_results_filepath'])


def all():
    plt.plot(
        [int(k) for k in linear_regression_results.keys()],
        [v['MSE'] for v in linear_regression_results.values()],
        label='Linear Regression'
    )

    plt.plot(
        [int(k) for k in simple_average_results.keys()],
        [v['MSE'] for v in simple_average_results.values()],
        label='Simple Average'
    )

    min_k = min([int(k) for k in simple_average_results.keys()])
    max_k = max([int(k) for k in simple_average_results.keys()])

    plt.plot(
        range(min_k, max_k),
        [very_simple_average_results['all']['MSE'] for _ in range(min_k, max_k)],
        label='All Average'
    )

    plt.plot(
        range(min_k, max_k),
        [very_simple_average_results['station']['MSE'] for _ in range(min_k, max_k)],
        label='Station Average'
    )

    plt.plot(
        range(min_k, max_k),
        [very_simple_average_results['day_of_week']['MSE'] for _ in range(min_k, max_k)],
        label='Day of Week Average'
    )

    plt.plot(
        range(min_k, max_k),
        [very_simple_average_results['station_and_day_of_week']['MSE'] for _ in range(min_k, max_k)],
        label='Station + Day of Week Average',
        color="blue"
    )

    plt.plot(
        [int(k) for k in lstm_results.keys()],
        [v['false'] for v in lstm_results.values()],
        label='LSTM',
        color='green'
    )

    plt.plot(
        [int(k) for k in lstm_results.keys()],
        [v['true'] for v in lstm_results.values()],
        label='LSTM + station',
        color='red'
    )

    plt.legend()

    plt.show()


def linear_regression():

    plt.plot(
        [int(k) for k in linear_regression_results.keys()],
        [v['MSE'] for v in linear_regression_results.values()],
        label='Linear Regression'
    )

    plt.xlabel("K (Number of previous days used for prediction)")
    plt.ylabel("MSE")

    plt.title("Linear Regression")

    plt.show()

    plt.plot(
        [int(k) for k in linear_regression_results.keys()],
        [round((1 - v['rate']) * 100, 2) for v in linear_regression_results.values()],
        label='Linear Regression'
    )

    plt.xlabel("K (Number of previous days used for prediction)")
    plt.ylabel("%")

    plt.title("% Validation Data Points (with respect to all data)")

    plt.show()


def lstm_and_linear_regression():

    plt.plot(
        [int(k) for k in lstm_results.keys()],
        [v['false'] for v in lstm_results.values()],
        label='LSTM',
        color='blue'
    )

    plt.plot(
        [int(k) for k in lstm_results.keys()],
        [v['true'] for v in lstm_results.values()],
        label='LSTM + station',
        color='red'
    )

    plt.plot(
        [int(k) for k in linear_regression_results.keys()],
        [v['MSE'] for v in linear_regression_results.values()],
        label='Linear Regression'
    )

    plt.xlabel("K (Number of previous days used for prediction)")
    plt.ylabel("MSE")
    plt.legend()

    plt.title("LSTM")

    plt.show()


def simple_average():

    plt.plot(
        [int(k) for k in simple_average_results.keys()],
        [v['MSE'] for v in simple_average_results.values()],
        label='Simple Average'
    )

    plt.xlabel("K (Number of previous days used for prediction)")
    plt.ylabel("MSE")
    plt.title("Simple Average")

    plt.show()


def ema():

    plt.plot(
        [float(k) for k in ema_results.keys()],
        [v for v in ema_results.values()]
    )

    plt.xlabel("Alpha")
    plt.ylabel("MSE")
    plt.title("Exponentially Moving Average")

    plt.show()


if __name__ == "__main__":
    lstm_and_linear_regression()