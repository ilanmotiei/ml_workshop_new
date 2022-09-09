
import matplotlib.pyplot as plt
from utils import load_json_file, load_universal_hyperparameters


if __name__ == "__main__":

    universal_hyperparameters = load_universal_hyperparameters()

    linear_regression_results = load_json_file(universal_hyperparameters['linear_regression_results_filepath'])
    simple_average_results = load_json_file(universal_hyperparameters['simple_average_results_filepath'])
    very_simple_average_results = load_json_file(universal_hyperparameters['very_simple_average_results_filepath'])
    lstm_results = load_json_file(universal_hyperparameters['lstm_results_filepath'])

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
        [very_simple_average_results['MSE']['all'] for _ in range(min_k, max_k)],
        label='All Average'
    )

    plt.plot(
        range(min_k, max_k),
        [very_simple_average_results['MSE']['station'] for _ in range(min_k, max_k)],
        label='Station Average'
    )

    plt.plot(
        range(min_k, max_k),
        [very_simple_average_results['MSE']['day_of_week'] for _ in range(min_k, max_k)],
        label='Day of Week Average'
    )

    label = 'Day of Week Average'

