
from sklearn import linear_model
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

from metrics import compute_loss
from baseline_data import get_baseline_data
from utils import load_universal_hyperparameters, store_json_file


def train(
        train_data: List[Tuple[np.ndarray, np.ndarray]],
) -> linear_model.LinearRegression:

    regressor = linear_model.LinearRegression()

    x_train = train_data['xs']  # shape = (n_samples, k * len(nombres))
    y_train = train_data['ys']  # shape = (n_samples, len(nombres))

    regressor.fit(x_train, y_train)

    return regressor


def validate(
        regressor: linear_model.LinearRegression,
        val_data: List[Tuple[np.ndarray, np.ndarray]],
) -> float:

    x_val = val_data['xs']  # shape = (n_samples, k * len(nombres))
    y_val = val_data['ys']  # shape = (n_samples, len(nombres))

    y_val_pred = regressor.predict(x_val)  # shape = (n_samples, len(nombres))

    mean_mse = compute_loss(
        gt=y_val,
        prediction=y_val_pred
    )

    return mean_mse


if __name__ == "__main__":
    universal_hyperparameters = load_universal_hyperparameters()

    results = {

    }

    for k in tqdm(list(range(1, 14)) + list(range(14, 200, 7))):
        baseline_data = get_baseline_data(
            train_set_rate=universal_hyperparameters['train_set_rate'],
            k=k
        )

        avg_mse = validate(
            regressor=train(
                train_data=baseline_data['train_data']
            ),
            val_data=baseline_data['val_data']
        )

        print(f"============ K={k}===AVG MSE={avg_mse} ============")

        results[k] = {
            'MSE': avg_mse,
            '# train data points': baseline_data['train_data']['xs'].shape[0],
            '# validation data points': baseline_data['val_data']['xs'].shape[0],
        }

        results[k].update({
            'rate': results[k]['# train data points']
                    / (results[k]['# validation data points'] + results[k]['# train data points'])
        })

        store_json_file(
            json_data=results,
            filepath=universal_hyperparameters['linear_regression_results_filepath']
        )


