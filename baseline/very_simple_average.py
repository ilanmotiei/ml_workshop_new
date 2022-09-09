
from typing import List, Tuple
import numpy as np

from metrics import compute_loss
from baseline_data import get_baseline_data
from utils import load_universal_hyperparameters, store_json_file


def validate(
        val_data: List[Tuple[np.ndarray, np.ndarray]],
) -> float:

    x_val = val_data['xs']  # shape = (n_samples, len(nombres))
    y_val = val_data['ys']  # shape = (n_samples, len(nombres))

    y_val_pred = np.nanmean(
        x_val,
        axis=0,
        keepdims=True,
        where=~np.isnan(x_val)
    )  # shape = (1, len(nombres))

    mean_mse = compute_loss(
        gt=y_val,
        prediction=y_val_pred
    )

    return mean_mse


if __name__ == "__main__":
    universal_hyperparameters = load_universal_hyperparameters()

    results = {

    }

    baseline_data = get_baseline_data(
        train_set_rate=0,
        k=1,
        flatten_xs=True
    )

    avg_mse = validate(
        val_data=baseline_data['val_data']
    )

    print(f"============ AVG MSE={avg_mse} ============")

    results = {
        'MSE': avg_mse,
        '# data points': baseline_data['val_data']['xs'].shape[0],
    }

    store_json_file(
        json_data=results,
        filepath=universal_hyperparameters['very_simple_average_results_filepath']
    )


