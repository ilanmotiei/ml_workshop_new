
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

from metrics import compute_loss
from baseline_data import get_baseline_data
from utils import load_universal_hyperparameters, store_json_file


def validate(
        val_data: List[Tuple[np.ndarray, np.ndarray]],
) -> float:

    x_val = val_data['xs']  # shape = (n_samples, k, len(nombres))
    y_val = val_data['ys']  # shape = (n_samples, len(nombres))

    y_val_pred = np.nanmean(
        x_val,
        axis=1,
        where=~np.isnan(x_val)
    )  # shape = (n_samples, len(nombres))

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
            train_set_rate=0,
            k=k,
            flatten_xs=False
        )

        avg_mse = validate(
            val_data=baseline_data['val_data']
        )

        print(f"============ K={k}===AVG MSE={avg_mse} ============")

        results[k] = {
            'MSE': avg_mse,
            '# data points': baseline_data['val_data']['xs'].shape[0],
        }

        store_json_file(
            json_data=results,
            filepath=universal_hyperparameters['simple_average_results_filepath']
        )


