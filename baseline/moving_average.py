
from baseline_data import get_baseline_data
import numpy as np

from data import stations_normalized_noise_data
from metrics import compute_loss
from tqdm import tqdm
from utils import load_json_file, store_json_file, load_universal_hyperparameters


universal_hyperparameters = load_universal_hyperparameters()


def validate(
    alpha: float
) -> float:
    avg_mse = 0
    n = 0

    for station_name, station_normalized_noise_data in stations_normalized_noise_data.items():
        for i in range(1, len(station_normalized_noise_data)):
            prediction = np.nansum(
                station_normalized_noise_data[:i] * np.array([[alpha * ((1-alpha) ** ((i-1)-j))] for j in range(i)]),
                axis=0
            )

            for j, elem in np.ndenumerate(prediction):
                if np.isnan(elem):
                    prediction[j] = np.nan

            loss = compute_loss(
                gt=station_normalized_noise_data[i],
                prediction=prediction
            )

            avg_mse += loss
            n += 1

    return avg_mse / n


if __name__ == "__main__":

    results = {

    }

    for alpha in tqdm([1]):
        results[alpha] = validate(
            alpha=alpha
        )

        # store_json_file(
        #     json_data=results,
        #     filepath=universal_hyperparameters['ema_results_filepath']
        # )

        print(results[alpha])


