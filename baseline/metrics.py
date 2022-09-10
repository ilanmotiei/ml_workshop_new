import math

import numpy as np


def compute_loss(
        gt: np.ndarray,  # shape = (B, num_values_per_day) or (num_values_per_day, )
        prediction: np.ndarray  # shape = same
) -> float:

    loss_per_elem = (prediction - gt) ** 2  # shape = same

    loss = 0
    n = 0
    for _, elem in np.ndenumerate(loss_per_elem):
        if np.isnan(elem):
            continue

        loss += elem
        n += 1

    if n == 0:
        return 0

    return loss / n