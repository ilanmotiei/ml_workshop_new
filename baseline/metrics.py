import math

import numpy as np


def compute_loss(
        gt: np.ndarray,  # shape = (B, num_values_per_day) or (num_values_per_day, )
        prediction: np.ndarray  # shape = same
) -> float:

    loss_per_elem = (prediction - gt) ** 2  # shape = same

    return np.nanmean(
        loss_per_elem,
        where=~np.isnan(loss_per_elem)
    )
