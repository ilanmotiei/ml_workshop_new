
import numpy as np

import sys
sys.path.append('.')

from data import NoiseDataset


def get_baseline_data(
        train_set_rate: float,
        k: int,
        flatten_xs: bool = True
) -> dict:

    train_data = NoiseDataset(
        train_set_rate=train_set_rate,
        is_train=True,
        k=k,
        tensorize=False,
        fulfill_missing_history_data=True
    )

    val_data = NoiseDataset(
        train_set_rate=train_set_rate,
        is_train=False,
        k=k,
        tensorize=False,
        fulfill_missing_history_data=True
    )

    def organize(
            data: NoiseDataset,
            is_train: bool
    ) -> dict:

        xs = []
        ys = []

        for item in data:
            x = item['history_data']

            if flatten_xs:
                x = x.flatten()

            y = item['next_day']
            if is_train and np.any(np.isnan(y)):
                continue
            else:
                xs.append(x)
                ys.append(y)

        return {
            'xs': np.stack(xs) if len(xs) > 0 else None,
            'ys': np.stack(ys) if len(ys) > 0 else None
        }

    return {
        'train_data': organize(
            train_data,
            is_train=True
        ),
        'val_data': organize(
            val_data,
            is_train=False
        )
    }
