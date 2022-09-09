from typing import List

import numpy as np
import torch
from tqdm import tqdm

import pandas as pd
from torch.utils.data import Dataset
from utils import load_universal_hyperparameters


import warnings
warnings.simplefilter(action='ignore')


universal_hyperparameters = load_universal_hyperparameters()


def preprocess(
        noise_data: pd.DataFrame,
        nombres: List[str] = None
) -> dict:
    noise_data = noise_data[['station_name', 'published_dt'] + nombres]
    # noise_data = noise_data.dropna(
    #     axis=0,
    #     how='any'
    # )

    normalized_data = {

    }

    for station_name, station_noise_data in tqdm(noise_data.groupby('station_name')):
        station_noise_data['published_dt'] = pd.to_datetime(station_noise_data['published_dt'])
        station_noise_data = station_noise_data.sort_values(by='published_dt')

        normalized_data[station_name] = station_noise_data[nombres].to_numpy()
        # ^ : shape = (n, len(nomrbes)), where n is the number of days

    return normalized_data


stations_normalized_noise_data = preprocess(
    noise_data=pd.read_csv(universal_hyperparameters['data_filepath']),
    nombres=universal_hyperparameters['nombres'],
)  # stations_normalized_noise_data[station_name] = np.ndarray(of shape (n, len(nombres))


class NoiseDataset(Dataset):

    def __init__(
            self,
            train_set_rate: float,  # between 0 and 1
            is_train: bool,
            k: int,
            tensorize: bool,
            fulfill_missing_history_data: bool
    ):
        super().__init__()

        self.stations_normalized_noise_data = {

        }

        for station_name, station_normalized_noise_data in stations_normalized_noise_data.items():
            n = len(station_normalized_noise_data)
            split_idx = int(n * train_set_rate)

            if is_train:
                self.stations_normalized_noise_data[station_name] = station_normalized_noise_data[:split_idx]
            else:
                self.stations_normalized_noise_data[station_name] = station_normalized_noise_data[split_idx:]

        self.data = [

        ]

        for station_index, (station_name, station_normalized_noise_data) \
                in enumerate(self.stations_normalized_noise_data.items()):
            for i in range(k, len(station_normalized_noise_data)):
                data_point = {
                    'history_data': station_normalized_noise_data[i - k: i],
                    'next_day': station_normalized_noise_data[i],
                    'station_name': station_name,
                    'station_index': station_index,
                    # 'day_of_week':
                }

                if fulfill_missing_history_data:
                    data_point['history_data'] = NoiseDataset.fulfill_missing_input_data(data_point['history_data'])

                    if np.any(np.isnan(data_point['history_data'])):
                        continue

                if tensorize:
                    data_point['history_data'] = torch.Tensor(data_point['history_data'])
                    data_point['next_day'] = torch.Tensor(data_point['next_day'])
                    data_point['station_index'] = torch.Tensor([data_point['station_index']])

                self.data.append(
                    data_point
                )

    def __len__(
            self
    ) -> int:
        return len(self.data)

    def __getitem__(
            self,
            idx: int
    ) -> dict:
        return self.data[idx]

    @staticmethod
    def collate_fn(
            batch: List[dict]
    ) -> dict:
        return {
            'history_data': torch.stack([i['history_data'] for i in batch]),  # size = (B, k, len(self.nombres))
            'gt': torch.stack([i['next_day'] for i in batch]),  # size = (B, len(self.nombres))
            'station_index': torch.Tensor([i['station_index'] for i in batch])  # size = (B, )
        }

    @staticmethod
    def fulfill_missing_input_data(
            history_data: np.ndarray  # shape = (k, len(nombres))
    ) -> np.ndarray:

        nans_mask = np.isnan(history_data)
        history_data = np.nan_to_num(history_data)
        history_data = (~ nans_mask) * history_data + \
                       nans_mask * np.nanmean(history_data, axis=0, keepdims=True, where=~nans_mask)  # shape = (1, num_values_per_day)
        # ^ : includes broadcasting

        return history_data
