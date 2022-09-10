import os
import sys

from pytorch_lightning.callbacks import EarlyStopping
from tqdm import tqdm

sys.path.append('.')


from data import NoiseDataset
from lstm import LstmModel
from utils import load_universal_hyperparameters
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils import load_json_file, store_json_file


if __name__ == "__main__":

    universal_hyperparameters = load_universal_hyperparameters()

    lr = 1e-03
    batch_size = 128
    num_workers = 16

    results_filepath = universal_hyperparameters['lstm_results_filepath']


    if os.path.isfile(results_filepath):
        results = load_json_file(
            filepath=results_filepath
        )
    else:
        results = [

        ]

    lstm_results = {
        'experiment': 'lstm_with_station_photo',
        'results': {

        }
    }

    results.append(lstm_results)

    for k in tqdm(list(range(1, 14)) + list(range(14, 200, 7))):
        hidden_size = 10 * k
        station_embedding_size = 2 * k

        lstm_results['results'][k] = {

        }

        for include_station_embedding in [False, True]:
            model = LstmModel(
                num_nombres=len(universal_hyperparameters['nombres']),
                hidden_size=hidden_size,
                lr=lr,
                include_station_embedding=include_station_embedding,
                station_embedding_size=station_embedding_size,
                include_station_image=True,
                finetune_photo_embedder=True
            )

            train_data = NoiseDataset(
                train_set_rate=universal_hyperparameters['train_set_rate'],
                is_train=True,
                k=k,
                tensorize=True,
                fulfill_missing_history_data=True
            )

            val_data = NoiseDataset(
                train_set_rate=universal_hyperparameters['train_set_rate'],
                is_train=False,
                k=k,
                tensorize=True,
                fulfill_missing_history_data=True
            )

            train_dataloader = DataLoader(
                dataset=train_data,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                collate_fn=NoiseDataset.collate_fn
            )

            val_dataloader = DataLoader(
                dataset=val_data,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                collate_fn=NoiseDataset.collate_fn
            )

            early_stopping_callback = EarlyStopping(
                monitor='Validation MSE',
                min_delta=0.00,
                patience=5,
                verbose=True,
                mode="min"
            )

            trainer = pl.Trainer(
                devices=1,
                accelerator='cuda',
                callbacks=[
                    early_stopping_callback
                ]
            )

            trainer.fit(
                model,
                train_dataloader,
                val_dataloader
            )

            lstm_results['results'][k][include_station_embedding] = float(early_stopping_callback.best_score)

            store_json_file(
                json_data=results,
                filepath=results_filepath
            )
