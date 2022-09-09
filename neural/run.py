
import sys
sys.path.append('.')


from data import NoiseDataset
from lstm import LstmModel
from utils import load_universal_hyperparameters
import pytorch_lightning as pl
from torch.utils.data import DataLoader


if __name__ == "__main__":

    universal_hyperparameters = load_universal_hyperparameters()

    k = 1
    hidden_size = 10
    lr = 1e-04
    batch_size = 128
    num_workers = 4



    model = LstmModel(
        num_nombres=len(universal_hyperparameters['nombres']),
        hidden_size=hidden_size,
        lr=lr,
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

    trainer = pl.Trainer(
        devices=1,
        accelerator='mps',
    )

    trainer.fit(
        model,
        train_dataloader,
        val_dataloader
    )