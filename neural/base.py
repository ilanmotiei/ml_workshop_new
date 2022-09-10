
import torch
import pytorch_lightning as pl
from torch.optim.adam import Adam


class Base(pl.LightningModule):
    def __init__(
            self,
            lr: float
    ):
        super().__init__()
        self.lr = lr

    def compute_loss(
            self,
            gt: torch.Tensor,
            prediction: torch.Tensor,
            history_data: torch.Tensor
    ) -> torch.Tensor:

        gt_mask = torch.isnan(gt)
        gt.masked_fill_(mask=gt_mask, value=0)
        loss_per_elem = (prediction - gt) ** 2  # shape = (B, num_values_per_day)
        loss_per_elem.masked_fill_(mask=gt_mask, value=0)
        loss = torch.nanmean(loss_per_elem)

        if torch.all(torch.isnan(loss)):
            print("Nan loss: Aborting")
            print(torch.mean(torch.isnan(prediction).float()))
            print(torch.mean(gt_mask.float()))
            print(torch.mean(torch.isnan(history_data).float()))
            prediction = self.forward(
                history_data,
                # ^ : shape = (B, k, num_values_per_day)history_data,
                None
            )  # shape = (B, num_values_per_day)

            print(torch.mean(torch.isnan(prediction).float()))

            exit(1)

        return loss

    def step(
            self,
            batch: dict
    ) -> dict:

        prediction = self.forward(
            batch['history_data'],
            # ^ : shape = (B, k, num_values_per_day)history_data,
            batch['station_index'],  # shape = (B, )
            batch['station_photo']  # shape = (B, 3, 500, 500)
        )  # shape = (B, num_values_per_day)

        loss = self.compute_loss(
            gt=batch['gt'],  # shape = (B, num_values_per_day),
            prediction=prediction,
            history_data=batch['history_data']
        )

        return {
            'loss': loss
        }

    def training_step(
            self,
            batch: dict,
            batch_idx: int
    ) -> dict:
        res = self.step(batch)
        self.log(
            name='Training MSE',
            value=res['loss'],
            on_epoch=True
        )

        return res

    def validation_step(
            self,
            batch: dict,
            batch_idx: int
    ) -> None:
        res = self.step(batch)
        self.log(
            name='Validation MSE',
            value=res['loss'],
            on_epoch=True
        )



