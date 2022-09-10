import sys

from torch.optim import Adam

from base import Base
import torch
from transformers import AutoModel


class LstmModel(Base):

    def __init__(
            self,
            num_nombres: int,
            hidden_size: int,
            lr: float,
            include_station_embedding: bool,
            station_embedding_size: int,
            include_station_image: bool,
            finetune_photo_embedder: bool
    ):
        super().__init__(
            lr=lr
        )

        self.lstm = torch.nn.LSTM(
            input_size=num_nombres,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.include_station_embedding = include_station_embedding
        self.include_station_image = include_station_image
        self.finetune_photo_embedder = finetune_photo_embedder

        feat_size = hidden_size

        if self.include_station_embedding:
            self.station_embedding = torch.nn.Embedding(
                num_embeddings=70,
                embedding_dim=station_embedding_size
            )
            feat_size += station_embedding_size

        if self.include_station_image:
            self.photo_embedder = AutoModel.from_pretrained('fxmarty/resnet-tiny-beans')
            feat_size += 64

        self.proj = torch.nn.Linear(
            in_features=feat_size,
            out_features=num_nombres
        )

    def forward(
            self,
            history_data: torch.Tensor,  # shape = (B, k, num_bombres)
            # IMPORTANT: WE'RE GETTING IT AFTER IT'S MISSING PARTS WERE FULFILLED
            station_index: torch.Tensor,  # shape = (B, )
            station_photos: torch.Tensor  # shape = (B, 3, 500, 500)
    ) -> torch.Tensor:

        output, (hn, cn) = self.lstm(history_data)
        # ^ : hn.shape = (1, B, hidden_size)

        features = hn.squeeze(dim=0)  # shape = (B, hidden_size)

        if self.include_station_embedding:
            features = torch.concat(
                [
                    features,
                    self.station_embedding(station_index.long())  # shape = (B, station_embedding_dim)
                ],
                dim=1
            )

        if self.include_station_image:
            station_photos_embeddings = \
                self.photo_embedder(station_photos).pooler_output.squeeze(dim=3).squeeze(dim=2)
            # ^ : shape = (B, embedding_dim=64)

            features = torch.concat(
                [
                    features,
                    station_photos_embeddings
                ],
                dim=1
            )

        prediction = self.proj(features)  # shape = (B, num_nombres)

        return prediction

    def configure_optimizers(self):

        if self.include_station_image and (not self.finetune_photo_embedder):
            for param in self.photo_embedder.parameters():
                param.requires_grad = False

        return Adam(
            params=self.parameters(),
            lr=self.lr
        )
