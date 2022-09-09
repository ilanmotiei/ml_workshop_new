
from base import Base
import torch


class LstmModel(Base):

    def __init__(
            self,
            num_nombres: int,
            hidden_size: int,
            lr: float,
            include_station_embedding: bool
    ):
        super().__init__(
            lr=lr
        )

        self.include_station_embedding = include_station_embedding

        if self.include_station_embedding:
            self.station_embedding = torch.nn.Embedding(
                num_embeddings=70,
                embedding_dim=num_nombres
            )

        self.lstm = torch.nn.LSTM(
            input_size=num_nombres,
            hidden_size=hidden_size,
            proj_size=num_nombres,
            batch_first=True,
        )

    def forward(
            self,
            history_data: torch.Tensor,  # shape = (B, k, num_bombres)
            # IMPORTANT: WE'RE GETTING IT AFTER IT'S MISSING PARTS WERE FULFILLED
            station_index: torch.Tensor  # shape = (B, )
    ) -> torch.Tensor:

        if self.include_station_embedding:
            history_data = torch.concat(
                [
                    history_data,
                    self.station_embedding(station_index.long()).unsqueeze(dim=1)  # shape = (B, 1, hidden_size)
                ],
                dim=1
            )

        output, (hn, cn) = self.lstm(history_data)
        # ^ : hn.shape = (1, B, num_nombres)

        return hn.squeeze(dim=0)
