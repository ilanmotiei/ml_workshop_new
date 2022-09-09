
from base import Base
import torch


class LstmModel(Base):

    def __init__(
            self,
            num_nombres: int,
            hidden_size: int,
            lr: float
    ):
        super().__init__(
            lr=lr
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

        output, (hn, cn) = self.lstm(history_data)
        # ^ : hn.shape = (1, B, num_nombres)

        return hn.squeeze(dim=0)
