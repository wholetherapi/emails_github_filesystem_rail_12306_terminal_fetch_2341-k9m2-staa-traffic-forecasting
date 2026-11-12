"""Training entry point for the sparse temporal attention model."""

import argparse
import yaml
import torch
from torch import nn

from src.dataset import load_series, windowed
from src.model import SparseTemporalAttention


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    series = load_series(f"data/{config['dataset']}.csv")
    x, y = windowed(series, horizon=config["horizon"])

    model = SparseTemporalAttention(config["hidden_dim"], config["heads"])
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.HuberLoss()

    for epoch in range(config["epochs"]):
        batch = torch.tensor(x[:256], dtype=torch.float32)
        target = torch.tensor(y[:256], dtype=torch.float32)
        prediction = model(batch, adjacency=torch.ones(batch.shape[1]))
        loss = criterion(prediction, target)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        print(f"epoch={epoch} loss={loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
