import json
from pathlib import Path

import numpy as np
import pandas as pd


class Logger:
    def __init__(self, run_number, exp_name):
        self.run_number = run_number
        self._records = []

    def log(self, step, tx_matrices, tx_power_indices, reward):
        link_indices, ap_indices, station_indices = np.where(tx_matrices == 1)

        power_indices = tx_power_indices[link_indices, ap_indices]

        self._records.append(
            {
                "run": self.run_number,
                "step": step,
                "link_indices": link_indices.tolist(),
                "ap_indices": ap_indices.tolist(),
                "station_indices": station_indices.tolist(),
                "tx_power_indices": power_indices.tolist(),
                "reward": reward.item() if isinstance(reward, np.generic) else reward,
            }
        )

    def dataframe(self):
        columns = [
            "run",
            "step",
            "link_indices",
            "ap_indices",
            "station_indices",
            "tx_power_indices",
            "reward"
        ]
        return pd.DataFrame(self._records, columns=columns)

    def save(self, directory="logs/"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        path = directory / f"run_{self.run_number}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=4)

    def clear(self):
        self._records.clear()
