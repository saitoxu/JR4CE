import logging
import os
import random

import numpy as np
import torch


def getLogger(name, path, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    os.makedirs(path, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s: %(message)s")
    log_file = f"{path}train.log"
    with open(log_file, "w"):
        pass
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.best_value = 0.0
        self.count = 0
        self.patience = patience

    def __call__(self, value: float):
        should_save, should_stop = False, False
        if value > self.best_value:
            self.best_value = value
            self.count = 0
            should_save = True
        else:
            self.count += 1
        if self.count >= self.patience:
            should_stop = True
        return should_save, should_stop


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class NegativeSampler:
    def __init__(self, path: str, job_size: int) -> None:
        self.observed = {}
        self.all_jobs = set(range(job_size))
        datasets = ["train.txt", "val.txt", "test.txt"]
        for dataset in datasets:
            with open(f"{path}/{dataset}") as f:
                for line in f:
                    user_id, *job_ids = map(int, line.split(" "))
                    user_id = int(user_id)
                    for job_id in job_ids:
                        if user_id not in self.observed:
                            self.observed[user_id] = set()
                        self.observed[user_id].add(job_id)

    def sample(self, user_id: int, size: int) -> list[int]:
        observed = self.observed[user_id]
        candidates = list(self.all_jobs - observed)
        sample_size = min(size, len(candidates))
        result = random.sample(candidates, sample_size)
        return result
