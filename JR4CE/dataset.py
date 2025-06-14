import random

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        neg_sample_size: int,
        threshold_user: float,
        threshold_item: float,
    ):
        (
            self.user_size,
            self.item_size,
            self.entity_size,
            item_relations,
            user_preference_relations,
            user_current_relations,
        ) = _load_info(data_path)
        self._data, self.ui_adj_mtx = self._load_data(
            self.user_size, self.item_size, data_path
        )
        self._all_item_ids = set(list(range(self.item_size)))
        self._neg_sample_size = neg_sample_size
        self._user_observed_items = _load_user_observed_items(data_path)
        self.entity_item_map = _load_entity_item_map(
            data_path, self.entity_size, self.user_size, self.item_size
        )
        (
            self.item_edge_index,
            self.item_edge_type,
            self.user_current_edge_index,
            self.user_current_edge_type,
            self.user_preference_edge_index,
            self.user_preference_edge_type,
            self.num_relations,
        ) = _load_kg(
            data_path, item_relations, user_preference_relations, user_current_relations
        )
        self._div_data = _diversity_data_augmentation(
            data_path,
            self.user_size,
            self.item_size,
            threshold_user,
            threshold_item,
            item_relations,
            user_preference_relations,
            user_current_relations,
        )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        user_id, item_id = self._data[idx]
        neg_item_ids = self._negative_sampling(user_id, self._neg_sample_size)
        div_item_id = random.choice(self._div_data.get(user_id, [-1]))

        return (
            user_id,
            item_id,
            torch.tensor(neg_item_ids),
            div_item_id,
        )

    def data(self, device: str) -> dict:
        return dict(
            entity_item_map=self.entity_item_map.to(device),
            item_edge_index=self._add_reverse_edges(self.item_edge_index.to(device)),
            user_current_edge_index=self._add_reverse_edges(
                self.user_current_edge_index.to(device)
            ),
            user_preference_edge_index=self._add_reverse_edges(
                self.user_preference_edge_index.to(device)
            ),
            ui_adj_mtx=torch.Tensor(self.ui_adj_mtx.todense()).to(device),
        )

    def _add_reverse_edges(self, edge_index) -> torch.Tensor:
        reversed_edge_index = torch.stack([edge_index.T[1], edge_index.T[0]], dim=0).T
        _edge_index = torch.cat([edge_index, reversed_edge_index], dim=0)
        return _edge_index

    def _negative_sampling(self, user_id: int, size: int) -> list[int]:
        weights = np.ones(self.item_size)
        weights[list(self._user_observed_items[user_id])] = 0
        weights = weights / np.sum(weights)
        return list(
            np.random.choice(self.item_size, size=size, replace=False, p=weights)
        )

    def _load_data(
        self, user_size: int, item_size: int, data_path: str
    ) -> tuple[list[list[int]], sp.lil_matrix]:
        data = []
        ui_adj_mtx = sp.lil_matrix((user_size, item_size))

        with open(f"{data_path}/train.txt") as f:
            for line in f:
                user_id, *item_ids = list(map(lambda x: int(x), line.split(" ")))
                for item_id in item_ids:
                    data.append([user_id, item_id])
                    ui_adj_mtx[user_id, item_id] = 1

        a = sp.csr_matrix((self.user_size, self.user_size))
        b = sp.csr_matrix((self.item_size, self.item_size))
        mat = sp.vstack(
            [sp.hstack([a, ui_adj_mtx]), sp.hstack([ui_adj_mtx.transpose(), b])]
        )
        mat = (mat != 0) * 1.0
        mat: sp.csr_matrix = (
            mat + sp.eye(mat.shape[0])
        ) * 1.0  # <class 'scipy.sparse._csr.csr_matrix'>
        return data, mat.tolil()


def eval_dataset(data_path: str, data_file: str) -> list[tuple[int, int, list[int]]]:
    data = []
    _, item_size, _, _, _, _ = _load_info(data_path)
    all_item_ids = set(list(range(item_size)))
    user_observed_items = _load_user_observed_items(data_path)

    with open(f"{data_path}/{data_file}") as f:
        for line in f:
            user_id, item_id = list(map(lambda x: int(x), line.split(" ")))
            observed_items = user_observed_items[user_id]
            neg_item_ids = list(all_item_ids - observed_items - {item_id})
            neg_item_ids = sorted(neg_item_ids)
            data.append((user_id, item_id, neg_item_ids))
    return data


def _load_kg(
    data_path: str,
    item_relations: list[int],
    user_preference_relations: list[int],
    user_current_relations: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    item_edge_index = []
    item_edge_type = []
    user_current_edge_index = []
    user_current_edge_type = []
    user_preference_edge_index = []
    user_preference_edge_type = []
    num_relations = 0
    with open(f"{data_path}/kg.txt") as f:
        for line in f:
            h, r, t = map(int, line.split())
            if r in item_relations:
                item_edge_index.append([h, t])
                item_edge_type.append(r)
                num_relations = max(num_relations, r + 1)
            elif r in user_current_relations:
                user_current_edge_index.append([h, t])
                user_current_edge_type.append(r)
                num_relations = max(num_relations, r + 1)
            elif r in user_preference_relations:
                user_preference_edge_index.append([h, t])
                user_preference_edge_type.append(r)
                num_relations = max(num_relations, r + 1)
            else:
                raise ValueError(f"Unknown relation {r} in kg.txt")
    return (
        torch.tensor(item_edge_index),
        torch.tensor(item_edge_type),
        torch.tensor(user_current_edge_index),
        torch.tensor(user_current_edge_type),
        torch.tensor(user_preference_edge_index),
        torch.tensor(user_preference_edge_type),
        num_relations,
    )


def _load_user_observed_items(data_path: str) -> dict[int, set[int]]:
    user_observed_items = {}
    files = ["train.txt"]
    for file in files:
        with open(f"{data_path}/{file}") as f:
            for line in f:
                user_id, *item_ids = map(int, line.split())
                if user_id not in user_observed_items:
                    user_observed_items[user_id] = set()
                user_observed_items[user_id] |= set(item_ids)
    return user_observed_items


def _load_entity_item_map(
    data_path: str, entity_size: int, user_size: int, item_size: int
) -> torch.Tensor:
    entity_item_map = np.zeros((entity_size, item_size), dtype=bool)
    with open(f"{data_path}/kg.txt") as f:
        for line in f:
            h, _, t = map(int, line.split())
            if t < item_size:
                entity_item_map[h][t] = 1
    t = torch.tensor(entity_item_map, dtype=torch.bool)
    return t[(item_size + user_size) :]


def _load_info(
    data_path: str,
) -> tuple[int, int, int, list[int], list[int], list[int], list[int]]:
    data = []
    with open(f"{data_path}/info.txt") as f:
        for line in f:
            data.append(list(map(int, line.split())))
    user_size = data[0][0]
    item_size = data[0][1]
    entity_size = data[0][2]
    item_relations = data[1]
    user_preference_relations = data[2]
    user_current_relations = data[3]
    return (
        user_size,
        item_size,
        entity_size,
        item_relations,
        user_preference_relations,
        user_current_relations,
    )


def _diversity_data_augmentation(
    data_path: str,
    user_size: int,
    item_size: int,
    threshold_user: float,
    threshold_item: float,
    item_relations: list[int],
    user_preference_relations: list[int],
    user_current_relations: list[int],
) -> dict[int, list[int]]:
    interactions = {}
    with open(f"{data_path}/train.txt") as f:
        for line in f:
            user_id, *item_ids = list(map(lambda x: int(x), line.split(" ")))
            interactions[user_id] = item_ids
    preferences = {}
    currents = {}
    items = {}
    with open(f"{data_path}/kg.txt") as f:
        for line in f:
            h, r, t = map(int, line.split())
            if r in user_preference_relations:
                preferences.setdefault(t - item_size, set()).add(h)
            elif r in user_current_relations:
                currents.setdefault(t - item_size, set()).add(h)
            elif r in item_relations:
                items.setdefault(h, set()).add(t)
            else:
                raise ValueError(f"Unknown relation {r} in kg.txt")

    data = {}
    for target_user_id in range(user_size):
        for user_id in range(user_size):
            preference = preferences.get(target_user_id, set())
            current = currents.get(user_id, set())
            jaccard = len(preference & current) / len(preference | current)
            if jaccard < threshold_user:
                continue
            for item_id in interactions[user_id]:
                item_attributes = items.get(item_id, set())
                item_jaccard = len(preference & item_attributes) / len(
                    preference | item_attributes
                )
                dissimilarity = 1 - item_jaccard
                if dissimilarity < threshold_item:
                    continue
                data.setdefault(target_user_id, set()).add(item_id)

    data = {k: list(v) for k, v in data.items()}
    return data
