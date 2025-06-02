import pickle
import random

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

R_IND_JOB = 0
R_OCC_JOB = 1
R_EMP_JOB = 2

R_HOPE_IND_USER = 3
R_HOPE_OCC_USER = 4
R_HOPE_EMP_USER = 5

R_CUR_IND_USER = 6
R_CUR_OCC_USER = 7
R_CUR_EMP_USER = 8

R_INC_JOB = 9
R_HOPE_INC_USER = 10
R_CUR_INC_USER = 11


class TrainDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        neg_sample_size: int,
        threshold_user: float,
        threshold_item: float,
        neg_uniform: bool,
    ):
        self.user_size = _get_size(f"{data_path}/user_original_id_map.txt")
        self.item_size = _get_size(f"{data_path}/item_original_id_map.txt")
        industry_size = _get_size(f"{data_path}/industry_original_id_map.txt")
        job_type_size = _get_size(f"{data_path}/job_type_original_id_map.txt")
        employment_type_size = _get_size(
            f"{data_path}/employment_type_original_id_map.txt"
        )
        annual_income_size = _get_size(f"{data_path}/annual_income_original_id_map.txt")
        self.entity_size = (
            self.user_size
            + self.item_size
            + industry_size
            + job_type_size
            + employment_type_size
            + annual_income_size
        )
        self._data, self.ui_adj_mtx = self._load_data(
            self.user_size, self.item_size, data_path
        )
        self._all_item_ids = set(list(range(self.item_size)))
        self._neg_sample_size = neg_sample_size
        self._neg_uniform = neg_uniform
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
        ) = _load_kg(data_path)
        self.user_neg_sampling_weights = _user_neg_sampling_weights(
            data_path,
            self.entity_size,
            self.item_size,
            self.item_edge_index,
            self.user_current_edge_index,
            self.user_preference_edge_index,
            self._user_observed_items,
        )
        self._div_data = _preference_beyond_items(
            data_path, self.user_size, self.item_size, threshold_user, threshold_item
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
        if self._neg_uniform:
            # 一様分布からサンプリング
            weights = np.ones(self.item_size)
            weights[list(self._user_observed_items[user_id])] = 0
            weights = weights / np.sum(weights)
        else:
            # ユーザーに近いアイテムを高い確率でサンプリング
            weights = self.user_neg_sampling_weights[user_id]
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
    item_size = _get_size(f"{data_path}/item_original_id_map.txt")
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


def _load_kg(data_path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            if r in [R_IND_JOB, R_OCC_JOB, R_EMP_JOB, R_INC_JOB]:
                item_edge_index.append([h, t])
                item_edge_type.append(r)
                num_relations = max(num_relations, r + 1)
            elif r in [R_CUR_IND_USER, R_CUR_OCC_USER, R_CUR_EMP_USER, R_CUR_INC_USER]:
                user_current_edge_index.append([h, t])
                user_current_edge_type.append(r)
                num_relations = max(num_relations, r + 1)
            elif r in [
                R_HOPE_IND_USER,
                R_HOPE_OCC_USER,
                R_HOPE_EMP_USER,
                R_HOPE_INC_USER,
            ]:
                user_preference_edge_index.append([h, t])
                user_preference_edge_type.append(r)
                num_relations = max(num_relations, r + 1)
    return (
        torch.tensor(item_edge_index),
        torch.tensor(item_edge_type),
        torch.tensor(user_current_edge_index),
        torch.tensor(user_current_edge_type),
        torch.tensor(user_preference_edge_index),
        torch.tensor(user_preference_edge_type),
        num_relations,
    )


def _get_size(data_path: str) -> int:
    try:
        _size = 0
        with open(data_path) as f:
            for _ in f:
                _size += 1
        return _size
    except FileNotFoundError:
        return 0


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


def _user_neg_sampling_weights(
    data_path: str,
    entity_size: int,
    item_size: int,
    item_edge_index: torch.Tensor,
    user_current_edge_index: torch.Tensor,
    user_preference_edge_index: torch.Tensor,
    user_observed_items: dict[int, set[int]],
):
    user_neg_sampling_weights = {}
    try:
        with open(f"{data_path}/user_neg_sampling_weights.pickle", "rb") as f:
            user_neg_sampling_weights = pickle.load(f)
            return user_neg_sampling_weights
    except FileNotFoundError:
        pass
    edge_index = torch.cat(
        [item_edge_index, user_current_edge_index, user_preference_edge_index], dim=0
    )
    data = Data(edge_index=edge_index.T, num_nodes=entity_size)
    graph = to_networkx(data, to_undirected=True)

    for user_id, observed_items in user_observed_items.items():
        start_node = item_size + user_id
        max_hop = 0
        weights = []
        for item_id in range(item_size):
            end_node = item_id
            hop = nx.shortest_path_length(graph, start_node, end_node)
            max_hop = max(max_hop, hop)
            weights.append(hop)
        weights = np.array(weights, dtype=float)
        weights = max_hop - weights + 1
        weights[list(observed_items)] = 0
        weights /= np.sum(weights)
        user_neg_sampling_weights[user_id] = weights

    with open(f"{data_path}/user_neg_sampling_weights.pickle", "wb") as f:
        pickle.dump(user_neg_sampling_weights, f)

    return user_neg_sampling_weights


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


def _preference_beyond_items(
    data_path: str,
    user_size: int,
    item_size: int,
    threshold_user: float,
    threshold_item: float,
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
            if r in [
                R_HOPE_IND_USER,
                R_HOPE_OCC_USER,
                R_HOPE_EMP_USER,
                R_HOPE_INC_USER,
            ]:
                preferences.setdefault(t - item_size, set()).add(h)
            elif r in [R_CUR_IND_USER, R_CUR_OCC_USER, R_CUR_EMP_USER, R_CUR_INC_USER]:
                currents.setdefault(t - item_size, set()).add(h)
            elif r in [R_IND_JOB, R_OCC_JOB, R_EMP_JOB, R_INC_JOB]:
                items.setdefault(h, set()).add(t)

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


if __name__ == "__main__":
    data_path = "datasets/glit4th_2022"
    dataset = TrainDataset(
        data_path=data_path,
        neg_sample_size=2,
        threshold_user=0.9,
        threshold_item=0.3,
        neg_uniform=True,
    )
    # user_ids = [59, 2810, 4617, 343, 91, 4114, 3565, 4185, 605, 4548, 2805]
    # for user_id in user_ids:
    #     print(user_id, len(dataset._div_data[user_id]))
    with open("item_sizes2.txt", "w") as f:
        for user_id, item_ids in dataset._div_data.items():
            f.write(f"{user_id} {len(item_ids)}\n")
