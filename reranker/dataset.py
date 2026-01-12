import numpy as np
import torch


def load_entity_item_map(
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


def get_user_size(data_path: str) -> int:
    # まずinfo.txtから読み取りを試みる
    info_path = f"{data_path}/info.txt"
    try:
        with open(info_path) as f:
            first_line = f.readline().strip()
            user_size, _, _ = map(int, first_line.split())
            return user_size
    except (FileNotFoundError, ValueError):
        # info.txtが存在しないか、読み取りに失敗した場合は従来の方法
        return _get_size(f"{data_path}/user_original_id_map.txt")


def get_item_size(data_path: str) -> int:
    # まずinfo.txtから読み取りを試みる
    info_path = f"{data_path}/info.txt"
    try:
        with open(info_path) as f:
            first_line = f.readline().strip()
            _, item_size, _ = map(int, first_line.split())
            return item_size
    except (FileNotFoundError, ValueError):
        # info.txtが存在しないか、読み取りに失敗した場合は従来の方法
        return _get_size(f"{data_path}/item_original_id_map.txt")


def get_entity_size(data_path: str) -> int:
    # まずinfo.txtから読み取りを試みる
    info_path = f"{data_path}/info.txt"
    try:
        with open(info_path) as f:
            first_line = f.readline().strip()
            _, _, entity_size = map(int, first_line.split())
            return entity_size
    except (FileNotFoundError, ValueError):
        # info.txtが存在しないか、読み取りに失敗した場合は従来の方法
        user_size = get_user_size(data_path)
        item_size = get_item_size(data_path)
        industry_size = _get_size(f"{data_path}/industry_original_id_map.txt")
        job_type_size = _get_size(f"{data_path}/job_type_original_id_map.txt")
        employment_type_size = _get_size(
            f"{data_path}/employment_type_original_id_map.txt"
        )
        annual_income_size = _get_size(f"{data_path}/annual_income_original_id_map.txt")
        return (
            user_size
            + item_size
            + industry_size
            + job_type_size
            + employment_type_size
            + annual_income_size
        )


def eval_dataset(data_path: str, data_file: str) -> list[tuple[int, int, list[int]]]:
    data = []
    # info.txtから読み取るためにget_item_sizeを使用
    item_size = get_item_size(data_path)
    all_item_ids = set(list(range(item_size)))
    user_observed_items = _load_user_observed_items(data_path)

    with open(f"{data_path}/{data_file}") as f:
        for line in f:
            user_id, item_id = list(map(lambda x: int(x), line.split(" ")))
            # コールドスタート設定の場合、user_idがtrainに存在しない可能性がある
            observed_items = user_observed_items.get(user_id, set())
            neg_item_ids = list(all_item_ids - observed_items - {item_id})
            neg_item_ids = sorted(neg_item_ids)
            data.append((user_id, item_id, neg_item_ids))
    return data


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
