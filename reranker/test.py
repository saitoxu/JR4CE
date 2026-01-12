import time

import torch
from torch import nn

from JR4CE.utils import seed_everything
from reranker.dataset import (
    eval_dataset,
    get_entity_size,
    get_item_size,
    get_user_size,
    load_entity_item_map,
)
from reranker.dpp import DPP
from reranker.metrics import evaluate
from reranker.mmr import MMR
from reranker.parser import parse_args


class RankingModel(nn.Module):
    def __init__(
        self,
        user_size: int,
        item_size: int,
        entity_size,
        entity_item_map: torch.Tensor,
        users: torch.Tensor,
        items: torch.Tensor,
    ):
        super(RankingModel, self).__init__()
        self.user_size = user_size
        self.item_size = item_size
        self.attr_size = entity_size - user_size - item_size
        self.entity_item_map = entity_item_map
        self.users = users
        self.items = items

    def forward(self):
        return self.users, self.items


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    args = parse_args()
    seed_everything(args.seed)

    data_path = f"{args.data_path}/{args.dataset}"
    test_data = eval_dataset(data_path, "test.txt")

    user_size = get_user_size(data_path)
    item_size = get_item_size(data_path)
    entity_size = get_entity_size(data_path)
    entity_item_map = load_entity_item_map(data_path, entity_size, user_size, item_size)

    # embeddingパスの読み込み（引数で指定されたパスを優先）
    if args.user_embeddings_path:
        users = torch.load(args.user_embeddings_path)
    else:
        raise ValueError("user_embeddings_path must be specified")

    if args.item_embeddings_path:
        items = torch.load(args.item_embeddings_path)
    else:
        raise ValueError("item_embeddings_path must be specified")

    # embeddingのサイズをチェック
    if users.shape[0] != user_size:
        raise ValueError(
            f"User embedding size mismatch: expected {user_size}, got {users.shape[0]}"
        )
    if items.shape[0] != item_size:
        raise ValueError(
            f"Item embedding size mismatch: expected {item_size}, got {items.shape[0]}"
        )

    model = RankingModel(
        user_size=user_size,
        item_size=item_size,
        entity_size=entity_size,
        entity_item_map=entity_item_map,
        users=users,
        items=items,
    ).to(device)

    if args.model == "mmr":
        reranker = MMR(lambda_factor=args.lambda_factor, model=model)
    elif args.model == "dpp":
        reranker = DPP(temperature=args.temperature, model=model)
    else:
        raise ValueError("Invalid model name.")

    Ks = eval(args.Ks)

    # 実行時間の計測
    start_time = time.time()
    evaluate(
        test_data,
        model,
        reranker,
        Ks,
        device,
        log=print,
        save_ranking=True,
        use_parallel=args.use_parallel,
        n_workers=args.n_workers,
    )
    elapsed_time = time.time() - start_time

    print(f"\n実行時間: {elapsed_time:.2f}秒")
    if args.use_parallel:
        print(f"並列処理を使用（ワーカー数: {args.n_workers}）")
    else:
        print("逐次処理を使用")
