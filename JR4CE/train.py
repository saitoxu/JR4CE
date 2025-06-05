import os
from time import time

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .dataset import TrainDataset, eval_dataset
from .metrics import evaluate
from .model import Model
from .parser import parse_args
from .utils import EarlyStopping, getLogger, seed_everything


def train(
    train_dataloader: DataLoader,
    graph_data,
    model,
    optimizer,
    device,
    logger,
    scheduler,
    args,
):
    size = len(train_dataloader.dataset)
    model.train()

    for batch, data in enumerate(train_dataloader):
        users, pos_items, neg_item_lists, div_pos_items = data
        users, pos_items, neg_item_lists, div_pos_items = (
            users.to(device),
            pos_items.to(device),
            neg_item_lists.to(device),
            div_pos_items.to(device),
        )

        rec_loss = model.rec_loss(users, pos_items, neg_item_lists, graph_data)

        index = div_pos_items > 0
        _users = users[index]
        _div_pos_items = div_pos_items[index]
        _neg_item_lists = neg_item_lists[index]
        if args.div_lambda > 0.0 and len(_users) > 0:
            div_loss = model.rec_loss(
                _users, _div_pos_items, _neg_item_lists, graph_data
            )
            loss = rec_loss + args.div_lambda * div_loss
        else:
            loss = rec_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(users)
            logger.debug(f"Loss: {loss:>7f} [{current:>8d}/{size:>8d}]")

    scheduler.step()


if __name__ == "__main__":
    start = int(time())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = parse_args()
    seed_everything(args.seed)

    logger = getLogger(name=__name__, path=args.save_path + "/")

    for key, value in vars(args).items():
        logger.debug(f"{key}: {value}")

    data_path = f"{args.data_path}/{args.dataset}"
    num_workers = 2 if os.cpu_count() > 1 else 0

    train_dataset = TrainDataset(
        data_path=data_path,
        neg_sample_size=args.neg_size,
        threshold_user=args.threshold_user,
        threshold_item=args.threshold_item,
    )
    graph_data = train_dataset.data(device)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=num_workers, shuffle=True
    )
    val_data = eval_dataset(data_path, "val.txt")

    weight_decay = 0.0001
    model = Model(
        user_size=train_dataset.user_size,
        item_size=train_dataset.item_size,
        entity_size=train_dataset.entity_size,
        dim=args.dim,
        num_gcn_layer=args.num_gcn_layer,
        kg_module=args.kg_module == 1,
        cf_module=args.cf_module == 1,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)
    early_stop = EarlyStopping(args.patience)

    Ks = eval(args.Ks)
    epoch = args.epoch
    val_interval = args.val_interval
    evaluate(val_data, model, Ks, device, graph_data, logger.debug)
    for t in range(args.epoch):
        logger.debug(f"Epoch {t+1}")
        logger.debug("-" * 32)
        train(
            train_dataloader,
            graph_data,
            model,
            optimizer,
            device,
            logger,
            scheduler,
            args,
        )
        if (t + 1) % val_interval == 0:
            mrr, _, _ = evaluate(val_data, model, Ks, device, graph_data, logger.debug)
            # early stopping
            should_save, should_stop = early_stop(mrr)
            if should_save:
                torch.save(model, args.save_path + "/best.pth")
            if should_stop:
                epoch = t + 1
                logger.debug("Early stopping.")
                break
    end = int(time())
    logger.debug("Done!")
