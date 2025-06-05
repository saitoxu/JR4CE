import numpy as np
import torch

item_entity_map = {}


def evaluate(eval_data, model, Ks, device, graph_data, log=print):
    global item_entity_map
    model.eval()
    users = []
    ranks = []
    ranking_lists = []
    batch_size = 16

    if not item_entity_map:
        entity_item_map = graph_data["entity_item_map"]
        for i in range(model.item_size):
            entity_ids = torch.nonzero(entity_item_map.T[i]).squeeze().tolist()
            if isinstance(entity_ids, int):
                entity_ids = [entity_ids]
            item_entity_map[i] = set(entity_ids)

    def create_batch_items(batch, item_size):
        pos_items = []
        observed_items_list = []
        all_items = set(list(range(item_size)))
        for data in batch:
            pos_item = data[1]
            unobserved_items = data[2]
            observed_items = list(all_items - set(unobserved_items) - {pos_item})
            pos_items.append(pos_item)
            observed_items_list.append(observed_items)
        return observed_items_list, pos_items

    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            batch = eval_data[i : i + batch_size]
            batch_users = torch.tensor([data[0] for data in batch]).to(device)
            observed_items_list, pos_items = create_batch_items(batch, model.item_size)

            all_items = torch.tensor(list(range(model.item_size))).to(device)
            batch_items = all_items.repeat(len(batch), 1)
            _users, _items = model(graph_data)
            _users = _users[batch_users].unsqueeze(1)
            _items = _items[batch_items]
            preds = torch.sum(_users * _items, dim=-1)
            for user, observed_items, pos_item, pred in zip(
                batch_users, observed_items_list, pos_items, preds
            ):
                pred[observed_items] = -np.inf
                indices = torch.argsort(pred, descending=True).cpu()
                rank = np.where(indices == pos_item)[0][0] + 1
                ranking = indices.tolist()[: -len(observed_items)]
                users.append(int(user.cpu()))
                ranks.append(rank)
                ranking_lists.append(ranking)

    mrr, hrs, ndcgs, diversities, ccs = _calc_metrics(
        ranks, Ks, ranking_lists, item_entity_map, model.attr_size
    )
    _display_metrics(mrr, hrs, ndcgs, diversities, ccs, log)
    return mrr, hrs, ndcgs


def _calc_metrics(ranks, Ks, ranking_lists, item_entity_map, attr_size):
    diversities = []
    for ranking, rank in zip(ranking_lists, ranks):
        _diversities = _intra_list_similarity(ranking, Ks, item_entity_map, rank)
        diversities.append(_diversities)
    diversities = np.array(diversities).mean(axis=0).tolist()
    return (
        _mrr(ranks),
        _hit_ratios(ranks, Ks),
        _ndcgs(ranks, Ks),
        diversities,
        _category_coverage(ranking_lists, Ks, item_entity_map, attr_size),
    )


def _category_coverage(
    ranking_lists: list[list[int]],
    Ks: list[int],
    item_entity_map: dict[int, set],
    attr_size: int,
) -> list[float]:
    total_category_size = attr_size
    results = []
    for k in Ks:
        ccs = []
        for ranking_list in ranking_lists:
            categories_in_ranking = set()
            for i in range(k):
                job_id = ranking_list[i]
                job_categories = item_entity_map[job_id]
                categories_in_ranking |= job_categories
            category_size = len(categories_in_ranking)
            cc = category_size / total_category_size
            ccs.append(cc)
        result = sum(ccs) / len(ccs)
        results.append(result)
    return results


def _mrr(ranks):
    return np.array(list(map(lambda x: 1 / x, ranks))).sum() / len(ranks)


def _hit_ratios(ranks, Ks):
    results = []
    for k in Ks:
        hr = len(list(filter(lambda x: x <= k, ranks))) / len(ranks)
        results.append(hr)
    return results


def _ndcgs(ranks, Ks):
    results = []
    for k in Ks:
        ndcg = np.array(
            list(
                map(lambda x: 1 / np.log2(x + 1), list(filter(lambda x: x <= k, ranks)))
            )
        ).sum() / len(ranks)
        results.append(ndcg)
    return results


def _intra_list_similarity(
    ranks: list[int],
    Ks: list[int],
    item_entity_map: dict[int, set[int]],
    rank: int,
    beta: float = 1.0,
) -> list[float]:
    diversities = []
    for k in Ks:
        diversities_k = []
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                item_i = ranks[i]
                item_j = ranks[j]
                d = _disimilarity(item_i, item_j, item_entity_map)
                w_i = 1 if i == rank - 1 else beta
                w_j = 1 if j == rank - 1 else beta
                d = w_i * w_j * d
                diversities_k.append(d)
        diversity_k = sum(diversities_k) / len(diversities_k)
        diversities.append(diversity_k)
    return diversities


def _disimilarity(i: int, j: int, item_entity_map: dict[int, set[int]]) -> float:
    i_attributes = item_entity_map[i]
    j_attributes = item_entity_map[j]
    overlap_coefficient = len(i_attributes & j_attributes) / min(
        len(i_attributes), len(j_attributes)
    )
    return 1.0 - overlap_coefficient


def _display_metrics(mrr, hrs, ndcgs, diversities, ccs, log):
    rounded_hrs = list(map(lambda x: float(f"{x:>7f}"), hrs))
    rounded_ndcgs = list(map(lambda x: float(f"{x:>7f}"), ndcgs))
    rounded_diversities = list(map(lambda x: float(f"{x:>7f}"), diversities))
    rounded_ccs = list(map(lambda x: float(f"{x:>7f}"), ccs))
    log(f"MRR:\t{mrr:>7f}")
    log(f"HRs:\t{rounded_hrs}")
    log(f"nDCGs:\t{rounded_ndcgs}")
    log(f"Divs:\t{rounded_diversities}")
    log(f"CCs:\t{rounded_ccs}")
