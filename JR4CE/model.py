import torch
from torch import nn
from torch.nn import functional as F

from .gat import GAT


class Model(torch.nn.Module):
    def __init__(
        self,
        user_size: int,
        item_size: int,
        entity_size: int,
        dim: int,
        num_gcn_layer: int,
        kgl_module: bool,
        cf_module: bool,
        device: str,
        use_edge_type: bool = False,
        edge_type_size: int = 0,
    ):
        super(Model, self).__init__()

        self.user_size = user_size
        self.item_size = item_size
        self.entity_size = entity_size
        self.attr_size = self.entity_size - self.item_size - self.user_size
        self.dim = dim
        self.device = device
        self.embed = nn.Embedding(self.entity_size, dim)
        nn.init.xavier_uniform_(self.embed.weight, gain=nn.init.calculate_gain("relu"))
        self.gat = GAT(
            features=self.dim,
            num_layers=1,
            num_heads=1,
            dropout=0.1,
            add_self_loops=False,
        )
        self.gcn_layers = nn.ModuleList([GCNLayer() for _ in range(num_gcn_layer)])
        self._kgl_module = kgl_module
        self._cf_module = cf_module
        self.use_edge_type = use_edge_type
        if use_edge_type and edge_type_size > 0:
            self.edge_embed = nn.Embedding(edge_type_size, dim)
            nn.init.xavier_uniform_(
                self.edge_embed.weight, gain=nn.init.calculate_gain("relu")
            )
        else:
            self.edge_embed = None

    def forward(self, data):
        items = self._item_representations(data)
        users = self._user_representations(data)
        ui_adj_mtx = self._create_ui_adj_mtx(data)

        if not self._cf_module:
            return users, items

        users_and_items = torch.concat([users, items], axis=0)
        embeds_list = [users_and_items]
        for gcn in self.gcn_layers:
            embeds = gcn(ui_adj_mtx, embeds_list[-1])
            embeds_list.append(embeds)
        embeds: torch.Tensor = sum(embeds_list[1:]) / len(embeds_list[1:])
        embeds = F.normalize(embeds)

        items = embeds[self.user_size :]
        users = embeds[: self.user_size]

        return users, items

    def _create_ui_adj_mtx(self, data):
        return normalize_adj_matrix(data["ui_adj_mtx"], device=self.device)

    def _item_representations(self, data) -> torch.Tensor:
        if not self._kgl_module:
            return self.embed.weight[: self.item_size, :]

        edge_attr = None
        if self.use_edge_type and self.edge_embed is not None:
            edge_attr = self.edge_embed(data["item_edge_type"])

        entities = self.gat(
            self.embed.weight, data["item_edge_index"].T, edge_attr=edge_attr
        )
        return entities[: self.item_size, :]

    def _user_representations(self, data) -> torch.Tensor:
        if not self._kgl_module:
            return self.embed.weight[self.item_size + self.attr_size :, :]

        edge_index = torch.cat(
            [data["user_current_edge_index"], data["user_preference_edge_index"]], dim=0
        )
        edge_attr = None
        if self.use_edge_type and self.edge_embed is not None:
            edge_type = torch.cat(
                [data["user_current_edge_type"], data["user_preference_edge_type"]],
                dim=0,
            )
            edge_attr = self.edge_embed(edge_type)

        entities = self.gat(self.embed.weight, edge_index.T, edge_attr=edge_attr)
        users = entities[self.item_size : self.item_size + self.user_size]
        return users

    def rec_loss(self, users, pos_items, neg_item_lists, data):
        _users, _items = self(data)
        _users = _users[users]  # (batch_size, dim)
        _pos_items = _items[pos_items]  # (batch_size, dim)
        neg_items = _items[neg_item_lists]  # (batch_size, neg_size, dim)

        preds = torch.sum(_users * _pos_items, dim=-1).unsqueeze(-1)  # (batch_size, 1)
        neg_preds = torch.sum(
            _users.unsqueeze(1) * neg_items, dim=-1
        )  # (batch_size, neg_size)

        rec_loss = -torch.sum(torch.log(torch.sigmoid(preds - neg_preds)))
        return rec_loss


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return torch.spmm(adj, embeds)


def normalize_adj_matrix(adj_matrix, device="cuda"):
    """
    Normalize adjacency matrix using symmetric normalization.
    Input can be either a dense or sparse tensor.

    Args:
        adj_matrix (torch.Tensor): Input adjacency matrix
        device (str): Device to perform computation on ('cuda' or 'cpu')

    Returns:
        torch.Tensor: Normalized adjacency matrix in COO format
    """
    # Move input to specified device
    adj_matrix = adj_matrix.to(device)

    # If input is dense, convert to sparse
    if not adj_matrix.is_sparse:
        adj_matrix = adj_matrix.to_sparse()

    # Calculate degree
    degrees = torch.sparse.sum(adj_matrix, dim=1).to_dense()

    # Calculate D^(-1/2)
    d_inv_sqrt = torch.pow(degrees, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0

    # Get indices and values from sparse matrix
    indices = adj_matrix.indices()
    values = adj_matrix.values()

    # Apply normalization
    normalized_values = values * d_inv_sqrt[indices[0]] * d_inv_sqrt[indices[1]]

    # Create normalized sparse matrix
    normalized_adj = torch.sparse_coo_tensor(
        indices, normalized_values, adj_matrix.size(), device=device
    )

    return normalized_adj
