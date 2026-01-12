import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(
        self, features, num_layers, num_heads, dropout=0.2, add_self_loops=True
    ):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList(
            [
                GATConv(
                    in_channels=features,
                    out_channels=features,
                    heads=num_heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, edge_index, edge_attr=None):
        _x = x
        for gat_layer in self.gat_layers:
            _x = gat_layer(_x, edge_index, edge_attr=edge_attr)
            _x = F.normalize(_x)
        return _x
