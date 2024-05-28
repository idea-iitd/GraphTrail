import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import GCNConv, GATConv

from wrappers.gin import GINConv


class GAT_MUTAG(torch.nn.Module):
    def __init__(self, hidden_dim=20, dropout=0, pooling='add'):
        torch.manual_seed(7)
        super().__init__()
        num_node_features = 7
        num_edge_features = 4
        num_classes = 2
        gat_args = dict(
            edge_dim=num_edge_features,
            add_self_loops=False,
        )
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.num_layers = 3
        self.fc = torch.nn.Linear(hidden_dim, num_classes)
        self.pooling = pooling

        self.convs.append(GATConv(num_node_features, hidden_dim, **gat_args))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Follow-up GCN layers.
        for i in range(self.num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, **gat_args))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch, edge_attr):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            # Dropout after every layer.
            x = F.dropout(x, p=self.dropout, training=self.training)
        node_embs = x
        if self.pooling == 'add':
            graph_emb = pyg.nn.pool.global_add_pool(node_embs, batch)
        elif self.pooling == 'mean':
            graph_emb = pyg.nn.pool.global_mean_pool(node_embs, batch)
        elif self.pooling == 'max':
            graph_emb = pyg.nn.pool.global_max_pool(node_embs, batch)
        out = self.fc(graph_emb)
        return out, node_embs, graph_emb


class GCN_MUTAG(torch.nn.Module):
    def __init__(self, hidden_dim=20, dropout=0, pooling='add'):
        torch.manual_seed(7)
        super().__init__()
        gat_args = dict(
            add_self_loops=False,
        )
        num_node_features = 7
        num_edge_features = 4
        num_classes = 2
        gat_args = dict(
            edge_dim=num_edge_features,
            add_self_loops=False,
        )
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.num_layers = 3
        self.pooling = pooling

        self.convs.append(GCNConv(num_node_features, hidden_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Follow-up GCN layers.
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            # Dropout after every layer.
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_embs = x
        if self.pooling == 'add':
            graph_emb = pyg.nn.pool.global_add_pool(node_embs, batch)
        elif self.pooling == 'mean':
            graph_emb = pyg.nn.pool.global_mean_pool(node_embs, batch)
        elif self.pooling == 'max':
            graph_emb = pyg.nn.pool.global_max_pool(node_embs, batch)
        out = self.fc(graph_emb)

        return out, node_embs, graph_emb


class GIN_MUTAG(torch.nn.Module):
    def __init__(self, hidden_dim=20, dropout=0, pooling='add'):
        torch.manual_seed(7)
        super().__init__()
        gat_args = dict(
            add_self_loops=False,
        )
        num_node_features = 7
        num_classes = 2
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.num_layers = 3
        self.pooling = pooling

        self.convs.append(GINConv(num_node_features, hidden_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Follow-up GCN layers.
        for i in range(self.num_layers - 1):
            self.convs.append(GINConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            # Dropout after every layer.
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_embs = x
        if self.pooling == 'add':
            graph_emb = pyg.nn.pool.global_add_pool(node_embs, batch)
        elif self.pooling == 'mean':
            graph_emb = pyg.nn.pool.global_mean_pool(node_embs, batch)
        elif self.pooling == 'max':
            graph_emb = pyg.nn.pool.global_max_pool(node_embs, batch)
        out = self.fc(graph_emb)

        return out, node_embs, graph_emb


class GAT_Mutagenicity(torch.nn.Module):
    def __init__(self, hidden_dim=20, dropout=0, pooling='add'):
        torch.manual_seed(7)
        super().__init__()
        num_node_features = 14
        num_edge_features = 3
        num_classes = 2
        gat_args = dict(
            edge_dim=num_edge_features,
            add_self_loops=False,
        )
        self.pooling = pooling
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.num_layers = 3
        self.fc = torch.nn.Linear(hidden_dim, num_classes)
        self.convs.append(GATConv(num_node_features, hidden_dim, **gat_args))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        for i in range(self.num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, **gat_args))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch, edge_attr):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            # Dropout after every layer.
            x = F.dropout(x, p=self.dropout, training=self.training)
        node_embs = x
        if self.pooling == 'add':
            graph_emb = pyg.nn.pool.global_add_pool(node_embs, batch)
        elif self.pooling == 'mean':
            graph_emb = pyg.nn.pool.global_mean_pool(node_embs, batch)
        elif self.pooling == 'max':
            graph_emb = pyg.nn.pool.global_max_pool(node_embs, batch)
        out = self.fc(graph_emb)
        return out, node_embs, graph_emb


class GCN_Mutagenicity(torch.nn.Module):
    def __init__(self, hidden_dim=64, dropout=0, pooling='add'):
        torch.manual_seed(7)
        super().__init__()
        num_node_features = 14
        num_edge_features = 3
        num_classes = 2
        gat_args = dict(
            edge_dim=num_edge_features,
            add_self_loops=False,
        )
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.num_layers = 3
        self.pooling = pooling

        self.convs.append(GCNConv(num_node_features, hidden_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Follow-up GCN layers.
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            # Dropout after every layer.
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_embs = x
        if self.pooling == 'add':
            graph_emb = pyg.nn.pool.global_add_pool(node_embs, batch)
        elif self.pooling == 'mean':
            graph_emb = pyg.nn.pool.global_mean_pool(node_embs, batch)
        elif self.pooling == 'max':
            graph_emb = pyg.nn.pool.global_max_pool(node_embs, batch)
        out = self.fc(graph_emb)

        return out, node_embs, graph_emb


class GIN_Mutagenicity(torch.nn.Module):
    def __init__(self, hidden_dim=20, dropout=0, pooling='add'):
        torch.manual_seed(7)
        super().__init__()
        gat_args = dict(
            add_self_loops=False,
        )
        num_node_features = 14
        num_classes = 2
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.num_layers = 3
        self.pooling = pooling

        self.convs.append(GINConv(num_node_features, hidden_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Follow-up GCN layers.
        for i in range(self.num_layers - 1):
            self.convs.append(GINConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            # Dropout after every layer.
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_embs = x
        if self.pooling == 'add':
            graph_emb = pyg.nn.pool.global_add_pool(node_embs, batch)
        elif self.pooling == 'mean':
            graph_emb = pyg.nn.pool.global_mean_pool(node_embs, batch)
        elif self.pooling == 'max':
            graph_emb = pyg.nn.pool.global_max_pool(node_embs, batch)
        out = self.fc(graph_emb)

        return out, node_embs, graph_emb


class GAT_BAMultiShapesDataset(torch.nn.Module):
    def __init__(self, hidden_dim=20, dropout=0, pooling='add'):
        torch.manual_seed(7)
        super().__init__()
        num_node_features = 10
        num_classes = 2
        gat_args = dict(
            add_self_loops=False,
        )
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.num_layers = 3
        self.fc = torch.nn.Linear(hidden_dim, num_classes)
        self.pooling = pooling
        self.convs.append(GATConv(num_node_features, hidden_dim, **gat_args))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Follow-up GCN layers.
        for i in range(self.num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, **gat_args))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            # Dropout after every layer.
            x = F.dropout(x, p=self.dropout, training=self.training)
        node_embs = x
        if self.pooling == 'add':
            graph_emb = pyg.nn.pool.global_add_pool(node_embs, batch)
        elif self.pooling == 'mean':
            graph_emb = pyg.nn.pool.global_mean_pool(node_embs, batch)
        elif self.pooling == 'max':
            graph_emb = pyg.nn.pool.global_max_pool(node_embs, batch)
        out = self.fc(graph_emb)
        return out, node_embs, graph_emb


class GIN_BAMultiShapesDataset(torch.nn.Module):
    def __init__(self, hidden_dim=20, dropout=0, pooling='add'):
        torch.manual_seed(7)
        super().__init__()
        gat_args = dict(
            add_self_loops=False,
        )
        num_node_features = 10
        num_classes = 2
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.num_layers = 3
        self.pooling = pooling

        self.convs.append(GINConv(num_node_features, hidden_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Follow-up GCN layers.
        for i in range(self.num_layers - 1):
            self.convs.append(GINConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            # Dropout after every layer.
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_embs = x
        if self.pooling == 'add':
            graph_emb = pyg.nn.pool.global_add_pool(node_embs, batch)
        elif self.pooling == 'mean':
            graph_emb = pyg.nn.pool.global_mean_pool(node_embs, batch)
        elif self.pooling == 'max':
            graph_emb = pyg.nn.pool.global_max_pool(node_embs, batch)
        out = self.fc(graph_emb)

        return out, node_embs, graph_emb


class GCN_BAMultiShapesDataset(torch.nn.Module):
    def __init__(self, hidden_dim=20, dropout=0, pooling='add'):
        torch.manual_seed(7)
        super().__init__()
        gat_args = dict(
            add_self_loops=False,
        )
        num_node_features = 10
        num_classes = 2
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.num_layers = 3
        self.pooling = pooling

        self.convs.append(GCNConv(num_node_features, hidden_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Follow-up GCN layers.
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            # Dropout after every layer.
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_embs = x
        if self.pooling == 'add':
            graph_emb = pyg.nn.pool.global_add_pool(node_embs, batch)
        elif self.pooling == 'mean':
            graph_emb = pyg.nn.pool.global_mean_pool(node_embs, batch)
        elif self.pooling == 'max':
            graph_emb = pyg.nn.pool.global_max_pool(node_embs, batch)
        out = self.fc(graph_emb)

        return out, node_embs, graph_emb


class GAT_NCI1(torch.nn.Module):
    def __init__(self, hidden_dim=20, dropout=0, pooling='add'):
        torch.manual_seed(7)
        super().__init__()
        num_node_features = 37
        num_classes = 2
        gat_args = dict(
            add_self_loops=False,
        )
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.num_layers = 3
        self.fc = torch.nn.Linear(hidden_dim, num_classes)
        self.pooling = pooling
        self.convs.append(GATConv(num_node_features, hidden_dim, **gat_args))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Follow-up GCN layers.
        for i in range(self.num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, **gat_args))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            # Dropout after every layer.
            x = F.dropout(x, p=self.dropout, training=self.training)
        node_embs = x
        if self.pooling == 'add':
            graph_emb = pyg.nn.pool.global_add_pool(node_embs, batch)
        elif self.pooling == 'mean':
            graph_emb = pyg.nn.pool.global_mean_pool(node_embs, batch)
        elif self.pooling == 'max':
            graph_emb = pyg.nn.pool.global_max_pool(node_embs, batch)
        out = self.fc(graph_emb)
        return out, node_embs, graph_emb


class GIN_NCI1(torch.nn.Module):
    def __init__(self, hidden_dim=20, dropout=0, pooling='add'):
        torch.manual_seed(7)
        super().__init__()
        gat_args = dict(
            add_self_loops=False,
        )
        num_node_features = 37
        num_classes = 2
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.num_layers = 3
        self.pooling = pooling

        self.convs.append(GINConv(num_node_features, hidden_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Follow-up GCN layers.
        for i in range(self.num_layers - 1):
            self.convs.append(GINConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            # Dropout after every layer.
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_embs = x
        if self.pooling == 'add':
            graph_emb = pyg.nn.pool.global_add_pool(node_embs, batch)
        elif self.pooling == 'mean':
            graph_emb = pyg.nn.pool.global_mean_pool(node_embs, batch)
        elif self.pooling == 'max':
            graph_emb = pyg.nn.pool.global_max_pool(node_embs, batch)
        out = self.fc(graph_emb)

        return out, node_embs, graph_emb


class GCN_NCI1(torch.nn.Module):
    def __init__(self, hidden_dim=64, dropout=0, pooling='add'):
        torch.manual_seed(7)
        super().__init__()
        gat_args = dict(
            add_self_loops=False,
        )
        num_node_features = 37
        num_classes = 2
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.num_layers = 3
        self.pooling = pooling

        self.convs.append(GCNConv(num_node_features, hidden_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # Follow-up GCN layers.
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            # Dropout after every layer.
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_embs = x
        if self.pooling == 'add':
            graph_emb = pyg.nn.pool.global_add_pool(node_embs, batch)
        elif self.pooling == 'mean':
            graph_emb = pyg.nn.pool.global_mean_pool(node_embs, batch)
        elif self.pooling == 'max':
            graph_emb = pyg.nn.pool.global_max_pool(node_embs, batch)
        out = self.fc(graph_emb)

        return out, node_embs, graph_emb
