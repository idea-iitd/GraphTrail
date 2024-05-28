import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
# from torch_sparse import matmul


class GINConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(GINConv, self).__init__()

        mlp = torch.nn.Sequential(torch.nn.Linear(in_channels, out_channels),
                                  torch.nn.BatchNorm1d(out_channels),
                                  torch.nn.Linear(out_channels, out_channels),
                                  torch.nn.BatchNorm1d(out_channels))
        self.conv = GINConvWrapper(mlp, **kwargs)

    def forward(self, x, edge_index, edge_weight):
        return self.conv(x, edge_index, edge_weight)


class GINConvWrapper(MessagePassing):
    def __init__(self, nn, eps=0., train_eps=False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_weight, size=None):
        """"""
        if isinstance(x, torch.Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    # def message_and_aggregate(self, adj_t, x):
    #     adj_t = adj_t.set_value(None, layout=None)
    #     return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return f'{self.__class__.__name__}(nn={self.nn})'