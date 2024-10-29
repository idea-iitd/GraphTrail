"""GNN model from EiG-Search"""

# * >>> gin_conv.py
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch.nn import Linear, ReLU, Sequential
from torch.nn import BatchNorm1d as BN
import torch.nn.functional as F
from typing import Callable, Optional, Union, Any

import torch
from torch import Tensor
# from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size


def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)


class GINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False, layer_number=-1,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        self.layer_number = layer_number
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None,
                size: Size = None) -> Tensor:
        """"""

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(
            edge_index, x=x, edge_weight=edge_weight, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r
        return self.nn(out)

    # def message(self, x_j: Tensor) -> Tensor:
    #     return x_j

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    # def message_and_aggregate(self, adj_t: SparseTensor,
    #                           x: OptPairTensor) -> Tensor:
    #     adj_t = adj_t.set_value(None, layout=None)
    #     return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class GINEConv(MessagePassing):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 edge_dim: Optional[int] = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            if hasattr(self.nn[0], 'in_features'):
                in_channels = self.nn[0].in_features
            else:
                in_channels = self.nn[0].in_channels
            self.lin = Linear(edge_dim, in_channels)
        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


# * gin.py


class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                ReLU(inplace=False),
                Linear(hidden, hidden),
                ReLU(inplace=False),
                BN(hidden),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(inplace=False),
                        Linear(hidden, hidden),
                        ReLU(inplace=False),
                        BN(hidden),
                    ), train_eps=True))
        
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch):
        # x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        node_embs = x

        x = global_mean_pool(x, batch)
        graph_emb = x

        # x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        out = x

        # out = F.log_softmax(x, dim=-1)
        return out, node_embs, graph_emb
    
    def fc(self, emb):
        x = F.relu(self.lin1(emb))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    # def get_gemb(self, data):
    #     x, edge_index, batch = data.x.float(), data.edge_index, data.batch
    #     x = self.conv1(x, edge_index)
    #     for conv in self.convs:
    #         x = conv(x, edge_index)
    #     x = global_mean_pool(x, batch)
    #     return x

    # def fwd_weight(self, x, edge_index, edge_weight=None):
    #     batch = torch.zeros(x.shape[0]).to(x.device).type(torch.int64)
    #     if edge_weight is None:
    #         edge_weight = torch.ones(
    #             edge_index.shape[1]).float().to(edge_index.device)
    #     x = self.conv1(x, edge_index)
    #     for conv in self.convs:
    #         x = conv(x, edge_index)
    #     x = global_mean_pool(x, batch)
    #     x = F.relu(self.lin1(x))
    #     x = F.dropout(x, p=0.5, training=self.training)
    #     x = self.lin2(x)
    #     return F.log_softmax(x, dim=-1)

    # def fwd(self, x, edge_index, de=None, epsilon=None, edge_weight=None):
    #     batch = torch.zeros(x.shape[0]).to(x.device).type(torch.int64)
    #     if edge_weight is None:
    #         edge_weight = torch.ones(
    #             edge_index.shape[1]).float().to(edge_index.device)
    #     if de is not None:
    #         edge_weight[de] = epsilon
    #         edl, edr = edge_index[0, de], edge_index[1, de]
    #         rev_de = int((torch.logical_and(
    #             edge_index[0] == edr, edge_index[1] == edl) == True).nonzero()[0])
    #         edge_weight[rev_de] = epsilon
    #     x = self.conv1(x.float(), edge_index, edge_weight=edge_weight)
    #     for o, conv in enumerate(self.convs):
    #         x = conv(x, edge_index, edge_weight=edge_weight)
    #     x = global_mean_pool(x, batch)
    #     x = F.relu(self.lin1(x))
    #     x = F.dropout(x, p=0.5, training=self.training)
    #     x = self.lin2(x)
    #     # return x
    #     return F.log_softmax(x, dim=-1)

    # def fwd_cam(self, data, edge_weight):
    #     x, edge_index, batch = data.x.float(), data.edge_index, data.batch
    #     x = self.conv1(x, edge_index, edge_weight=edge_weight)
    #     for conv in self.convs:
    #         x = conv(x, edge_index, edge_weight=edge_weight)
    #     x = global_mean_pool(x, batch)
    #     # x = global_add_pool(x, batch)
    #     x = F.relu(self.lin1(x))
    #     x = F.dropout(x, p=0.5, training=self.training)
    #     x = self.lin2(x)
    #     # return F.softmax(x, dim=-1)
    #     return x

    # def fwd_base(self, x, edge_index):
    #     x, edge_index = x.float(), edge_index
    #     batch = torch.zeros(x.shape[0]).to(x.device).type(torch.int64)

    #     x = self.conv1(x, edge_index)
    #     for conv in self.convs:
    #         x = conv(x, edge_index)
    #     x = global_mean_pool(x, batch)
    #     x = F.relu(self.lin1(x))
    #     x = F.dropout(x, p=0.5, training=self.training)
    #     x = self.lin2(x)
    #     return x

    # def fwd_base_other(self, x, edge_index, ie, value):
    #     batch = torch.zeros(x.shape[0]).to(x.device).type(torch.int64)
    #     edge_weight = torch.ones(
    #         edge_index.shape[1]).float().to(edge_index.device)
    #     edge_weight[ie] = value

    #     x = self.conv1(x.float(), edge_index, edge_weight=edge_weight)
    #     for conv in self.convs:
    #         x = conv(x, edge_index, edge_weight=edge_weight)
    #     x = global_mean_pool(x, batch)
    #     x = F.relu(self.lin1(x))
    #     x = F.dropout(x, p=0.5, training=self.training)
    #     x = self.lin2(x)
    #     return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
