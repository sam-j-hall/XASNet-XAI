# --- Load modules
from __future__ import annotations
import warnings
import os
import os.path as osp
import numpy as np
from typing import List, Optional, Any, Callable, Dict
# --- Pytorch
import torch
import torch.nn.functional as F
from torch.nn import Linear, LSTM, ModuleList, Dropout, Embedding
# --- PyG
import torch_geometric.nn as geomnn
from torch_geometric.io import fs
from torch_geometric.nn import MessagePassing, SumAggregation
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.data import Dataset, download_url, extract_zip
from torch_geometric.typing import OptTensor
# --- Local modules
from .XASNet_GAT import *
from .XASNet_GraphNet.modules import GraphNetwork
from .XASNet_SchNet import *
from .utils.weight_init import kaiming_orthogonal_init

gnn_layers = {
    'gat': geomnn.GATConv,
    'gcn': geomnn.GCNConv,
    'gatv2': geomnn.GATv2Conv,
    'graphConv': geomnn.GraphConv,
    'sage': geomnn.SAGEConv
    }

qm9_target_dict: Dict[int, str] = {
    0: 'dipole_moment',
    1: 'isotropic_polarizability',
    2: 'homo',
    3: 'lumo',
    4: 'gap',
    5: 'electronic_spatial_extent',
    6: 'zpve',
    7: 'energy_U0',
    8: 'energy_U',
    9: 'enthalpy_H',
    10: 'free_energy',
    11: 'heat_capacity',
}

class XASNet_GNN(torch.nn.Module):
    """
    General implementation of XASNet. The class provides multi-layer GNN 
    and supports different GNN types e.g. GCN, GATv2, GAT, GraphConv.

    Args:
        gnn_name: The type of GNN to train including gcn, gatv2, gat, graphconv.  
        num_layers: Number of GNN layers. 
        in_channels: List of input channels (same size of layers).
        out_channels: List of output channels (same size of layers).
        num_targets: Number of target data points in energy axis 
                of XAS spectrum. Defaults to 100.
        heads: Number of heads in gat and gatv2.
        gat_dp: The rate of dropout in case of gat and gatv2.
    """
    def __init__(
        self, 
        gnn_name: str, 
        num_layers: int,
        in_channels: List[int],
        out_channels: List[int],
        num_targets: int,
        dropout: float,
        heads: Optional[int] = None,
        gat_dp: float = 0,
        ) -> None:
        super().__init__()
        assert gnn_name in gnn_layers
        assert num_layers > 0
        assert len(in_channels) == num_layers and \
        len(out_channels) == num_layers

        self.gnn_name = gnn_name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_targets = num_targets
        self.dp_rate = dropout
        self.num_layers = num_layers
        self.heads = heads

        gnn_layer = gnn_layers[gnn_name]
        int_layers = []
      
        for i, in_c, out_c in zip(range(num_layers - 1), 
                                    in_channels[:-1], out_channels[:-1]):
                  
                if i == 0 and heads: 
                    int_layers.append(gnn_layer(in_c, out_c, heads=heads))
                elif i==0:
                    int_layers.append(gnn_layer(in_c, out_c))
                elif i > 0 and heads:
                    int_layers.append(gnn_layer(in_c*heads, out_c, heads=heads)) 
                elif i > 0:
                    int_layers.append(gnn_layer(in_c, out_c))
                int_layers.append(torch.nn.ReLU(inplace=True))
        
        if heads:
            int_layers.append(gnn_layer(in_channels[-1]*heads, 
                                        out_channels[-1], heads=1, 
                                        dropout=gat_dp))
        else:
            int_layers.append(gnn_layer(in_channels[-1], 
                                        out_channels[-1]))
        
        self.interaction_layers = torch.nn.ModuleList(int_layers)

        self.dropout = torch.nn.Dropout(p=self.dp_rate)
        self.out = torch.nn.Linear(out_channels[-1], num_targets)      
        self.reset_parameters()
        
    def reset_parameters(self):
        for m in self.interaction_layers:
            if isinstance(m, geomnn.GATConv):
                layers = [m.lin_src, m.lin_dst]
            elif isinstance(m, geomnn.GCNConv):
                layers = [m.lin]
            elif isinstance(m, geomnn.GATv2Conv):
                layers = [m.lin_r, m.lin_l]
            elif isinstance(m, geomnn.GraphConv):
                layers = [m.lin_rel, m.lin_root]
            elif isinstance(m.geomnn.SAGEConv):
                layers = [m.lin_l]
                
            for layer in layers:
                kaiming_orthogonal_init(layer.weight.data)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.0)
        
        kaiming_orthogonal_init(self.out.weight.data)
        self.out.bias.data.fill_(0.0)   

    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                batch_seg: torch.Tensor) -> torch.Tensor:   
    
        for layer in self.interaction_layers[:-1]:
            if isinstance(layer, geomnn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
                x = self.dropout(x)
                
        x = self.interaction_layers[-1](x, edge_index)        
        x = geomnn.global_mean_pool(x, batch_seg)
        x = self.dropout(x)
        out = self.out(x)
        return out

class XASNet_GAT(torch.nn.Module):
    """
    More detailed and custom implementation of GAT with different types of GAT layers.
    The model can get deeper using prelayers and residual layers. Moreover, jumping knowledge mechanism 
    as an additional layer is applied to focus on important parts of the node's environment.

    Args:
        node_features_dim: The dimension of node features.
        num_layers: Number of GNN layers. 
        in_channels: List of input channels (same size of layers).
        out_channels: List of output channels (same size of layers).
        n_heads: Number of heads.
        targets: Number of target data points in energy axis 
                of XAS spectrum. Defaults to 100.
        gat_type: Type of the gat layer.
        use_residuals: If true, residual layers is used.
        use_jk: If true, jumping knowledge mechanism is applied.
    """
    def __init__(
        self, 
        node_features_dim: int,
        n_layers: int,
        in_channels: List[int],
        out_channels: List[int],
        n_heads: int,
        targets: int,
        gat_type: str = 'gat_custom',
        use_residuals: bool = False,
        use_jk: bool = False
        ):
        super().__init__()

        self.node_features_dim = node_features_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.targets = targets
        self.gat_type = gat_type
        self.use_residuals = use_residuals
        self.use_jk = use_jk

        assert len(in_channels) == n_layers
        gat_types = {
            'gat_custom' : GATLayerCus,
            'gatv2_custom' : GATv2LayerCus,
            'gat' : geomnn.GATConv,
            'gatv2' : geomnn.GATv2Conv
        }
        assert gat_type in gat_types

        if use_residuals:
            self.pre_layer = LinearLayer(node_features_dim, in_channels[0], activation='relu')
            self.res_block = Residual_block(in_channels[0], 4, activation='relu')

        gat_layers = []

        for i, c_in, c_out in zip(range(n_layers-1), \
            in_channels[:-1], out_channels[:-1]):
            if i == 0:
                gat_layers.append(gat_types[gat_type](c_in, c_out, 
                heads=n_heads))
            elif i > 0:
                gat_layers.append(gat_types[gat_type](c_in*n_heads, c_out, 
                heads=n_heads))
            gat_layers.append(torch.nn.ReLU(inplace=True))
    
        gat_layers.append(gat_types[gat_type](in_channels[-1]*n_heads, out_channels[-1]))
        self.gat_layers = torch.nn.ModuleList(gat_layers)

        #jumping knowledge layers
        self.lstm = LSTM(out_channels[-2]*n_heads, out_channels[-2], 
        num_layers=3, bidirectional=True, batch_first=True)
        self.attn = Linear(2*out_channels[-2], 1)

        self.dropout = torch.nn.Dropout(p=0.3)
        self.linear_out = LinearLayer(out_channels[-1], targets)

    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                batch_seg: torch.Tensor) -> torch.Tensor:

        x = self.pre_layer(x)
        x = self.res_block(x)

        xs = []
        for layer in self.gat_layers[:-1]:
            if isinstance(layer, MessagePassing):
                x = layer(x, edge_index)
            else: 
                x = layer(x)
                x = self.dropout(x)
                xs.append(x.flatten(1).unsqueeze(-1))
        
        xs = torch.cat(xs, dim=-1).transpose(1, 2)
        alpha, _ = self.lstm(xs)
        alpha = self.attn(alpha).squeeze(-1)
        alpha = torch.softmax(alpha, dim=-1)
        x = (xs * alpha.unsqueeze(-1)).sum(1)

        x = self.gat_layers[-1](x, edge_index)
        x = global_mean_pool(x, batch_seg)
        x = self.dropout(x)
        out = self.linear_out(x)
        return out

class XASNet_GraphNet(torch.nn.Module):
    """
    GraphNet implementation of XASNet. The global, node and edge states 
    are used in the messsage function.
    """
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 hidden_channels: int,
                 out_channels: int,
                 gat_hidd: int,
                 gat_out: int,
                 n_layers: int = 3,
                 n_targets: int = 100):
        """
        Args:
            node_dim (int): Dimension of the nodes' attribute in the graph data.
            edge_dim (int): Dimension of the edges' attribute in the graph data.
            hidden_channels (int): Hidden channels in GraphNet layers.
            out_channels (int): Output channels in GraphNet layers.
            gat_hidd (int): Hidden channels for GAT layer used to obtain 
                the global state of each input graph.
            gat_out (int): Output channels for GAT layer used to obtain 
                the global state of each input graph.
            n_layers (int, optional): Number of layers in GraphNet. Defaults to 3.
            n_targets (int, optional): Number of target data points in energy axis 
                of XAS spectrum. Defaults to 100.
        """
        super().__init__()
        assert n_layers > 0

        #preparing the parameters for global, node and edge models 
        feat_in_node = node_dim + 2*edge_dim + gat_out
        feat_in_edge = 2*out_channels + edge_dim + gat_out
        feat_in_glob = 2*out_channels + gat_out
        node_model_params0 = {"feat_in": feat_in_node, 
                              "feat_hidd": hidden_channels, 
                              "feat_out": out_channels}  
        edge_model_params0 = {"feat_in": feat_in_edge, 
                              "feat_hidd": hidden_channels, 
                              "feat_out": out_channels} 
        global_model_params0 = {"feat_in": feat_in_glob, 
                                "feat_hidd": hidden_channels, 
                                "feat_out": out_channels}

        all_params = {"graphnet0": {"node_model_params": node_model_params0,
            "edge_model_params": edge_model_params0,
            "global_model_params": global_model_params0,
            "gat_in": node_dim,
            "gat_hidd": gat_hidd,
            "gat_out": gat_out}}
        
        for i in range(1, n_layers):
            all_params[f"graphnet{i}"] = {
                "node_model_params": {"feat_in": 4*out_channels, 
                                      "feat_hidd": hidden_channels, 
                                      "feat_out": out_channels},
                "edge_model_params": {"feat_in": 4*out_channels, 
                                      "feat_hidd": hidden_channels, 
                                      "feat_out": out_channels},
                "global_model_params": {"feat_in": 3*out_channels, 
                                      "feat_hidd": hidden_channels, 
                                      "feat_out": out_channels},
                "gat_in": node_dim,
                "gat_hidd": gat_hidd,
                "gat_out": gat_out
                }
            
        graphnets = []
        for v in all_params.values():
            graphnets.append(GraphNetwork(**v))

        self.graphnets = ModuleList(graphnets)

        self.dropout = Dropout(p=0.3)
        self.output_dense = Linear(out_channels, n_targets)
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_orthogonal_init(self.output_dense.weight.data)

    def forward(self, graph: Any) -> torch.Tensor:
        for graphnet in self.graphnets:
            graph = graphnet(graph)

        x = global_add_pool(graph.x, graph.batch)
        
        x = self.dropout(x)
        out = self.output_dense(x)
        return out
    
class XASNet_SchNet(torch.nn.Module):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
    the interactions blocks of the form.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

    .. note::

        For an example of using a pretrained SchNet variant, see
        `examples/qm9_pretrained_schnet.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        qm9_pretrained_schnet.py>`_.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        interaction_graph (callable, optional): The function used to compute
            the pairwise interaction graph and interatomic distances. If set to
            :obj:`None`, will construct a graph based on :obj:`cutoff` and
            :obj:`max_num_neighbors` properties.
            If provided, this method takes in :obj:`pos` and :obj:`batch`
            tensors and should return :obj:`(edge_index, edge_weight)` tensors.
            (default :obj:`None`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        readout (str, optional): Whether to apply :obj:`"add"` or :obj:`"mean"`
            global aggregation. (default: :obj:`"add"`)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction, *e.g.*, for
            target 0 of :class:`torch_geometric.datasets.QM9`.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
    """

    url = 'http://www.quantum-machine.org/datasets/trained_schnet_models.zip'

    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        interaction_graph: Optional[Callable] = None,
        max_num_neighbors: int = 32,
        readout: str = 'add',
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: OptTensor = None,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.dipole = dipole
        self.sum_aggr = SumAggregation()
        self.readout = aggr_resolver('sum' if self.dipole else readout)
        self.mean = mean
        self.std = std
        self.scale = None

        if self.dipole:
            import ase

            atomic_mass = torch.from_numpy(ase.data.atomic_masses)
            self.register_buffer('atomic_mass', atomic_mass)

        # Support z == 0 for padding atoms so that their embedding vectors
        # are zeroed and do not receive any gradients.
        self.embedding = Embedding(100, hidden_channels, padding_idx=0)

        if interaction_graph is not None:
            self.interaction_graph = interaction_graph
        else:
            self.interaction_graph = RadiusInteractionGraph(
                cutoff, max_num_neighbors)

        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels, 200)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    @staticmethod
    def from_qm9_pretrained(
        root: str,
        dataset: Dataset,
        target: int,
    ) -> Tuple['SchNet', Dataset, Dataset, Dataset]:  # pragma: no cover
        r"""Returns a pre-trained :class:`SchNet` model on the
        :class:`~torch_geometric.datasets.QM9` dataset, trained on the
        specified target :obj:`target`.
        """
        import ase
        import schnetpack as spk  # noqa

        assert target >= 0 and target <= 12
        is_dipole = target == 0

        units = [1] * 12
        units[0] = ase.units.Debye
        units[1] = ase.units.Bohr**3
        units[5] = ase.units.Bohr**2

        root = osp.expanduser(osp.normpath(root))
        os.makedirs(root, exist_ok=True)
        folder = 'trained_schnet_models'
        if not osp.exists(osp.join(root, folder)):
            path = download_url(SchNet.url, root)
            extract_zip(path, root)
            os.unlink(path)

        name = f'qm9_{qm9_target_dict[target]}'
        path = osp.join(root, 'trained_schnet_models', name, 'split.npz')

        split = np.load(path)
        train_idx = split['train_idx']
        val_idx = split['val_idx']
        test_idx = split['test_idx']

        # Filter the splits to only contain characterized molecules.
        idx = dataset.data.idx
        assoc = idx.new_empty(idx.max().item() + 1)
        assoc[idx] = torch.arange(idx.size(0))

        train_idx = assoc[train_idx[np.isin(train_idx, idx)]]
        val_idx = assoc[val_idx[np.isin(val_idx, idx)]]
        test_idx = assoc[test_idx[np.isin(test_idx, idx)]]

        path = osp.join(root, 'trained_schnet_models', name, 'best_model')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            state = fs.torch_load(path, map_location='cpu')

        net = SchNet(
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0,
            dipole=is_dipole,
            atomref=dataset.atomref(target),
        )

        net.embedding.weight = state.representation.embedding.weight

        for int1, int2 in zip(state.representation.interactions,
                              net.interactions):
            int2.mlp[0].weight = int1.filter_network[0].weight
            int2.mlp[0].bias = int1.filter_network[0].bias
            int2.mlp[2].weight = int1.filter_network[1].weight
            int2.mlp[2].bias = int1.filter_network[1].bias
            int2.lin.weight = int1.dense.weight
            int2.lin.bias = int1.dense.bias

            int2.conv.lin1.weight = int1.cfconv.in2f.weight
            int2.conv.lin2.weight = int1.cfconv.f2out.weight
            int2.conv.lin2.bias = int1.cfconv.f2out.bias

        net.lin1.weight = state.output_modules[0].out_net[1].out_net[0].weight
        net.lin1.bias = state.output_modules[0].out_net[1].out_net[0].bias
        net.lin2.weight = state.output_modules[0].out_net[1].out_net[1].weight
        net.lin2.bias = state.output_modules[0].out_net[1].out_net[1].bias

        mean = state.output_modules[0].atom_pool.average
        net.readout = aggr_resolver('mean' if mean is True else 'add')

        dipole = state.output_modules[0].__class__.__name__ == 'DipoleMoment'
        net.dipole = dipole

        net.mean = state.output_modules[0].standardize.mean.item()
        net.std = state.output_modules[0].standardize.stddev.item()

        if state.output_modules[0].atomref is not None:
            net.atomref.weight = state.output_modules[0].atomref.weight
        else:
            net.atomref = None

        net.scale = 1.0 / units[target]

        return net, (dataset[train_idx], dataset[val_idx], dataset[test_idx])

    def forward(self, z: Tensor, pos: Tensor,
                batch: OptTensor = None) -> Tensor:
        r"""Forward pass.

        Args:
            z (torch.Tensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (torch.Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (torch.Tensor, optional): Batch indices assigning each atom
                to a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
        """
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            M = self.sum_aggr(mass, batch, dim=0)
            c = self.sum_aggr(mass * pos, batch, dim=0) / M
            h = h * (pos - c.index_select(0, batch))

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        out = self.readout(h, batch, dim=0)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')