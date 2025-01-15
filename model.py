import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv


class IMCHAN(nn.Module):
    def __init__(self, meta_paths,
                 in_size,
                 herb_fd,
                 target_fd,
                 hidden_size,
                 out_size,
                 num_heads,
                 dropout,
                 GAT_Layers,
                 W_size):
        super(IMCHAN, self).__init__()

        self.herb_fd = herb_fd
        self.target_fd = target_fd
        self.in_size = in_size
        self.sum_layers = nn.ModuleList()

    def forward(self, s_g, args):
        


class HAN(nn.Module):
    """
    Wang, Xiao, et al. "Heterogeneous graph attention network."
    The World Wide Web Conference. 2019.
    """
    def __init__(self, meta_paths,
                 in_size,
                 hidden_size,
                 out_size,
                 num_heads,
                 dropout,
                 GAT_Layers,
                 W_size):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size,
                                    num_heads[0], dropout, GAT_Layers, W_size))

        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size, bias=False)

    def forward(self, g, h):
        

class HANLayer(nn.Module):
    def __init__(self, meta_paths,
                 in_size,
                 out_size,
                 layer_num_heads,
                 dropout,
                 GAT_Layers,
                 W_size):

        super(HANLayer, self).__init__()
        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        self.w_h = nn.Linear(in_size, W_size, bias=False)
        self.nums_GAT = GAT_Layers

        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(W_size, out_size, layer_num_heads, dropout, 
                                   activation=F.elu, allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
       
