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
        for i in range(0, len(meta_paths)):
            self.sum_layers.append(HAN(meta_paths[i], in_size, hidden_size, out_size,
                                       num_heads, dropout, GAT_Layers, W_size))

    def forward(self, s_g, args):
        torch.manual_seed(args.seed)
        herb_feature = torch.randn((self.herb_fd, self.in_size)).to(args.device)
        target_feature = torch.randn((self.target_fd, self.in_size)).to(args.device)

        h1 = self.sum_layers[0](s_g[0], herb_feature)
        h2 = self.sum_layers[1](s_g[1], target_feature)

        return h1, h2, torch.matmul(h1, h2.t())


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
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)


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
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                    g, meta_path)
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            w_h = self.w_h(h)
            if self.nums_GAT == 1:
                semantic_embeddings.append(self.gat_layers[i](new_g, w_h).flatten(1))

        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)

        return (beta * z).sum(1)
