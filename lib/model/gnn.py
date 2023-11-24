import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, global_mean_pool


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, pool=global_mean_pool, gnn_type='GAT'):
        super().__init__()
        GraphConv = eval(gnn_type)
        self.gnn_type = gnn_type
        if gcn_layer_num == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim), GraphConv(hid_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, hid_dim)]
            for i in range(gcn_layer_num - 2):
                layers.append(GraphConv(hid_dim, hid_dim))
            layers.append(GraphConv(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)
        self.pool = pool

    def forward(self, x, edge_index, batch):
        for conv in self.conv_layers[0:-1]:
            x = conv(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, training=self.training)
        node_emb = self.conv_layers[-1](x, edge_index)
        graph_emb = self.pool(node_emb, batch.long())
        return graph_emb

    @staticmethod
    def act(x=None, act_type='leakyrelu'):
        if act_type == 'leakyrelu':
            return F.leaky_relu(x) if x is not None else torch.nn.LeakyReLU()
        elif act_type == 'tanh':
            return torch.tanh(x) if x is not None else torch.nn.Tanh()
