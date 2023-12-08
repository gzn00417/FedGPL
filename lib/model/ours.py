import torch
from torch_geometric.data import Batch, Data
from torch_geometric import nn as pyg_nn


class Ours(torch.nn.Module):
    def __init__(self, token_dim, token_num, group_num=1, cross_prune=0.1, inner_prune=0.3):
        super().__init__()
        self.cross_prune = cross_prune
        self.inner_prune = inner_prune
        self.token_list = torch.nn.ParameterList([torch.nn.Parameter(torch.empty(token_num, token_dim)) for i in range(group_num)])
        for token in self.token_list:
            torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)

        self.pool = pyg_nn.TopKPooling(in_channels=token_dim, ratio=0.5)

    def forward(self, graph_batch: Batch):
        pg_list = []
        for i, tokens in enumerate(self.token_list):
            # inner link: token-->token
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1

            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            edge_index = inner_adj.nonzero().t().contiguous()

            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long()))

        pg = Batch.from_data_list(pg_list)

        inner_edge_index = pg.edge_index
        token_num = pg.x.shape[0]

        re_graph_list = []
        for g in Batch.to_data_list(graph_batch):
            g_edge_index = g.edge_index + token_num

            cross_dot = torch.mm(pg.x, torch.transpose(g.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)

            cross_edge_index = cross_adj.nonzero().t().contiguous()
            cross_edge_index[1] = cross_edge_index[1] + token_num

            x = torch.cat([pg.x, g.x], dim=0)
            y = g.y

            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            data = Data(x=x, edge_index=edge_index, y=y)
            re_graph_list.append(data)

        prompted_graph = Batch.from_data_list(re_graph_list)
        x, edge_index, batch = prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch

        x, edge_index, _, batch, _, _ = self.pool(x=x, edge_index=edge_index, batch=batch)  # TopKPooling
        return x, edge_index, batch
