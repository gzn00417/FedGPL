import torch
from torch_geometric.data import Batch, Data
from torch_geometric import nn as pyg_nn
from torch import Tensor
from torch.nn import Parameter
from typing import Callable, Optional, Tuple, Union

from torch_geometric.nn.inits import uniform
from torch_geometric.utils import scatter, softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes

def select_top_k(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    batch: Tensor,
    min_score: Optional[float] = None,
    tol: float = 1e-7,
) -> Tensor:
    if min_score is not None:
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)

    elif ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')
        batch_size, max_num_nodes = num_nodes.size(0), int(num_nodes.max())

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ), -60000.0)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0), ), int(ratio))
            k = torch.min(k, num_nodes)
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        if isinstance(ratio, int) and (k == ratio).all():
            # If all graphs have exactly `ratio` or more than `ratio` entries,
            # we can just pick the first entries in `perm` batch-wise:
            index = torch.arange(batch_size, device=x.device) * max_num_nodes
            index = index.view(-1, 1).repeat(1, ratio).view(-1)
            index += torch.arange(ratio, device=x.device).repeat(batch_size)
        else:
            # Otherwise, compute indices per graph:
            index = torch.cat([
                torch.arange(k[i], device=x.device) + i * max_num_nodes
                for i in range(batch_size)
            ], dim=0)

        perm = perm[index]

    else:
        raise ValueError("At least one of 'min_score' and 'ratio' parameters "
                         "must be specified")

    return perm

def filter_adj(
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    perm: Tensor,
    scores: Tensor,  # 新增的评分矩阵
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index[0], edge_index[1]
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]
    scores = scores[mask] 

    num_edges = scores.size(0)
    topk = int(0.9 * num_edges)
    topk_indices = scores.topk(topk, largest=True).indices

    row = row[topk_indices]
    col = col[topk_indices]
    if edge_attr is not None:
        edge_attr = edge_attr[topk_indices]

    return torch.stack([row, col], dim=0), edge_attr

def maybe_num_nodes(edge_index: Tensor, num_nodes: Optional[int] = None) -> int:
    if num_nodes is not None:
        return num_nodes
    return int(edge_index.max()) + 1

class Ours(torch.nn.Module):
    def __init__(self, token_dim, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.in_channels = token_dim
        self.multiplier = 1
        self.nonlinearity = getattr(torch, 'tanh')
        self.min_score = None
        
        self.weight = Parameter(torch.Tensor(1, self.in_channels))
        self.reset_paramters()
    
    def reset_paramters(self):
        uniform(self.in_channels, self.weight)

    def forward(self, graph_batch: Batch):
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch

        x, edge_index, _, batch, _, _ = self.prompting(x=x, edge_index=edge_index, batch=batch)
        return x, edge_index, batch
    
    def prompting(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
            
        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = (attn * self.weight).sum(dim=-1)
        
        if self.min_score is None:
            score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)
            
        perm = select_top_k(x=score, ratio=self.ratio, batch=batch, min_score=self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_scores = torch.abs(score[edge_index[0]] - score[edge_index[1]])
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, edge_scores,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm]
