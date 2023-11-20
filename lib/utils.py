import os
import torch
import torch.nn.functional as F

def proto_aggregation(local_embeddings):
    embedding = 0*local_embeddings[0].data
    for i in local_embeddings:
        embedding += i.data
    global_embedding = embedding / len(local_embeddings)
    return global_embedding

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("create folder {}".format(path))
    else:
        print("folder exists! {}".format(path))

# used in prompt.py
def act(x=None, act_type='leakyrelu'):
    if act_type == 'leakyrelu':
        if x is None:
            return torch.nn.LeakyReLU()
        else:
            return F.leaky_relu(x)
    elif act_type == 'tanh':
        if x is None:
            return torch.nn.Tanh()
        else:
            return torch.tanh(x)
        
def PG_aggregation(local_PG_weights):
    weight_sum = {k: torch.zeros_like(v) for k, v in local_PG_weights[0].items()}

    for weights in local_PG_weights:
        for key, value in weights.items():
            weight_sum[key] += value

    num_models = len(local_PG_weights)
    aggregated_PG = {k: v / num_models for k, v in weight_sum.items()}

    return aggregated_PG