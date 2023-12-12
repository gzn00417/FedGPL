import torch
from copy import deepcopy

def fed_avg_prompt(client_list):
    client_weight = [deepcopy(client.prompt.state_dict()) for client in client_list]
    weight_sum = {k: torch.zeros_like(v) for k, v in client_weight[0].items()}
    for weights in client_weight:
        for key, value in weights.items():
            weight_sum[key] += value
    aggregated_weight = {k: v / len(client_weight) for k, v in weight_sum.items()}
    for client in client_list:
        client.prompt.load_state_dict(aggregated_weight, strict=True)
    return aggregated_weight


def fed_avg_answer(client_list):
    client_weight = [deepcopy(client.answer.state_dict()) for client in client_list]
    weight_sum = {k: torch.zeros_like(v) for k, v in client_weight[0].items()}
    for weights in client_weight:
        for key, value in weights.items():
            weight_sum[key] += value
    aggregated_weight = {k: v / len(client_weight) for k, v in weight_sum.items()}
    for client in client_list:
        client.answer.load_state_dict(aggregated_weight, strict=True)
    return aggregated_weight


def weighted_task_fed_avg(client_list, client_list_by_task, node_task_weight, edge_task_weight, graph_task_weight):
    task_prompt_weights = {}
    task_answer_weights = {}
    for task in ['node', 'edge', 'graph']:
        task_prompt_weights[task] = fed_avg_prompt(client_list_by_task[task])
        task_answer_weights[task] = fed_avg_answer(client_list_by_task[task])
    aggregated_prompt_weight = {k: (task_prompt_weights['node'][k] * node_task_weight + task_prompt_weights['edge'][k] * edge_task_weight + task_prompt_weights['graph'][k] * graph_task_weight) for k, v in task_prompt_weights['graph'].items()}
    aggregated_answer_weight = {k: (task_answer_weights['node'][k] * node_task_weight + task_answer_weights['edge'][k] * edge_task_weight + task_answer_weights['graph'][k] * graph_task_weight) for k, v in task_answer_weights['graph'].items()}
    for client in client_list:
        client.prompt.load_state_dict(aggregated_prompt_weight, strict=True)
        client.answer.load_state_dict(aggregated_answer_weight, strict=True)