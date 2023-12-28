import torch
from copy import deepcopy
import torch.nn.functional as F

from .dp import *

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

def fed_avg_prompt_last_epoch(client_list):
    client_weight = [deepcopy(client.prompt_last_epoch.state_dict()) for client in client_list]
    weight_sum = {k: torch.zeros_like(v) for k, v in client_weight[0].items()}
    for weights in client_weight:
        for key, value in weights.items():
            weight_sum[key] += value
    aggregated_weight = {k: v / len(client_weight) for k, v in weight_sum.items()}
    for client in client_list:
        client.prompt_last_epoch.load_state_dict(aggregated_weight, strict=True)
    return aggregated_weight


def fed_avg_answer_last_epoch(client_list):
    client_weight = [deepcopy(client.answer_last_epoch.state_dict()) for client in client_list]
    weight_sum = {k: torch.zeros_like(v) for k, v in client_weight[0].items()}
    for weights in client_weight:
        for key, value in weights.items():
            weight_sum[key] += value
    aggregated_weight = {k: v / len(client_weight) for k, v in weight_sum.items()}
    for client in client_list:
        client.answer_last_epoch.load_state_dict(aggregated_weight, strict=True)
    return aggregated_weight


def normalize_matrix(coefficient):
    tensor_coefficient = torch.tensor([[coefficient[key1][key2] for key2 in coefficient[key1]] for key1 in coefficient])

    col_sum = torch.sum(tensor_coefficient, dim=0)
    normalized_coefficient = tensor_coefficient / col_sum

    normalized_coefficient = {
        'node': {'node': normalized_coefficient[0, 0].item(), 'edge': normalized_coefficient[1, 0].item(), 'graph': normalized_coefficient[2, 0].item()},
        'edge': {'node': normalized_coefficient[0, 1].item(), 'edge': normalized_coefficient[1, 1].item(), 'graph': normalized_coefficient[2, 1].item()},
        'graph': {'node': normalized_coefficient[0, 2].item(), 'edge': normalized_coefficient[1, 2].item(), 'graph': normalized_coefficient[2, 2].item()}
    }
    # print(coefficient)
    return normalized_coefficient
        
def compute_coefficient(client_list_by_task, lr_prompt, lr_answer):
    task_prompt_weights = {}
    task_answer_weights = {}
    vector_prompt_weights = {}
    vector_answer_weights = {}
    for task in ['node', 'edge', 'graph']:
        task_prompt_weights[task] = fed_avg_prompt(client_list_by_task[task])
        task_answer_weights[task] = fed_avg_answer(client_list_by_task[task])
        vector_prompt_weights[task] = torch.cat([p.view(-1) for p in task_prompt_weights[task].values()])
        vector_answer_weights[task] = torch.cat([p.view(-1) for p in task_answer_weights[task].values()])

    task_prompt_weights_last_epoch = {}
    task_answer_weights_last_epoch = {}
    vector_prompt_weights_last_epoch = {}
    vector_answer_weights_last_epoch = {}
    for task in ['node', 'edge', 'graph']:
        task_prompt_weights_last_epoch[task] = fed_avg_prompt_last_epoch(client_list_by_task[task])
        task_answer_weights_last_epoch[task] = fed_avg_answer_last_epoch(client_list_by_task[task])
        vector_prompt_weights_last_epoch[task] = torch.cat([p.view(-1) for p in task_prompt_weights_last_epoch[task].values()])
        vector_answer_weights_last_epoch[task] = torch.cat([p.view(-1) for p in task_answer_weights_last_epoch[task].values()])

    prompt_coefficient = {
    'node': {'node': None, 'edge': None, 'graph': None},
    'edge': {'node': None, 'edge': None, 'graph': None},
    'graph': {'node': None, 'edge': None, 'graph': None}
    }
    answer_coefficient = {
    'node': {'node': None, 'edge': None, 'graph': None},
    'edge': {'node': None, 'edge': None, 'graph': None},
    'graph': {'node': None, 'edge': None, 'graph': None}
    }

    for source in ['node', 'edge', 'graph']:
        for target in ['node', 'edge', 'graph']:
            if source == target:
                prompt_coefficient[source][target] = torch.norm(vector_prompt_weights[target])
            else:
                temp1 = vector_prompt_weights[target] - vector_prompt_weights_last_epoch[target]
                temp2 = vector_prompt_weights[source] - vector_prompt_weights_last_epoch[source]
                prompt_coefficient[source][target] = torch.dot(temp1, temp2) / torch.norm(temp2)

    for source in ['node', 'edge', 'graph']:
        for target in ['node', 'edge', 'graph']:
            if source == target:
                answer_coefficient[source][target] = torch.norm(vector_answer_weights[target])
            else:
                temp1 = vector_answer_weights[target] - vector_answer_weights_last_epoch[target]
                temp2 = vector_answer_weights[source] - vector_answer_weights_last_epoch[source]
                answer_coefficient[source][target] = torch.dot(temp1, temp2) / torch.norm(temp2)

    prompt_coefficient = normalize_matrix(prompt_coefficient)
    answer_coefficient = normalize_matrix(answer_coefficient)

    return prompt_coefficient, answer_coefficient

def weighted_task_fed_avg(client_list_by_task, prompt_coefficient, answer_coefficient, epsilon):
    
    task_prompt_weights = {}
    task_answer_weights = {}

    for task in ['node', 'edge', 'graph']:
        task_prompt_weights[task] = fed_avg_prompt(client_list_by_task[task])
        task_answer_weights[task] = fed_avg_answer(client_list_by_task[task])

    aggregated_prompt_weight_dict = {
    'node': None,
    'edge': None,
    'graph': None
    }
    aggregated_answer_weight_dict = {
    'node': None,
    'edge': None,
    'graph': None
    }

    for task in ['node', 'edge', 'graph']:
        aggregated_prompt_weight = {k: (task_prompt_weights['node'][k] * prompt_coefficient['node'][task] +
                                        task_prompt_weights['edge'][k] * prompt_coefficient['edge'][task] +
                                        task_prompt_weights['graph'][k] * prompt_coefficient['graph'][task])
                                    for k in task_prompt_weights['graph'].keys()}
        aggregated_prompt_weight_dict[task] = aggregated_prompt_weight

        aggregated_answer_weight = {k: (task_answer_weights['node'][k] * answer_coefficient['node'][task] +
                                        task_answer_weights['edge'][k] * answer_coefficient['edge'][task] +
                                        task_answer_weights['graph'][k] * answer_coefficient['graph'][task])
                                    for k in task_answer_weights['graph'].keys()}
        aggregated_answer_weight_dict[task] = aggregated_answer_weight


    for task in ['node', 'edge', 'graph']:  
        for client in client_list_by_task[task]: 
            prompt_weights = deepcopy(client.prompt.state_dict())
            prompt = {}
            for key in prompt_weights:
                prompt[key] = 0.5 * deepcopy(prompt_weights[key]) + 0.5 * aggregated_prompt_weight_dict[task][key]
            answer_weights = deepcopy(client.answer.state_dict())
            answer = {}
            for key in answer_weights:
                answer[key] = 0.5 * deepcopy(answer_weights[key]) + 0.5 * aggregated_answer_weight_dict[task][key]
            
            # add dp (Federated Phase)
            prompt = (add_noise_for_state_dict(prompt, epsilon))
            answer = (add_noise_for_state_dict(answer, epsilon))
            client.prompt.load_state_dict(prompt, strict=True)
            client.answer.load_state_dict(answer, strict=True)

