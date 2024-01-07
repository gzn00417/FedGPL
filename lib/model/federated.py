import torch
from copy import deepcopy
import torch.nn.functional as F

from .dp import *

def avg_c(dprompt):
    stacked_dprompt = torch.stack(dprompt)
    average_dprompt = torch.mean(stacked_dprompt, dim=0)
    return average_dprompt

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
    tensor_coefficient[tensor_coefficient < 0] = 0
    col_sum = torch.sum(tensor_coefficient, dim=1)
    # print(tensor_coefficient)
    for index1, list in enumerate(tensor_coefficient):
        for index, value in enumerate(list):
            tensor_coefficient[index1][index] = value / col_sum[index1]
    # print(tensor_coefficient)
    # print(col_sum)
    normalized_coefficient = {
        'node': {'node': tensor_coefficient[0, 0].item(), 'edge': tensor_coefficient[0, 1].item(), 'graph': tensor_coefficient[0, 2].item()},
        'edge': {'node': tensor_coefficient[1, 0].item(), 'edge': tensor_coefficient[1, 1].item(), 'graph': tensor_coefficient[1, 2].item()},
        'graph': {'node': tensor_coefficient[2, 0].item(), 'edge': tensor_coefficient[2, 1].item(), 'graph': tensor_coefficient[2, 2].item()}
    }
    # print(normalized_coefficient)
    return normalized_coefficient
        
def compute_coefficient(client_list_by_task, lr_prompt, lr_answer):
    task_prompt_weights = {}
    task_answer_weights = {}
    vector_prompt_weights = {}
    vector_answer_weights = {}
    weights_vector = {}
    for task in ['node', 'edge', 'graph']:
        task_prompt_weights[task] = fed_avg_prompt(client_list_by_task[task])
        task_answer_weights[task] = fed_avg_answer(client_list_by_task[task])
        vector_prompt_weights[task] = torch.cat([p.view(-1) for p in task_prompt_weights[task].values()])
        vector_answer_weights[task] = torch.cat([p.view(-1) for p in task_answer_weights[task].values()])

    for task in ['node', 'edge', 'graph']:
        weights_vector[task] = torch.cat((vector_prompt_weights[task], vector_answer_weights[task]), dim=0)

    task_prompt_weights_last_epoch = {}
    task_answer_weights_last_epoch = {}
    vector_prompt_weights_last_epoch = {}
    vector_answer_weights_last_epoch = {}
    weights_vector_last_epoch = {}
    for task in ['node', 'edge', 'graph']:
        task_prompt_weights_last_epoch[task] = fed_avg_prompt_last_epoch(client_list_by_task[task])
        task_answer_weights_last_epoch[task] = fed_avg_answer_last_epoch(client_list_by_task[task])
        vector_prompt_weights_last_epoch[task] = torch.cat([p.view(-1) for p in task_prompt_weights_last_epoch[task].values()])
        vector_answer_weights_last_epoch[task] = torch.cat([p.view(-1) for p in task_answer_weights_last_epoch[task].values()])
    for task in ['node', 'edge', 'graph']:
        weights_vector_last_epoch[task] = torch.cat((vector_prompt_weights_last_epoch[task], vector_answer_weights_last_epoch[task]), dim=0)

    # prompt_coefficient = {
    # 'node': {'node': None, 'edge': None, 'graph': None},
    # 'edge': {'node': None, 'edge': None, 'graph': None},
    # 'graph': {'node': None, 'edge': None, 'graph': None}
    # }
    # answer_coefficient = {
    # 'node': {'node': None, 'edge': None, 'graph': None},
    # 'edge': {'node': None, 'edge': None, 'graph': None},
    # 'graph': {'node': None, 'edge': None, 'graph': None}
    # }
        
    coefficient = {
    'node': {'node': None, 'edge': None, 'graph': None},
    'edge': {'node': None, 'edge': None, 'graph': None},
    'graph': {'node': None, 'edge': None, 'graph': None}
    }

    # for source in ['node', 'edge', 'graph']:
    #     for target in ['node', 'edge', 'graph']:
    #         if source == target:
    #             prompt_coefficient[source][target] = torch.norm(vector_prompt_weights[target])
    #         else:
    #             temp1 = vector_prompt_weights[target] - vector_prompt_weights_last_epoch[target]
    #             temp2 = vector_prompt_weights[source] - vector_prompt_weights_last_epoch[target]
    #             prompt_coefficient[source][target] = torch.dot(temp1, temp2) / torch.norm(temp1)

    # for source in ['node', 'edge', 'graph']:
    #     for target in ['node', 'edge', 'graph']:
    #         if source == target:
    #             answer_coefficient[source][target] = torch.norm(vector_answer_weights[target])
    #         else:
    #             temp1 = vector_answer_weights[target] - vector_answer_weights_last_epoch[target]
    #             temp2 = vector_answer_weights[source] - vector_answer_weights_last_epoch[target]
    #             answer_coefficient[source][target] = torch.dot(temp1, temp2) / torch.norm(temp1)
    for target in ['node', 'edge', 'graph']:
        for source in ['node', 'edge', 'graph']:
            if source == target:
                coefficient[target][source] = torch.norm(weights_vector[target])
            else:
                temp1 = weights_vector[target] - weights_vector_last_epoch[target]
                temp2 = weights_vector[source] - weights_vector_last_epoch[target]
                coefficient[target][source] = torch.dot(temp1, temp2) / torch.norm(temp1)

    # prompt_coefficient = normalize_matrix(prompt_coefficient)
    # answer_coefficient = normalize_matrix(answer_coefficient)
    # coefficient['node']['node'] = torch.tensor(0)
    # coefficient['node']['edge'] = torch.tensor(0)
    # coefficient['node']['graph'] = torch.tensor(1)

    # coefficient['edge']['node'] = torch.tensor(0)
    # coefficient['edge']['edge'] = torch.tensor(0)
    # coefficient['edge']['graph'] = torch.tensor(1)

    # coefficient['graph']['node'] = torch.tensor(0)
    # coefficient['graph']['edge'] = torch.tensor(0)
    # coefficient['graph']['graph'] = torch.tensor(1)

    coefficient = normalize_matrix(coefficient)
    
    print(coefficient)

    return coefficient, coefficient

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
        # if task == 'node':
        #     aggregated_prompt_weight = {k: (task_prompt_weights['node'][k] * 0.2 +
        #                                     task_prompt_weights['edge'][k] * 0 +
        #                                     task_prompt_weights['graph'][k] * 0.8)
        #                                 for k in task_prompt_weights['graph'].keys()}
        #     aggregated_answer_weight = {k: (task_answer_weights['node'][k] * 0.2 +
        #                                     task_answer_weights['edge'][k] * 0 +
        #                                     task_answer_weights['graph'][k] * 0.8)
        #                                 for k in task_answer_weights['graph'].keys()}
        # elif task == 'edge':
        #     aggregated_prompt_weight = {k: (task_prompt_weights['node'][k] * 0 +
        #                                     task_prompt_weights['edge'][k] * 0.5 +
        #                                     task_prompt_weights['graph'][k] * 0.5)
        #                                 for k in task_prompt_weights['graph'].keys()}
        #     aggregated_answer_weight = {k: (task_answer_weights['node'][k] * 0 +
        #                                     task_answer_weights['edge'][k] * 0.5 +
        #                                     task_answer_weights['graph'][k] * 0.5)
        #                                 for k in task_answer_weights['graph'].keys()}
        # elif task == 'graph':
        #     aggregated_prompt_weight = {k: (task_prompt_weights['node'][k] * 0 +
        #                                     task_prompt_weights['edge'][k] * 0 +
        #                                     task_prompt_weights['graph'][k] * 1)
        #                                 for k in task_prompt_weights['graph'].keys()}
        #     aggregated_answer_weight = {k: (task_answer_weights['node'][k] * 0 +
        #                                     task_answer_weights['edge'][k] * 0 +
        #                                     task_answer_weights['graph'][k] * 1)
        #                                 for k in task_answer_weights['graph'].keys()}
        aggregated_prompt_weight = {k: (task_prompt_weights['node'][k] * prompt_coefficient[task]['node'] +
                                        task_prompt_weights['edge'][k] * prompt_coefficient[task]['edge'] +
                                        task_prompt_weights['graph'][k] * prompt_coefficient[task]['graph'])
                                    for k in task_prompt_weights['graph'].keys()}
        aggregated_answer_weight = {k: (task_answer_weights['node'][k] * answer_coefficient[task]['node'] +
                                        task_answer_weights['edge'][k] * answer_coefficient[task]['edge'] +
                                        task_answer_weights['graph'][k] * answer_coefficient[task]['graph'])
                                    for k in task_answer_weights['graph'].keys()}
        aggregated_prompt_weight_dict[task] = aggregated_prompt_weight
        aggregated_answer_weight_dict[task] = aggregated_answer_weight


    for task in ['node', 'edge', 'graph']:  
        for client in client_list_by_task[task]: 
            prompt_weights = deepcopy(client.prompt.state_dict())
            prompt = {}
            for key in prompt_weights:
                prompt[key] = aggregated_prompt_weight_dict[task][key]
            answer_weights = deepcopy(client.answer.state_dict())
            answer = {}
            for key in answer_weights:
                answer[key] = aggregated_answer_weight_dict[task][key]
            
            # add dp (Federated Phase)
            prompt = (add_noise_for_state_dict(prompt, epsilon))
            answer = (add_noise_for_state_dict(answer, epsilon))
            client.prompt.load_state_dict(prompt, strict=True)
            client.answer.load_state_dict(answer, strict=True)

