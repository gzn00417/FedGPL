import torch
import torch.distributions as dist
from torch.distributions.laplace import Laplace
import numpy as np

def add_laplace_noise(data, epsilon):
    sensitivity = torch.max(data) - torch.min(data)
    sensitivity = sensitivity.item()

    scale = sensitivity / epsilon

    noise = np.random.laplace(loc=0, scale=scale, size=data.size())
    noise = torch.from_numpy(noise).to('cuda')

    noise = noise.to(torch.float)

    # return data + noise
    return data

def add_noise_for_state_dict(state_dict, epsilon):
    # print(state_dict)
    for k, v in state_dict.items():
        state_dict[k] = add_laplace_noise(state_dict[k], epsilon)

    return state_dict