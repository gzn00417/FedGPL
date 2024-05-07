import pickle as pk
from random import shuffle
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt


def collect_data_iid(num_users, dataset_name, num_class, k, task_type: str = 'node', dataset_type: str = 'train', few_shot = False):
    tmp = {'node': 0, 'edge': 1, 'graph': 2}.get(task_type) * num_class
    data = []
    for task_id in range(num_class):
        data_path1 = f'./data/{dataset_name}/induced_graphs/task{task_id + tmp}.meta.{dataset_type}.support'
        data_path2 = f'./data/{dataset_name}/induced_graphs/task{task_id + tmp}.meta.{dataset_type}.query'
        with (open(data_path1, 'br') as f1, open(data_path2, 'br') as f2):
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            if few_shot:
                shuffle(data_list)
                few_data = data_list[:k]
            if few_shot:
                for g in few_data:
                    g.y = task_id
                    data.append(g)
            else:
                for g in data_list:
                    g.y = task_id
                    data.append(g)
            
    shuffle(data)
    client_samples = []
    
    batch_size = len(data) // num_users
    batches = [data[i*batch_size:(i+1)*batch_size] for i in range(num_users)]
    for batch_data in batches:
        client_samples.append(Batch.from_data_list(batch_data))

    return client_samples


def collect_data_noniid(num_users, dataset_name, num_class, task_type: str = 'node', dataset_type: str = 'train', dir_alpha = 0.99):
    datas = collect_data_iid(num_users, dataset_name, num_class, 100, task_type, dataset_type)
    targets = []
    for i in range(num_users):
        targets.append(datas[i].y.tolist())

    client_dict = hetero_dir_partition(targets, num_users, num_class, dir_alpha, min_require_size=None)

    client_samples = []

    for client_id, indices in client_dict.items():
        client_samples.append(Batch.from_data_list(datas[0][indices]))

    return client_samples


def get_dataset(dataset_name, num_class, num_users, type, k, alpha=0.99, few_shot=False):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    dataset = {
        'node': {'train': [], 'test': []},
        'edge': {'train': [], 'test': []},
        'graph': {'train': [], 'test': []},
    }
    if type == 'iid':
        # node
        dataset['node']['train'] = collect_data_iid(int(num_users/3), dataset_name, num_class, k, 'node', 'train', few_shot)
        dataset['node']['test'] = collect_data_iid(int(num_users/3), dataset_name, num_class, k, 'node', 'test', few_shot)
        # edge
        dataset['edge']['train'] = collect_data_iid(int(num_users/3), dataset_name, num_class, k, 'edge', 'train', few_shot)
        dataset['edge']['test'] = collect_data_iid(int(num_users/3), dataset_name, num_class, k, 'edge', 'test', few_shot)
        # graph
        dataset['graph']['train'] = collect_data_iid(int(num_users/3), dataset_name, num_class, k, 'graph', 'train', few_shot)
        dataset['graph']['test'] = collect_data_iid(int(num_users/3), dataset_name, num_class, k, 'graph', 'test', few_shot)
    else:
        # node
        dataset['node']['train'] = collect_data_noniid(int(num_users/3), dataset_name, num_class, 'node', 'train', alpha)
        dataset['node']['test'] = collect_data_noniid(int(num_users/3), dataset_name, num_class, 'node', 'test', alpha)
        # edge
        dataset['edge']['train'] = collect_data_noniid(int(num_users/3), dataset_name, num_class, 'edge', 'train', alpha)
        dataset['edge']['test'] = collect_data_noniid(int(num_users/3), dataset_name, num_class, 'edge', 'test', alpha)
        # graph
        dataset['graph']['train'] = collect_data_noniid(int(num_users/3), dataset_name, num_class, 'graph', 'train', alpha)
        dataset['graph']['test'] = collect_data_noniid(int(num_users/3), dataset_name, num_class, 'graph', 'test', alpha)
    return dataset


def hetero_dir_partition(targets, num_clients, num_classes, dir_alpha, min_require_size=None):
    if min_require_size is None:
        min_require_size = num_classes
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    num_samples = targets.shape[0]

    min_size = 0

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]

        for k in range(num_classes):
            idx_k = np.where(targets == k)
            idx_k=idx_k[1]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(dir_alpha, num_clients))
            proportions = np.array(
                [p * (len(idx_j) < num_samples / num_clients) for p, idx_j in
                 zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                         zip(idx_batch, np.split(idx_k, proportions))]

            min_size = min([len(idx_j) for idx_j in idx_batch])

    client_dict = dict()
    for cid in range(num_clients):
        np.random.shuffle(idx_batch[cid])
        client_dict[cid] = np.array(idx_batch[cid])
    return client_dict

