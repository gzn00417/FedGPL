import pickle as pk
from random import shuffle
from torch_geometric.data import Batch


def collect_data(dataset_name, num_class, task_type: str = 'node', dataset_type: str = 'train'):
    tmp = {'node': 0, 'edge': 1, 'graph': 2}.get(task_type) * num_class
    data = []
    for task_id in range(num_class):
        data_path1 = f'./data/{dataset_name}/induced_graphs/task{task_id + tmp}.meta.{dataset_type}.support'
        data_path2 = f'./data/{dataset_name}/induced_graphs/task{task_id + tmp}.meta.{dataset_type}.query'
        with (open(data_path1, 'br') as f1, open(data_path2, 'br') as f2):
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            for g in data_list:
                g.y = task_id
                data.append(g)
    shuffle(data)
    return Batch.from_data_list(data)


def get_dataset(dataset_name, num_class, num_users):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    dataset = {
        'node': {'train': [], 'test': []},
        'edge': {'train': [], 'test': []},
        'graph': {'train': [], 'test': []},
    }
    # node
    for i in range(num_users // 3):
        train_data = collect_data(dataset_name, num_class, 'node', 'train')
        test_data = collect_data(dataset_name, num_class, 'node', 'test')
        dataset['node']['train'].append(train_data)
        dataset['node']['test'].append(test_data)
    # edge
    for i in range(num_users // 3):
        train_data = collect_data(dataset_name, num_class, 'edge', 'train')
        test_data = collect_data(dataset_name, num_class, 'edge', 'test')
        dataset['edge']['train'].append(train_data)
        dataset['edge']['test'].append(test_data)
    # graph
    for i in range(num_users // 3):
        train_data = collect_data(dataset_name, num_class, 'graph', 'train')
        test_data = collect_data(dataset_name, num_class, 'graph', 'test')
        dataset['graph']['train'].append(train_data)
        dataset['graph']['test'].append(test_data)
    return dataset
