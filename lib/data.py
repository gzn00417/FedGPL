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


def multi_class_NIG(dataset_name, num_class):
    """
    NIG: node induced graphs
    :param dataset_name: e.g. CiteSeer
    :param num_class: e.g. 6 (for CiteSeer)
    :return: a batch of training graphs and testing graphs
    """
    train_data = collect_data(dataset_name, num_class, 'node', 'train')
    test_data = collect_data(dataset_name, num_class, 'node', 'test')
    return train_data, test_data


def multi_class_EIG(dataset_name, num_class):
    """
    NIG: node induced graphs
    :param dataname: e.g. CiteSeer
    :param num_class: e.g. 6 (for CiteSeer)
    :return: a batch of training graphs and testing graphs
    """
    train_data = collect_data(dataset_name, num_class, 'edge', 'train')
    test_data = collect_data(dataset_name, num_class, 'edge', 'test')
    return train_data, test_data


def multi_class_GIG(dataset_name, num_class):
    """
    NIG: node induced graphs
    :param dataname: e.g. CiteSeer
    :param num_class: e.g. 6 (for CiteSeer)
    :return: a batch of training graphs and testing graphs
    """
    train_data = collect_data(dataset_name, num_class, 'graph', 'train')
    test_data = collect_data(dataset_name, num_class, 'graph', 'test')
    return train_data, test_data


def get_dataset(args, num_class, num_users):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    dataset_name = args.data_name
    train_lists_group = []
    test_lists_group = []
    for i in range(1, num_users+1):
        if i < 4:
            train_list, test_list = multi_class_NIG(dataset_name, num_class)
        elif i < 7:
            train_list, test_list = multi_class_EIG(dataset_name, num_class)
        else:
            train_list, test_list = multi_class_GIG(dataset_name, num_class)
        train_lists_group.append(train_list)
        test_lists_group.append(test_list)
    return train_lists_group, test_lists_group
