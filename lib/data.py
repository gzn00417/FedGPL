import torch
import pickle as pk
import random
from random import shuffle
from torch.utils.data import DataLoader, Dataset

from torch_geometric.data import Batch
from collections import defaultdict


def multi_class_NIG(dataname, num_class,shots=100):
    """
    NIG: node induced graphs
    :param dataname: e.g. CiteSeer
    :param num_class: e.g. 6 (for CiteSeer)
    :return: a batch of training graphs and testing graphs
    """
    statistic = defaultdict(list)

    # load training NIG (node induced graphs)
    train_list = []
    for task_id in range(num_class):
        data_path1 = './data/{}/induced_graphs/task{}.meta.train.support'.format(dataname, task_id)
        data_path2 = './data/{}/induced_graphs/task{}.meta.train.query'.format(dataname, task_id)

        with (open(data_path1, 'br') as f1, open(data_path2, 'br') as f2):
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            # data_list = data_list[0:shots]
            data_list = random.sample(data_list, shots)
            # data_list = data_list[:shots]
            statistic['train'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                train_list.append(g)

    shuffle(train_list)
    train_data = Batch.from_data_list(train_list)

    test_list = []
    for task_id in range(num_class):
        data_path1 = './data/{}/induced_graphs/task{}.meta.test.support'.format(dataname, task_id)
        data_path2 = './data/{}/induced_graphs/task{}.meta.test.query'.format(dataname, task_id)

        with (open(data_path1, 'br') as f1, open(data_path2, 'br') as f2):
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            # data_list = data_list[0:shots]
            data_list = random.sample(data_list, shots)

            statistic['test'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                test_list.append(g)

    shuffle(test_list)
    test_data = Batch.from_data_list(test_list)

    for key, value in statistic.items():
        print("{}ing set (class_id, graph_num): {}".format(key, value))

    # train_loader = DataLoader(train_list, batch_size=100, shuffle=True)
    # for batch_id, train_batch in enumerate(train_loader):
    #     print("????????????????")

    return train_data, test_data, train_list,test_list

def multi_class_EIG(dataname, num_class,shots=100):
    """
    NIG: node induced graphs
    :param dataname: e.g. CiteSeer
    :param num_class: e.g. 6 (for CiteSeer)
    :return: a batch of training graphs and testing graphs
    """
    statistic = defaultdict(list)

    # load training NIG (node induced graphs)
    train_list = []
    for task_id in range(num_class):
        task_id = task_id + num_class
        data_path1 = './data/{}/induced_graphs/task{}.meta.train.support'.format(dataname, task_id)
        data_path2 = './data/{}/induced_graphs/task{}.meta.train.query'.format(dataname, task_id)

        with (open(data_path1, 'br') as f1, open(data_path2, 'br') as f2):
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['train'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id - num_class
                train_list.append(g)

    shuffle(train_list)
    train_data = Batch.from_data_list(train_list)

    test_list = []
    for task_id in range(num_class):
        temp = task_id + num_class
        data_path1 = './data/{}/induced_graphs/task{}.meta.test.support'.format(dataname, temp)
        data_path2 = './data/{}/induced_graphs/task{}.meta.test.query'.format(dataname, temp)

        with (open(data_path1, 'br') as f1, open(data_path2, 'br') as f2):
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]

            statistic['test'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                test_list.append(g)

    shuffle(test_list)
    test_data = Batch.from_data_list(test_list)

    for key, value in statistic.items():
        print("{}ing set (class_id, graph_num): {}".format(key, value))

    return train_data, test_data, train_list,test_list

def multi_class_GIG(dataname, num_class,shots=100):
    """
    NIG: node induced graphs
    :param dataname: e.g. CiteSeer
    :param num_class: e.g. 6 (for CiteSeer)
    :return: a batch of training graphs and testing graphs
    """
    statistic = defaultdict(list)

    # load training NIG (node induced graphs)
    train_list = []
    for task_id in range(num_class):
        task_id = task_id + num_class*2
        data_path1 = './data/{}/induced_graphs/task{}.meta.train.support'.format(dataname, task_id)
        data_path2 = './data/{}/induced_graphs/task{}.meta.train.query'.format(dataname, task_id)

        with (open(data_path1, 'br') as f1, open(data_path2, 'br') as f2):
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['train'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id - num_class*2
                train_list.append(g)

    shuffle(train_list)
    train_data = Batch.from_data_list(train_list)

    test_list = []
    for task_id in range(num_class):
        temp = task_id + num_class*2
        data_path1 = './data/{}/induced_graphs/task{}.meta.test.support'.format(dataname, temp)
        data_path2 = './data/{}/induced_graphs/task{}.meta.test.query'.format(dataname, temp)

        with (open(data_path1, 'br') as f1, open(data_path2, 'br') as f2):
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]

            statistic['test'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                test_list.append(g)

    shuffle(test_list)
    test_data = Batch.from_data_list(test_list)

    for key, value in statistic.items():
        print("{}ing set (class_id, graph_num): {}".format(key, value))

    return train_data, test_data, train_list,test_list


def get_dataset(args, num_class, num_users, shots):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    dataname = args.data_name
    train_lists_group = []
    test_lists_group = []
    for i in range(1, num_users+1):
        if i<4:
            _, _, train_list, test_list = multi_class_NIG(dataname, num_class, shots)
        elif i<7:
            _, _, train_list, test_list = multi_class_EIG(dataname, num_class, shots)
        else:
            _, _, train_list, test_list = multi_class_GIG(dataname, num_class, shots)
        train_lists_group.append(train_list)
        test_lists_group.append(test_list)

    
    return train_lists_group, test_lists_group


if __name__ == '__main__':
    pass
