#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithm', type=str, default='ProG', help='algorithm you want to run')
    parser.add_argument('--data_name', type=str, default='CiteSeer', help='name of dataset')
    parser.add_argument('--shots', type=int, default=100, help="num of shots")
    parser.add_argument('--num_classes', type=int, default=6, help="number of node classes")
    parser.add_argument('--num_users', type=int, default=9, help="number of users")
    parser.add_argument('--gnn_type', type=str, default='TransformerConv', help="type of gnn")
    parser.add_argument('--task_type', type=str, default='multi_class_classification', help="type of task")
    parser.add_argument('--rounds', type=int, default=50, help="Number of global communication rounds")
    parser.add_argument('--local_epochs', type=int, default=1, help="Number of local training epochs")
    parser.add_argument('--seed', type=int, default=2023, help="Seed")
    parser.add_argument('--token_number', type=int, default=10, help="number of tokens per_class")

    args = parser.parse_args()
    return args
