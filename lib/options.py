import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('-S', '--seed', type=int, default=2023, help='Seed')

    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--cross_prune', type=float, default=0.1)
    # data
    parser.add_argument('--few_shot', action='store_true', default=False)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('-D', '--dataset_name', type=str, default='CiteSeer', help='name of dataset')
    parser.add_argument('-N', '--num_classes', type=int, default=6, help='number of node classes')
    parser.add_argument('-C', '--num_users', type=int, default=9, help='number of users')

    # model
    parser.add_argument('-F', '--federated', type=str, default='HiDTA', choices=['FedAvg', 'HiDTA', 'scaffold', 'FedProx'], help='')
    parser.add_argument('-A', '--algorithm', type=str, default='VPG', choices=['VPG', 'ProG', 'GPF'], help='algorithm you want to run')
    parser.add_argument('-G', '--gnn_type', type=str, default='TransformerConv', help='type of gnn')
    parser.add_argument('-T', '--token_number', type=int, default=0, help='number of tokens per_class')
    parser.add_argument('--input_dim', type=int, default=100, help='')
    parser.add_argument('--hidden_dim', type=int, default=100, help='')
    parser.add_argument('--answer_layers', type=int, default=1, help='')

    # train
    parser.add_argument('--pre_train_algorithm', type=str, default='GraphCL', choices=['GraphCL', 'SimGRACE'], help='graph pre-training algorithm')
    parser.add_argument('--train_batch_size', type=int, default=16, help='train batch size.')
    parser.add_argument('--val_batch_size', type=int, default=10000, help='validation batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of global communication rounds')
    parser.add_argument('--local_epochs', type=int, default=1, help='Number of local training epochs')
    parser.add_argument('--monitor', type=str, default='ACC', help='training monitor')
    parser.add_argument('--patience', type=int, default=16, help='')
    # train prompt
    parser.add_argument('--lr_prompt', type=float, default=0.15, help='')
    parser.add_argument('--wd_prompt', type=float, default=1e-6, help='')
    parser.add_argument('--gamma_prompt', type=float, default=0.95, help='')
    parser.add_argument('--step_size_prompt', type=int, default=10, help='')
    # train answer
    parser.add_argument('--lr_answer', type=float, default=0.2, help='')
    parser.add_argument('--wd_answer', type=float, default=1e-8, help='')
    parser.add_argument('--gamma_answer', type=float, default=0.95, help='')
    parser.add_argument('--step_size_answer', type=int, default=10, help='')

    # data distribution
    parser.add_argument('--data_type', type=str, default='iid', help='')
    parser.add_argument('--alpha', type=float, default=0.3, help='')

    # privacy
    parser.add_argument('--epsilon', type=float, default=100, help='')

    # scaffold
    parser.add_argument('--lr_global_control_prompt', type=float, default=0.02)
    parser.add_argument('--lr_global_control_answer', type=float, default=0.02)
    parser.add_argument('--lr_global_prompt', type=float, default=0.02)
    parser.add_argument('--lr_global_answer', type=float, default=0.02)

    # FedProx
    parser.add_argument('--lr_prox', type=float, default=0.1)

    args = parser.parse_args()
    return args
