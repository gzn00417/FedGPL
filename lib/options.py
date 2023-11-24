import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('--seed', type=int, default=2023, help="Seed")
    # data
    parser.add_argument('--dataset_name', type=str, default='CiteSeer', help='name of dataset')
    parser.add_argument('--num_classes', type=int, default=6, help="number of node classes")
    parser.add_argument('--num_users', type=int, default=9, help="number of users")
    parser.add_argument('--task_type', type=str, default='multi_class_classification', help="type of task")

    # model
    parser.add_argument('--algorithm', type=str, default='Ours', choices=['Ours', 'ProG'], help='algorithm you want to run')
    parser.add_argument('--gnn_type', type=str, default='TransformerConv', help="type of gnn")
    parser.add_argument('--token_number', type=int, default=10, help="number of tokens per_class")
    parser.add_argument('--input_dim', type=int, default=100, help='')
    parser.add_argument('--hidden_dim', type=int, default=100, help='')
    parser.add_argument('--answer_layers', type=int, default=1, help='')

    # train
    parser.add_argument('--train_batch_size', type=int, default=16, help='train batch size.')
    parser.add_argument('--val_batch_size', type=int, default=10000, help='validation batch size.')
    parser.add_argument('--epochs', type=int, default=50, help="Number of global communication rounds")
    parser.add_argument('--local_epochs', type=int, default=1, help="Number of local training epochs")
    parser.add_argument('--monitor', type=str, default='ACC', help='training monitor')
    parser.add_argument('--patience', type=int, default=16, help='')
    # train prompt
    parser.add_argument('--lr_prompt', type=float, default=0.02, help='')
    parser.add_argument('--wd_prompt', type=float, default=1e-6, help='')
    parser.add_argument('--gamma_prompt', type=float, default=0.95, help='')
    parser.add_argument('--step_size_prompt', type=int, default=64, help='')
    # train answer
    parser.add_argument('--lr_answer', type=float, default=0.02, help='')
    parser.add_argument('--wd_answer', type=float, default=1e-8, help='')
    parser.add_argument('--gamma_answer', type=float, default=0.95, help='')
    parser.add_argument('--step_size_answer', type=int, default=128, help='')

    args = parser.parse_args()
    return args
