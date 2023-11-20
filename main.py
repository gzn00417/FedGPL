import copy, sys
import numpy as np
from tqdm import tqdm
import torch
import random
from pathlib import Path
import os
import wandb

os.environ['WANDB_API_KEY'] = '0ac5494dda18ee4c0537c51e9c7df96769f4a5cf'
wandb.login()

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from lib.update import LocalUpdate
from lib.utils import exp_details, PG_aggregation
from lib.data import get_dataset
from lib.options import args_parser
from lib.prompt import GNN, HeavyPrompt

wandb.init(
    project = "Federated-Graph-Prompt",
    config = {
        "learning_rate": 0.01,
        "architecture": "GNN",
        "dataset": "CiteSeer",
        "epochs": 200,
    }
)

def FedProG(args, train_lists_group, test_lists_group, local_PG_list, local_ans_list, gnn, task_type):

    idxs_users = np.arange(args.num_users)

    train_loss = []

    for round in tqdm(range(args.rounds)):

        local_PG_weights, local_answing_weights = [], []
        local_losses, local_acc, local_macro_f1 =  [], [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        for idx in idxs_users:
            local_PG = LocalUpdate(args=args, train_dataset=train_lists_group[idx], test_dataset=test_lists_group[idx], task_type=task_type)

            PG_weight, answ_weight, loss, acc, macro_f1= local_PG.update_weights(PG=copy.deepcopy(local_PG_list[idx]), answering=copy.deepcopy(local_ans_list[idx]), gnn=gnn)

            local_PG_weights.append(copy.deepcopy(PG_weight))
            local_answing_weights.append(copy.deepcopy(answ_weight))

            local_losses.append(copy.deepcopy(loss))
            local_acc.append(copy.deepcopy(acc))
            local_macro_f1.append(copy.deepcopy(macro_f1))

        # Aggregating all PGs
        aggregated_PG = PG_aggregation(local_PG_weights)

        # local_weights_list = local_PG_weights
        local_answing_weights_list = local_answing_weights

        # Update all local PGs with aggregated PG
        for idx in idxs_users:
            local_PG = copy.deepcopy(local_PG_list[idx])
            # local_PG.load_state_dict(local_weights_list[idx], strict=True)
            local_PG.load_state_dict(aggregated_PG, strict=True)
            local_PG_list[idx] = local_PG

        for idx in idxs_users:
            local_answing = copy.deepcopy(local_ans_list[idx])
            local_answing.load_state_dict(local_answing_weights_list[idx], strict=True)
            local_ans_list[idx] = local_answing

        # Compute the average loss, acc and avg_f1 of all local trainings in this round
        loss_avg = sum(local_losses) / len(local_losses)
        avg_acc = sum(local_acc) / len(local_acc)
        avg_f1 = sum(local_macro_f1) / len(local_macro_f1)
        wandb.log({"acc": avg_acc, "macro_f1": avg_f1, "loss_total": loss_avg})

        # Compute the average loss, acc and avg_f1 of different tasks
        # Here, node task: 0-2; edge task: 3-5; graph task: 6-8
        train_loss.append(loss_avg)
        print("Node-level Acc:")
        print(sum(local_acc[:3])/3)
        print("Edge-level Acc:")
        print(sum(local_acc[3:6])/3)
        print("Graph-level Acc:")
        print(sum(local_acc[-3:])/3)

    wandb.finish()

if __name__ == '__main__':

    args = args_parser()
    exp_details(args)

    input_dim, hid_dim = 100, 100
    tnpc = args.token_number  # token number per class

    # set random seeds
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    # load dataset and user groups

    train_lists_group, test_lists_group = get_dataset(args, args.num_classes ,args.num_users, args.shots)

    # Build models
    if args.algorithm == 'ProG':
        local_gnn_list, local_PG_list, local_ans_list = [], [], []

        # Load the pretrained GNNs, for simplicity, actually use one GNN model for all clients
        gnn = GNN(input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=2, gnn_type=args.gnn_type)
        pre_train_path = './pre_trained_gnn/{}.GraphCL.{}.pth'.format(args.data_name, args.gnn_type)
        gnn.load_state_dict(torch.load(pre_train_path))
        print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))
        for p in gnn.parameters():
            p.requires_grad = False

        for i in range(0, args.num_users):

            # Build Prompts & head layers (answering)
            PG = HeavyPrompt(token_dim=input_dim, token_num=args.token_number, cross_prune=0.1, inner_prune=0.3)
            answering = torch.nn.Sequential(
                        torch.nn.Linear(hid_dim, args.num_classes),
                        torch.nn.Softmax(dim=1))
            
            gnn.to(args.device)
            PG.to(args.device)
            answering.to(args.device)

            local_gnn_list.append(gnn)
            local_PG_list.append(PG)
            local_ans_list.append(answering)
            
        FedProG(args, train_lists_group, test_lists_group, local_PG_list, local_ans_list, gnn, args.task_type)
