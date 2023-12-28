from typing import Dict, List
import torch
from torch import nn
from torch_geometric.loader import DataLoader
import torchmetrics
import pytorch_lightning as pl
from copy import deepcopy

from .data import get_dataset
from .model.federated import *
from .model.dp import *

class Server(pl.LightningModule):

    def __init__(self, pre_trained_gnn: nn.Module, prompt: nn.Module, answer: nn.Module, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['pre_trained_gnn', 'prompt', 'answer'])
        self.automatic_optimization = False
        self.validation_step_outputs: Dict[str, List] = {'node': [], 'edge': [], 'graph': []}
        # init pretrain GNN
        self.pre_trained_gnn = pre_trained_gnn
        # init prompt & answer
        self.global_prompt = prompt
        self.global_answer = answer
        # init clients
        self.init_clients()

    def init_clients(self):
        # init data
        dataset = get_dataset(self.hparams.dataset_name, self.hparams.num_classes, self.hparams.num_users, self.hparams.data_type, self.hparams.alpha)
        self.client_list = []
        self.client_list_by_task: Dict[str, List] = {'node': [], 'edge': [], 'graph': []}
        num_clients_per_task = self.hparams.num_users // 3
        # node task
        for i in range(num_clients_per_task):
            client = self.init_client('node', dataset['node']['train'][i], dataset['node']['test'][i])
            self.client_list.append(client)
            self.client_list_by_task['node'].append(client)
        # edge task
        for i in range(num_clients_per_task):
            client = self.init_client('edge', dataset['edge']['train'][i], dataset['edge']['test'][i])
            self.client_list.append(client)
            self.client_list_by_task['edge'].append(client)
        # graph task
        for i in range(num_clients_per_task):
            client = self.init_client('graph', dataset['graph']['train'][i], dataset['graph']['test'][i])
            self.client_list.append(client)
            self.client_list_by_task['graph'].append(client)

    def init_client(self, task, train_dataset, test_dataset):
        return Client(
            task=task,
            hparams=self.hparams,
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            pre_trained_gnn=self.pre_trained_gnn,
            prompt=deepcopy(self.global_prompt),
            answer=deepcopy(self.global_answer),
        )

    def configure_optimizers(self):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(list(range(len(self.client_list))), batch_size=1)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(list(range(len(self.client_list))), batch_size=1)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(list(range(len(self.client_list))), batch_size=1)

    def on_fit_start(self):
        for client in self.client_list:
            client.to(self.device)

    def on_train_epoch_start(self):

        if self.hparams.federated == 'Local':
            return
        elif self.hparams.federated == 'FedAvg':
            fed_avg_prompt(self.client_list)
            fed_avg_answer(self.client_list)
        elif self.hparams.federated == 'TaskAvg':
            if self.current_epoch != 0:
                # add dp (Federated Phase)
                for client in self.client_list:
                    client.prompt.load_state_dict(add_noise_for_state_dict(client.prompt.state_dict(), self.hparams.epsilon))
                    client.answer.load_state_dict(add_noise_for_state_dict(client.answer.state_dict(), self.hparams.epsilon))
                prompt_coefficient, answer_coefficient = compute_coefficient(self.client_list_by_task, self.hparams.lr_prompt, self.hparams.lr_answer)
                weighted_task_fed_avg(self.client_list_by_task, prompt_coefficient, answer_coefficient, self.hparams.epsilon)
        else:
            raise Exception(f'Unknown federated optimization for {self.hparams.federated}.')

    def training_step(self, batch, *args, **kwargs):
        client = self.client_list[int(batch[0])]
        client.train(self.device, self.hparams.local_epochs)

    def on_train_epoch_end(self):
        loss = [client.loss for client in self.client_list]
        overall_loss = sum(loss) / len(loss)
        self.log('train_loss', overall_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, *args, **kwargs):
        client = self.client_list[int(batch[0])]
        client.validate(self.device)
        self.validation_step_outputs[client.task].append({'ACC': client.acc, 'F1': client.f1})

    def on_validation_epoch_end(self):
        overall_acc, overall_f1 = [], []
        for task in ['node', 'edge', 'graph']:
            client_acc, client_f1 = [], []
            for item in self.validation_step_outputs[task]:
                client_acc.append(item['ACC'])
                client_f1.append(item['F1'])
            self.validation_step_outputs[task].clear()
            task_acc = sum(client_acc) / len(client_acc)
            task_f1 = sum(client_f1) / len(client_f1)
            self.log(f'ACC_{task}', task_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'F1_{task}', task_f1, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            overall_acc.append(task_acc)
            overall_f1.append(task_f1)
        overall_acc = sum(overall_acc) / len(overall_acc)
        overall_f1 = sum(overall_f1) / len(overall_f1)
        self.log('ACC', overall_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('F1', overall_f1, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    def test_step(self, *args, **kwargs):
        self.validation_step(*args, **kwargs)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()


class Client(object):

    def __init__(
        self,
        task: str,
        hparams,
        train_dataset,
        val_dataset,
        pre_trained_gnn: torch.nn.Module,
        prompt: torch.nn.Module,
        answer: torch.nn.Module,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.task = task
        self.hparams = hparams
        # data
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, shuffle=True, pin_memory=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.hparams.val_batch_size, shuffle=False, pin_memory=True)
        # model
        self.pre_trained_gnn = pre_trained_gnn
        self.prompt = prompt
        self.answer = answer

        self.prompt_last_epoch = prompt
        self.answer_last_epoch = answer

        # pravcy
        self.epsilon = self.hparams.epsilon

        # init
        self.configure_optimizers()
        self.configure_evaluation()

    def forward(self, x):
        x, edge_index, batch = self.prompt(x)

        graph_emb = self.pre_trained_gnn(x, edge_index, batch)

        # return self.answer(graph_emb), graph_emb.clone().detach().requires_grad_(True), graph_emb, x.clone().detach().requires_grad_(True), x
        return self.answer(graph_emb)

    def configure_optimizers(self):
        self.optimizer_prompt = torch.optim.Adam(self.prompt.parameters(), lr=self.hparams.lr_prompt, weight_decay=self.hparams.wd_prompt)
        self.optimizer_answer = torch.optim.Adam(self.answer.parameters(), lr=self.hparams.lr_answer, weight_decay=self.hparams.wd_answer)
        self.scheduler_prompt = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer_prompt, step_size=self.hparams.step_size_prompt, gamma=self.hparams.gamma_prompt)
        self.scheduler_answer = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer_answer, step_size=self.hparams.step_size_answer, gamma=self.hparams.gamma_answer)

    def configure_evaluation(self):
        self.loss_function = nn.CrossEntropyLoss(reduction='mean')
        self.accuracy_function = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.f1_function = torchmetrics.classification.F1Score(task="multiclass", num_classes=self.hparams.num_classes, average="macro")
        self.loss, self.acc, self.f1 = None, None, None

    def to(self, device):
        self.prompt = self.prompt.to(device)
        self.answer = self.answer.to(device)
        self.accuracy_function = self.accuracy_function.to(device)
        self.f1_function = self.f1_function.to(device)

    def train(self, device: str, local_epochs: int = 5):
        self.prompt_last_epoch = deepcopy(self.prompt)
        self.answer_last_epoch = deepcopy(self.answer)
        for _ in range(local_epochs):
            for batch in self.train_dataloader:
                x, edge_index, tbatch = self.prompt(batch.to(device))
                # add dp (Local Training Phase)
                x = add_laplace_noise(x, self.epsilon)
                x_rg = x.clone().detach().requires_grad_(True)

                graph_emb = self.pre_trained_gnn(x_rg, edge_index, tbatch)
                graph_emb_rg = graph_emb.clone().detach().requires_grad_(True)

                pred = self.answer(graph_emb_rg)
                loss = self.loss_function(pred, batch.y)
                self.loss = loss.item()
                self.optimizer_answer.zero_grad()
                loss.backward()
                self.optimizer_answer.step()
                embedding_grad = graph_emb_rg.grad.clone()
                # add dp (Local Training Phase)
                embedding_grad = add_laplace_noise(embedding_grad, self.epsilon)

                if x_rg.grad is not None:
                    x_rg.grad.data.zero_()
                graph_emb.backward(embedding_grad)
                
                x_grad = x_rg.grad.clone()
                # add dp (Local Training Phase)
                x_grad = add_laplace_noise(x_grad, self.epsilon)

                self.optimizer_prompt.zero_grad()
                x.backward(x_grad)
                self.optimizer_prompt.step()


    def validate(self, device: str):
        for batch in self.val_dataloader:
            pred = self.forward(batch.to(device))
            y = batch.y.to(device)
            pred_class = torch.argmax(pred, dim=1).to(device)
            acc = self.accuracy_function(pred_class, y)
            f1 = self.f1_function(pred_class, y)
        acc = self.accuracy_function.compute()
        f1 = self.f1_function.compute()
        self.accuracy_function.reset()
        self.f1_function.reset()
        self.acc = acc.item()
        self.f1 = f1.item()
