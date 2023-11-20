from typing import Dict
import torch
from torch import nn
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
import torchmetrics
from lib.prompt import GNN, HeavyPrompt
from lib.options import args_parser
from lib.data import get_dataset
from lib.utils import PG_aggregation
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from copy import copy
import os
import wandb

os.environ['WANDB_API_KEY'] = '0ac5494dda18ee4c0537c51e9c7df96769f4a5cf'
wandb.login()


class Server(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.validation_step_outputs = []
        # init data
        train_lists_group, test_lists_group = get_dataset(args, args.num_classes, args.num_users, args.shots)
        # init pretrain GNN
        self.pre_trained_gnn = GNN(args.input_dim, hid_dim=args.hidden_dim, out_dim=args.hidden_dim, gcn_layer_num=2, gnn_type=args.gnn_type)
        self.pre_trained_gnn.load_state_dict(torch.load(f'./pre_trained_gnn/{args.data_name}.GraphCL.{args.gnn_type}.pth'))
        for p in self.pre_trained_gnn.parameters():
            p.requires_grad = False
        # init prompt & answer
        self.global_prompt = HeavyPrompt(token_dim=args.input_dim, token_num=args.token_number, cross_prune=0.1, inner_prune=0.3)
        self.global_answer = torch.nn.Sequential(torch.nn.Linear(args.hidden_dim, args.num_classes), torch.nn.Softmax(dim=1))
        # init client
        self.id2client: Dict[int, Client] = []
        for i in range(args.num_users):
            client = Client(
                server=self,
                args=args,
                train_dataset=train_lists_group[i],
                val_dataset=test_lists_group[i],
                pre_trained_gnn=self.pre_trained_gnn,
                prompt=copy(self.global_prompt),
                answer=copy(self.global_answer),
            )
            self.id2client.append(client)

    def configure_optimizers(self):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(list(range(len(self.id2client))), batch_size=1)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(list(range(len(self.id2client))), batch_size=1)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(list(range(len(self.id2client))), batch_size=1)

    def on_train_epoch_start(self):
        # FedAvg
        client_prompt_weight = []
        for client in self.id2client:
            prompt_weight = copy(client.prompt.state_dict())
            client_prompt_weight.append(prompt_weight)
        global_prompt_weight = PG_aggregation(client_prompt_weight)
        for client in self.id2client:
            client.prompt.load_state_dict(global_prompt_weight, strict=True)

    def training_step(self, batch, *args, **kwargs):
        client = self.id2client[int(batch[0])]
        client.train()

    def on_train_epoch_end(self):
        loss = [client.loss for client in self.id2client]
        overall_loss = sum(loss) / len(loss)
        self.log('train_loss', overall_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, *args, **kwargs):
        client = self.id2client[int(batch[0])]
        client.validate()
        self.validation_step_outputs.append({
            'ACC': client.acc,
            'F1': client.f1,
        })

    def on_validation_epoch_end(self):
        client_acc, client_f1 = [], []
        for item in self.validation_step_outputs:
            client_acc.append(item['ACC'])
            client_f1.append(item['F1'])
        overall_acc = sum(client_acc) / len(client_acc)
        overall_f1 = sum(client_f1) / len(client_f1)
        self.log('ACC', overall_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('F1', overall_f1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.validation_step_outputs.clear()

    def test_step(self, *args, **kwargs):
        self.validation_step(*args, **kwargs)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()


class Client(object):

    def __init__(
        self,
        server: Server,
        args,
        train_dataset,
        val_dataset,
        pre_trained_gnn: torch.nn.Module,
        prompt: torch.nn.Module,
        answer: torch.nn.Module
    ):
        super().__init__()
        self.automatic_optimization = False
        self.server = server
        self.args = args
        # data
        self.train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, pin_memory=True)
        # model
        self.pre_trained_gnn = pre_trained_gnn
        self.prompt = prompt
        self.answer = answer
        # init
        self.configure_optimizers()
        self.configure_evaluation()

    def forward(self, x):
        prompted_graph = self.prompt(x)
        graph_emb = self.pre_trained_gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
        return self.answer(graph_emb)

    def configure_optimizers(self):
        self.optimizer_prompt = torch.optim.Adam(self.prompt.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        self.optimizer_answer = torch.optim.Adam(self.answer.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        self.scheduler_prompt = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer_prompt, step_size=self.args.step_size, gamma=self.args.gamma)
        self.scheduler_answer = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer_answer, step_size=self.args.step_size, gamma=self.args.gamma)

    def configure_evaluation(self):
        self.loss_function = nn.CrossEntropyLoss(reduction='mean')
        self.accuracy_function = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.num_classes)
        self.f1_function = torchmetrics.classification.F1Score(task="multiclass", num_classes=args.num_classes, average="macro")
        self.loss, self.acc, self.f1 = None, None, None

    def train(self):
        for batch in self.train_dataloader:
            pred = self.forward(batch.to(self.server.device))
            loss = self.loss_function(pred, batch.y)
            self.loss = loss.item()
            self.optimizer_prompt.zero_grad()
            self.optimizer_answer.zero_grad()
            loss.backward()
            self.optimizer_prompt.step()
            self.optimizer_answer.step()
            self.scheduler_prompt.step()
            self.scheduler_answer.step()

    def validate(self):
        for batch in self.val_dataloader:
            pred = self.forward(batch.to(self.server.device))
            y = batch.y.cpu()
            pred_class = torch.argmax(pred, dim=1).cpu()
            acc = self.accuracy_function(pred_class, y)
            f1 = self.f1_function(pred_class, y)
        acc = self.accuracy_function.compute()
        f1 = self.f1_function.compute()
        self.accuracy_function.reset()
        self.f1_function.reset()
        self.acc = acc.item()
        self.f1 = f1.item()


if __name__ == '__main__':
    # hyper parameters
    args = args_parser()
    # training module
    module: pl.LightningModule = Server(args)
    # train & test
    early_stopping_callback = EarlyStopping(monitor=args.monitor, mode='max', min_delta=0.0, patience=args.patience, verbose=False, check_finite=True)
    model_checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor=args.monitor, mode='max')
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=args.epochs,
        logger=WandbLogger(
            save_dir='./',
            name='log',
            log_model='all',
        ),
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        enable_checkpointing=True,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0
    )
    print('Training model...')
    trainer.fit(module)
    # print('Testing model...')
    # trainer.test(module, ckpt_path='best')
