import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
import copy
import numpy as np
from lib.eva import acc_f1_over_batches


class LocalUpdate(object):
    def __init__(self, args, train_dataset, test_dataset, task_type):
        self.args = args
        self.train_loader = DataLoader(train_dataset, batch_size=args.shots, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=args.shots, shuffle=False)
        self.device = args.device

        self.input_dim, self.hid_dim = 100, 100
        self.lr, self.wd = 0.01, 1e-8
        self.tnpc = args.token_number  # token number

        self.task_type = task_type

    def update_weights(self, PG, answering, gnn):
        # Set mode to train model

        lossfn = nn.CrossEntropyLoss(reduction='mean')

        lossfn.to(self.device)

        opi_pg = optim.Adam(filter(lambda p: p.requires_grad, PG.parameters()), lr=self.lr, weight_decay=self.wd)
        opi_answer = optim.Adam(filter(lambda p: p.requires_grad, answering.parameters()), lr=0.01, weight_decay=1e-8)      

        local_loss, acc, macro_f1= [], [], []

        for j in range(0, self.args.local_epochs):
            running_loss = 0

            # Training PG; Forzing Head
            PG.train()
            answering.eval()

            for batch_id, train_batch in enumerate(self.train_loader):

                train_batch = train_batch.to(self.device)

                prompted_graph = PG(train_batch)
                graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
                pre = answering(graph_emb)

                loss = lossfn(pre, train_batch.y)

                opi_pg.zero_grad()
                loss.backward()
                opi_pg.step()
            

            # Training PG; Forzing Head
            PG.eval()
            answering.train()

            for batch_id, train_batch in enumerate(self.train_loader):

                train_batch = train_batch.to(self.device)

                prompted_graph = PG(train_batch)
                graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
                pre = answering(graph_emb)

                loss = lossfn(pre, train_batch.y)

                opi_answer.zero_grad()
                loss.backward()
                opi_answer.step()
                running_loss += loss.item()

                if batch_id % 5 == 4:  # report every 5 updates
                    last_loss = running_loss / 5  # loss per batch
                    print('epoch {}/{} | batch {}/{} | loss: {:.8f}'.format(j, self.args.local_epochs, batch_id+1, len(self.train_loader), last_loss))
                local_loss.append(running_loss)
                running_loss = 0

            # Evaluation
            PG.eval()
            answering.eval()

            acc_epoch, macro_f1_epoch = acc_f1_over_batches(self.test_loader, PG, gnn, answering, self.args.num_classes, self.task_type, self.device)
            acc.append(acc_epoch)
            macro_f1.append(macro_f1_epoch)

            PG = PG.to(self.device)
            gnn = gnn.to(self.device) 
            answering = answering.to(self.device)

        return PG.state_dict(), answering.state_dict(), sum(local_loss) / len(local_loss), sum(acc)/len(acc), sum(macro_f1)/len(macro_f1)