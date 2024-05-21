import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os
import wandb

from lib import *

os.environ['WANDB_API_KEY'] = ''
wandb.login()
wandb.init(
    entity='',
    project='FedGPL',
)

if __name__ == '__main__':
    # hyper parameters
    args = args_parser()
    pl.seed_everything(args.seed)
    print(args.few_shot)
    if args.algorithm == 'VPG':
        wandb.run.name = str(args.dataset_name) + '_' + str(args.federated) + '_' + 'E+G' + '_' + str(args.algorithm) + '' + str(args.ratio) + str(args.few_shot) + 'new'
    else:
        wandb.run.name = str(args.dataset_name) + '_' + str(args.federated) + '_' + 'E+G' + '_' + str(args.algorithm) + '' + str(args.cross_prune) + str(args.few_shot) + 'new'
    # model
    pre_trained_gnn = GNN(args.input_dim, hid_dim=args.hidden_dim, out_dim=args.hidden_dim, gcn_layer_num=2, gnn_type=args.gnn_type)
    pre_trained_gnn.load_state_dict(torch.load(f'./pre_trained_gnn/{args.dataset_name}.{args.pre_train_algorithm}.{args.gnn_type}.pth', map_location='cuda:3'))
    for p in pre_trained_gnn.parameters():
        p.requires_grad = False
    if args.algorithm == 'VPG':
        from lib import Ours
        # prompt = Ours(token_dim=args.input_dim, token_num=args.token_number)
        prompt = Ours(token_dim=args.input_dim, ratio=args.ratio)
    elif args.algorithm == 'ProG':
        from lib import HeavyPrompt
        prompt = HeavyPrompt(token_dim=args.input_dim, cross_prune=args.cross_prune)
    elif args.algorithm == 'GPF':
        from lib import GPF
        prompt = GPF(token_dim=args.input_dim, cross_prune=args.cross_prune)
    answer = Answer(args.hidden_dim, args.num_classes, args.answer_layers)

    # training module
    module: pl.LightningModule = Server(pre_trained_gnn, prompt, answer, **vars(args))
    # train & test
    early_stopping_callback = EarlyStopping(monitor=args.monitor, mode='max', min_delta=0.0, patience=args.patience, verbose=False, check_finite=True)
    model_checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor=args.monitor, mode='max')
    trainer = Trainer(
        accelerator='gpu',
        devices=[3],
        max_epochs=args.epochs,
        logger=WandbLogger(save_dir='./'),
        callbacks=[model_checkpoint_callback],
        enable_checkpointing=True,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0
    )
    print('Training model...')
    trainer.fit(module)
    # print('Testing model...')
    # trainer.test(module, ckpt_path='best')
