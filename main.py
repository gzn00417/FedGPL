import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os
import wandb

from lib import *

os.environ['WANDB_API_KEY'] = '0ac5494dda18ee4c0537c51e9c7df96769f4a5cf'
wandb.login()

if __name__ == '__main__':
    # hyper parameters
    args = args_parser()
    pl.seed_everything(args.seed)

    # model
    pre_trained_gnn = GNN(args.input_dim, hid_dim=args.hidden_dim, out_dim=args.hidden_dim, gcn_layer_num=2, gnn_type=args.gnn_type)
    pre_trained_gnn.load_state_dict(torch.load(f'./pre_trained_gnn/{args.dataset_name}.GraphCL.{args.gnn_type}.pth'))
    for p in pre_trained_gnn.parameters():
        p.requires_grad = False
    if args.algorithm == 'Ours':
        from lib import Ours
        prompt = Ours(token_dim=args.input_dim, token_num=args.token_number, group_num=1, cross_prune=0.1, inner_prune=0.3)
    elif args.algorithm == 'ProG':
        from lib import HeavyPrompt
        prompt = HeavyPrompt(token_dim=args.input_dim, cross_prune=0.1, inner_prune=0.3)
    answer = Answer(args.hidden_dim, args.num_classes, args.answer_layers)

    # training module
    module: pl.LightningModule = Server(args, pre_trained_gnn, prompt, answer)
    # train & test
    early_stopping_callback = EarlyStopping(monitor=args.monitor, mode='max', min_delta=0.0, patience=args.patience, verbose=False, check_finite=True)
    model_checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor=args.monitor, mode='max')
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=args.epochs,
        logger=WandbLogger(
            save_dir='./',
            name='Federated-Graph-Prompt',
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
