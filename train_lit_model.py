import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from pathlib import Path
from argparse import ArgumentParser
from LitModel import *
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.utilities.distributed import rank_zero_only

seed_everything(1994)

def setup_callbacks_loggers(args):
    
    log_path = Path('/home/yyousfi1/LogFiles/OneHotConv/')
    log_path = log_path/args.qf/args.stego_scheme/args.payload
    name = args.backbone
    version = args.version
    tb_logger = TensorBoardLogger(log_path, name=name, version=version)
    lr_logger = LearningRateLogger(logging_interval='epoch')
    ckpt_callback = ModelCheckpoint(filepath=Path(tb_logger.log_dir)/'checkpoints/{epoch:02d}_{val_FC_acc:.3f}', 
                                    save_top_k=5, save_last=True)
   
    return ckpt_callback, tb_logger, lr_logger


def main(args):
    """ Main training routine """
    
    model = LitModel(**vars(args))
    
    ckpt_callback, tb_logger, lr_logger = setup_callbacks_loggers(args)
    
    trainer = Trainer(checkpoint_callback=ckpt_callback,
                     logger=tb_logger,
                     callbacks=[lr_logger],
                     gpus=args.gpus,
                     min_epochs=args.epochs,
                     max_epochs=args.epochs,
                     precision=32,
                     row_log_interval=100,
                     log_save_interval=100,
                     distributed_backend='ddp',
                     benchmark=True,
                     sync_batchnorm=True,
                     resume_from_checkpoint=args.resume_from_checkpoint)
    
    trainer.logger.log_hyperparams(model.hparams)
    
    trainer.fit(model)


def run_cli():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    
    parent_parser = ArgumentParser(add_help=False)

    parser = LitModel.add_model_specific_args(parent_parser)
    
    parser.add_argument('--version',
                         default=None,
                         type=str,
                         metavar='V',
                         help='version or id of the net')
    parser.add_argument('--resume-from-checkpoint',
                         default=None,
                         type=str,
                         metavar='RFC',
                         help='path to checkpoint')
    
    args = parser.parse_args()

    main(args)


if __name__ == '__main__':
    run_cli()