import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from experiments.exp_ssl_vqvae import Exp_SSL_VQVAE

from preprocessing.preprocess_ucr import UCRDatasetImporter
from preprocessing.data_pipeline import build_data_pipeline
from utils import load_yaml_param_settings
from utils import save_model
import torch

torch.set_float32_matmul_precision('medium')

def train_SSL_VQVAE(
        config: dict,
        aug_train_data_loader: DataLoader,
        train_data_loader: DataLoader,
        test_data_loader: DataLoader,
        do_validate: bool,
        wandb_project_case_idx: str = '',
        wandb_run_name=''
        ):
    """
    Trainer for the SSL-VQVAE model. 
    ---
    Args:
        config (dict): config dictionary
        SSL_method (BarlowTwins or VICReg): SSL method to use
        aug_train_data_loader (DataLoader): train data loader with augmentations
        train_data_loader (DataLoader): train data loader without augmentations - for representation learning
        test_data_loader (DataLoader): test data loader - for representation learning / validation
        do_validate (bool): whether to validate during training
        wandb_project_case_idx (str): additional string to identify the run
        wandb_project_name (str): name of the wandb project
        wandb_run_name (str): name of the wandb run
   
    """
    #Wandb: Initialize a new run
    project_name =  'SSL_VQVAE-stage1'

    if wandb_project_case_idx != '':
        project_name += f'-{wandb_project_case_idx}'
    

    input_length = train_data_loader.dataset.X.shape[-1]

    train_model = Exp_SSL_VQVAE(input_length, 
                            non_aug_test_data_loader=test_data_loader,
                            non_aug_train_data_loader=train_data_loader, 
                            config=config, n_train_samples=len(train_data_loader.dataset))

    wandb_logger = WandbLogger(project=project_name, 
                               dir=f"RepL/{config['dataset']['dataset_name']}/BarlowTwinsVQVAE",
                               name=wandb_run_name, config=config)
    
    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         max_epochs=config['trainer_params']['max_epochs']['stage1_ssl'],
                         devices=config['trainer_params']['gpus'],
                         accelerator='gpu',
                         check_val_every_n_epoch=20)
    
    
    trainer.fit(train_model,
                train_dataloaders=aug_train_data_loader,
                val_dataloaders=test_data_loader if do_validate else None
                )
    
    # additional log
    n_trainable_params = sum(p.numel() for p in train_model.parameters() if p.requires_grad)
    wandb.log({'n_trainable_params:': n_trainable_params})

    # test
    wandb.finish()

    #print('saving the models...')
    
    #gamma = config['barlow_twins']['gamma']
    SSL_weigting = config['SSL']['weighting']
    SSL_method = config['SSL']['method_choice']
    save_model({f'{SSL_method}_{SSL_weigting}_encoder': train_model.encoder,
                f'{SSL_method}_{SSL_weigting}_decoder': train_model.decoder,
                f'{SSL_method}_{SSL_weigting}_vq_model': train_model.vq_model,
                }, id=config['dataset']['dataset_name'])
    
    
if __name__ == "__main__":
    config_dir = 'configs/config.yaml' #dir to config file

    config = load_yaml_param_settings(config_dir)

    # data pipeline
    dataset_importer = UCRDatasetImporter(**config['dataset'])
    batch_size = config['dataset']['batch_sizes']['stage1']
    train_data_loader_non_aug, test_data_loader= [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

    train_data_loader_aug = build_data_pipeline(batch_size, dataset_importer, config, augment=True, kind="train")

    #SSL_method = BarlowTwins(config)
    train_SSL_VQVAE(config, aug_train_data_loader = train_data_loader_aug,
                    train_data_loader=train_data_loader_non_aug,
                    test_data_loader=test_data_loader, do_validate=True)