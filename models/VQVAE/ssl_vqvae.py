import numpy as np
import matplotlib.pyplot as plt

from models.VQVAE.encoder_decoder import VQVAEEncoder, VQVAEDecoder
from models.VQVAE.vq import VectorQuantize
from models.SSL.vicreg import VICReg
from models.SSL.barlowtwins import BarlowTwins

from utils import (compute_downsample_rate,
                        encode_data,
                        time_to_timefreq,
                        timefreq_to_time,
                        quantize,
                        )

from models.base_model import BaseModel, detach_the_unnecessary

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from experiments.representation_tests import test_model_representations

class SSL_VQVAE(BaseModel):
    """
    VQVAE with a two branch encoder structure. Incorporates an additional SSL objective for the encoder.
    ---
    input_length: length of the input signal
    SSL_method: SSL method to use. Either Barlow Twins or VICReg is supported at this moment.
    non_aug_test_data_loader: test data loader without augmentation. For representation testing.
    non_aug_train_data_loader: train data loader without augmentation. For representation testing.
    config: config dict
    n_train_samples: number of training samples
    """
    def __init__(self,
                input_length,
                SSL_method,
                non_aug_test_data_loader,
                non_aug_train_data_loader,
                config: dict,
                n_train_samples: int
                ):
        super().__init__()

        self.config = config
        self.T_max = config['trainer_params']['max_epochs']['vqvae'] * (np.ceil(n_train_samples / config['dataset']['batch_sizes']['vqvae']) + 1)
        
        self.n_fft = config['VQVAE']['n_fft']
        dim = config['encoder']['dim']
        in_channels = config['dataset']['in_channels']

        downsampled_width = config['encoder']['downsampled_width']
        downsampled_rate = compute_downsample_rate(input_length, self.n_fft, downsampled_width)

        #encoder
        self.encoder = VQVAEEncoder(dim, 2*in_channels, downsampled_rate, config['encoder']['n_resnet_blocks'], config['encoder']['dropout_rate'])
        
        #vector quantiser
        self.vq_model = VectorQuantize(dim, config['VQVAE']['codebook']['size'], **config['VQVAE'])

        #decoder
        self.decoder = VQVAEDecoder(dim, 2 * in_channels, downsampled_rate, config['decoder']['n_resnet_blocks'], config['decoder']['dropout_rate'])

        #SSL objective
        self.SSL_method = SSL_method
        self.SSL_loss_weighting = config['TBE_VQVAE']['SSL_loss_weight']

        #save these for representation tests during training
        self.train_data_loader = non_aug_train_data_loader
        self.test_data_loader = non_aug_test_data_loader

    def forward(self, batch, training=True):
        """
        x1 --> u1 --> z1 --> z_q --> uhat --> xhat
                         |
                      SSL_Loss
                         |
        x2 --> E(u2) --> z2 
        
        """      

        if training: (x_view1, x_view2), y = batch
        else: 
            x_view1, y = batch
            use_view1 = True
    

        recons_loss = {'time': 0., 'timefreq': 0., 'perceptual': 0.}
        vq_loss = 0.
        perplexity = 0.

        C = x_view1.shape[1]

        #--- Encode view 1 ---
        u1 = time_to_timefreq(x_view1, self.n_fft, C) #STFT

        if not self.decoder.is_upsample_size_updated:
            self.decoder.register_upsample_size(torch.IntTensor(np.array(u1.shape[2:])))

        z1 = self.encoder(u1) #Encode

        if training:
            # --- Encode view 2 ---
            u2 = time_to_timefreq(x_view2, self.n_fft, C) #STFT
            
            z2 = self.encoder(u2) #Encode
            # --- SSL loss ---
            SSL_loss = self.SSL_method(z1, z2) #calculate the SSL loss given two encodings:
            
            use_view1 = np.random.rand() < 0.5 #randomly choose which view to use for reconstruction
        else:
            SSL_loss = torch.tensor(0.0) #no SSL loss if validation step

        # --- Vector Quantization ---
        z_q, indices, vq_loss, perplexity = quantize(z1, self.vq_model) if use_view1 else quantize(z2, self.vq_model) #Vector quantization

        #--- Reconstruction ---:
        uhat = self.decoder(z_q) #Decode
        xhat = timefreq_to_time(uhat, self.n_fft, C, original_length=x_view1.size(2) if use_view1 else x_view2.size(2)) #Inverse STFT
        target_x, target_u = (x_view1, u1) if use_view1 else (x_view2, u2)

        #--- VQVAE loss ---
        recons_loss['time'] = F.mse_loss(target_x, xhat)
        recons_loss['timefreq'] = F.mse_loss(target_u, uhat)

        # plot `x` and `xhat`
        r = np.random.uniform(0, 1)

        if r < 0.01 and training:
            b = np.random.randint(0, target_x.shape[0])
            c = np.random.randint(0, target_x.shape[1])
            fig, ax = plt.subplots()
            plt.suptitle(f'ep_{self.current_epoch}')
            
            label1 = "view1-target" if use_view1 else "view1"
            label2 = "view2-target" if not use_view1 else "view2"
            label3 = "view1 - reconstruction" if use_view1 else "view2 - reconstruction"
            alpha1 = 1 if use_view1 else 0.2
            alpha2 = 1 if not use_view1 else 0.2

            ax.plot(x_view1[b, c].cpu(), label=f"{label1}", c="gray", alpha=alpha1)
            ax.plot(x_view2[b, c].cpu(), label=f"{label2}", c="gray", alpha=alpha2)
            ax.plot(xhat[b,c].detach().cpu(), label=f"{label3}")
            ax.set_title('x')
            ax.set_ylim(-5, 5)
            fig.legend()
            wandb.log({"Reconstruction": wandb.Image(plt)})
            plt.close()
            
        return recons_loss, vq_loss, perplexity, SSL_loss
    
        
    def training_step(self, batch, batch_idx):
        x = batch
        #forward:
        recons_loss, vq_loss, perplexity, SSL_loss = self.forward(x)

        #calculate vqvae loss:
        vqvae_loss = recons_loss['time'] + recons_loss['timefreq'] + vq_loss['loss'] + recons_loss['perceptual'] 

        #total loss:
        loss = vqvae_loss + SSL_loss * self.SSL_loss_weighting

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {'loss': loss,
                     'recons_loss.time': recons_loss['time'],

                     'recons_loss.timefreq': recons_loss['timefreq'],

                     'commit_loss': vq_loss['commit_loss'],
                     #'commit_loss': vq_loss, #?
                     
                     'perplexity': perplexity,

                     'perceptual': recons_loss['perceptual'],

                     self.SSL_method.name + 'loss': SSL_loss * self.SSL_loss_weighting,
                     }
        
        wandb.log(loss_hist)

        detach_the_unnecessary(loss_hist)
        return loss_hist
    
    def validation_step(self, batch, batch_idx):
        x = batch
        recons_loss, vq_loss, perplexity, _ = self.forward(x, training=False)

        #only VQVAE loss
        loss = recons_loss['time'] + recons_loss['timefreq'] + vq_loss['loss'] + recons_loss['perceptual']

        # log
        val_loss_hist = {'validation_loss': loss,
                     'validation_recons_loss.time': recons_loss['time'],

                     'validation_recons_loss.timefreq': recons_loss['timefreq'],

                     'validation_commit_loss': vq_loss['commit_loss'],
                     #'validation_commit_loss': vq_loss, #?
                     
                     'validation_perplexity': perplexity,

                     'validation_perceptual': recons_loss['perceptual'],
                     }
        
        detach_the_unnecessary(val_loss_hist)
        wandb.log(val_loss_hist)

        return val_loss_hist


    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params': self.encoder.parameters(), 'lr': self.config['model_params']['LR']},
                                 {'params': self.decoder.parameters(), 'lr': self.config['model_params']['LR']},
                                 {'params': self.vq_model.parameters(), 'lr': self.config['model_params']['LR']},
                                 ],
                                weight_decay=self.config['model_params']['weight_decay'])
        
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, self.T_max)}


    def test_step(self, batch, batch_idx):
        x = batch
        recons_loss, vq_loss, perplexity, _ = self.forward(x)

        loss = recons_loss['time'] + recons_loss['timefreq'] + vq_loss['loss'] + recons_loss['perceptual']
        
        # log
        loss_hist = {'loss': loss,
                     'recons_loss.time': recons_loss['time'],

                     'recons_loss.timefreq': recons_loss['timefreq'],

                     'commit_loss': vq_loss['commit_loss'],
                     #'commit_loss': vq_loss, #?
                     
                     'perplexity': perplexity,

                     'perceptual': recons_loss['perceptual'],
                     }
        
        detach_the_unnecessary(loss_hist)
        return loss_hist
    
    # ---- Representation testing ------
    def on_train_epoch_end(self):
        if self.config['representations']['test_stage1']:
            tested = False
            if self.current_epoch % 300 == 0 and self.current_epoch != 0:
                wandb.log(test_model_representations(
                    encode_data(self.train_data_loader, self.encoder, self.config['VQVAE']['n_fft'], self.vq_model),
                    encode_data(self.test_data_loader, self.encoder, self.config['VQVAE']['n_fft'], self.vq_model))
                )
                tested = True

            if self.current_epoch == self.config['trainer_params']['max_epochs']['barlowvqvae']-1 and tested == False:
                wandb.log(test_model_representations(
                    encode_data(self.train_data_loader, self.encoder, self.config['VQVAE']['n_fft'], self.vq_model),
                    encode_data(self.test_data_loader, self.encoder, self.config['VQVAE']['n_fft'], self.vq_model))
                )

    def on_train_epoch_start(self):
        if self.current_epoch == 0 and self.config['representations']['test_stage1']:
            wandb.log(test_model_representations(
                encode_data(self.train_data_loader, self.encoder, self.config['VQVAE']['n_fft'], self.vq_model),
                encode_data(self.test_data_loader, self.encoder, self.config['VQVAE']['n_fft'], self.vq_model))
            )
