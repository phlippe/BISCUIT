import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import numpy as np
from copy import deepcopy

import sys
sys.path.append('../../')
from models.shared import CosineWarmupScheduler, Encoder, Decoder, visualize_ae_reconstruction, SimpleEncoder, SimpleDecoder


class Autoencoder(pl.LightningModule):
    """ Simple Autoencoder network """

    def __init__(self, num_latents,
                       c_in=3,
                       c_hid=64,
                       lr=1e-3,
                       warmup=500, 
                       max_iters=100000,
                       img_width=64,
                       noise_level=0.05,
                       regularizer_weight=1e-4,
                       action_size=-1,
                       mi_reg_weight=0.0,
                       **kwargs):
        """
        Parameters
        ----------
        num_latents : int
                      Number of latents in the bottleneck.
        c_in : int
               Number of input channels (3 for RGB)
        c_hid : int
                Hidden dimensionality to use in the network
        lr : float
             Learning rate to use for training.
        warmup : int
                 Number of learning rate warmup steps
        max_iters : int
                    Number of max. training iterations. Needed for 
                    cosine annealing of the learning rate.
        img_width : int
                    Width of the input image (assumed to be equal to height)
        noise_level : float
                      Standard deviation of the added noise to the latents.
        """
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.img_width == 32:
            EncoderClass = SimpleEncoder
            DecoderClass = SimpleDecoder
        else:
            EncoderClass = Encoder
            DecoderClass = Decoder
        self.encoder = EncoderClass(num_latents=self.hparams.num_latents,
                                    c_hid=self.hparams.c_hid,
                                    c_in=self.hparams.c_in,
                                    width=self.hparams.img_width,
                                    act_fn=nn.SiLU,
                                    residual=True,
                                    num_blocks=2,
                                    variational=False)
        self.decoder = DecoderClass(num_latents=self.hparams.num_latents + max(0, self.hparams.action_size),
                                    c_hid=self.hparams.c_hid,
                                    c_out=self.hparams.c_in,
                                    width=self.hparams.img_width,
                                    num_blocks=2,
                                    act_fn=nn.SiLU)
        if self.hparams.action_size > 0 and self.hparams.mi_reg_weight > 0:
            self.action_mi_estimator = nn.Sequential(
                nn.Linear(self.hparams.action_size + self.hparams.num_latents, self.hparams.c_hid),
                nn.SiLU(),
                nn.Linear(self.hparams.c_hid, self.hparams.c_hid),
                nn.SiLU(),
                nn.Linear(self.hparams.c_hid, 1)
            )
            self.action_mi_estimator_copy = deepcopy(self.action_mi_estimator)
            for p in self.action_mi_estimator_copy.parameters():
                p.requires_grad = False
        else:
            self.action_mi_estimator = None
            self.action_mi_estimator_copy = None

    def forward(self, x, actions=None, return_z=False):
        z = self.encoder(x)
        # Adding noise to latent encodings preventing potential latent space collapse
        z_samp = z + torch.randn_like(z) * self.hparams.noise_level
        if actions is not None and self.hparams.action_size > 0:
            z_samp = torch.cat([z_samp, actions], dim=-1)
        x_rec = self.decoder(z_samp)
        if return_z:
            return x_rec, z
        else:
            return x_rec 

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.0)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def _get_loss(self, batch, mode='train'):
        # Trained by standard MSE loss
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            imgs, actions = batch
        else:
            imgs, actions = batch, None
        x_rec, z = self.forward(imgs, actions=actions, return_z=True)
        loss_rec = F.mse_loss(x_rec, imgs)
        loss_reg = (z ** 2).mean()
        self.log(f'{mode}_loss_rec', loss_rec)
        self.log(f'{mode}_loss_reg', loss_reg)
        self.log(f'{mode}_loss_reg_weighted', loss_reg * self.hparams.regularizer_weight)
        with torch.no_grad():
            self.log(f'{mode}_loss_rec_mse', F.mse_loss(x_rec, imgs))
            self.log(f'{mode}_loss_rec_abs', torch.abs(x_rec - imgs).mean())
        
            noncompressed_rec = F.mse_loss(x_rec, imgs, reduction='none')
            self.log(f'{mode}_loss_rec_max', noncompressed_rec.max())
            self.log(f'{mode}_loss_rec_smaller_01', (noncompressed_rec < 0.1).float().mean())
            self.log(f'{mode}_loss_rec_smaller_001', (noncompressed_rec < 0.01).float().mean())
            self.log(f'{mode}_loss_rec_smaller_0001', (noncompressed_rec < 0.001).float().mean())
        loss = loss_rec + loss_reg * self.hparams.regularizer_weight

        if self.action_mi_estimator is not None and mode == 'train':
            # Mutual information regularization
            loss_mi_reg_model, loss_mi_reg_latents = self._get_mi_reg_loss(z, actions)
            loss = loss + (loss_mi_reg_model + loss_mi_reg_latents) * self.hparams.mi_reg_weight
            self.log(f'{mode}_loss_mi_reg_model', loss_mi_reg_model)
            self.log(f'{mode}_loss_mi_reg_latents', loss_mi_reg_latents)
            self.log(f'{mode}_loss_mi_reg_latents_weighted', loss_mi_reg_latents * self.hparams.mi_reg_weight)
        return loss

    def _get_mi_reg_loss(self, z, actions):
        # Mutual information regularization
        z = z + torch.randn_like(z) * self.hparams.noise_level
        true_inp = torch.cat([z, actions], dim=-1)
        perm = torch.randperm(z.shape[0], device=z.device)
        fake_inp = torch.cat([z[perm], actions], dim=-1)
        inp = torch.stack([true_inp, fake_inp], dim=1).flatten(0, 1)
        model_out = self.action_mi_estimator(inp.detach()).reshape(z.shape[0], 2)
        model_loss = -F.log_softmax(model_out, dim=1)[:,0].mean()
        model_acc = (model_out[:,0] > model_out[:,1]).float().mean()
        self.log('train_mi_reg_model_acc', model_acc)

        for p1, p2 in zip(self.action_mi_estimator.parameters(), self.action_mi_estimator_copy.parameters()):
            p2.data.copy_(p1.data)
        latents_out = self.action_mi_estimator_copy(inp).reshape(z.shape[0], 2)
        latents_loss = -F.log_softmax(latents_out, dim=1).mean()

        return model_loss, latents_loss

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='train')
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='val')
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='val')
        self.log('test_loss', loss)

    @staticmethod
    def get_callbacks(exmp_inputs=None, cluster=False, **kwargs):
        img_callback = AELogCallback(exmp_inputs, every_n_epochs=10)
        # Create learning rate callback
        lr_callback = LearningRateMonitor('step')
        return [lr_callback, img_callback]


class AELogCallback(pl.Callback):
    """ Callback for visualizing predictions """

    def __init__(self, exmp_inputs, every_n_epochs=5, prefix=''):
        super().__init__()
        if isinstance(exmp_inputs, (tuple, list)):
            self.imgs, self.actions = exmp_inputs
        else:
            self.imgs, self.actions = exmp_inputs, None
        self.every_n_epochs = every_n_epochs
        self.prefix = prefix

    def on_train_epoch_end(self, trainer, pl_module):
        def log_fig(tag, fig):
            trainer.logger.experiment.add_image(f'{self.prefix}{tag}', fig, global_step=trainer.global_step, dataformats='HWC')

        if self.imgs is not None and (trainer.current_epoch+1) % self.every_n_epochs == 0:
            images = self.imgs.to(trainer.model.device)
            trainer.model.eval()
            rand_idxs = np.random.permutation(images.shape[0])
            if self.actions is None or pl_module.hparams.action_size <= 0:
                actions = None
            else:
                actions = self.actions.to(trainer.model.device)
            log_fig(f'reconstruction_seq', visualize_ae_reconstruction(trainer.model, images[:8], 
                                                                       actions[:8] if actions is not None else None))
            log_fig(f'reconstruction_rand', visualize_ae_reconstruction(trainer.model, images[rand_idxs[:8]], 
                                                                        actions[rand_idxs[:8]] if actions is not None else None))
            trainer.model.train()