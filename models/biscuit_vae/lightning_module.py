from typing import Any, Dict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from collections import defaultdict

import sys
sys.path.append('../')
from models.shared import CosineWarmupScheduler, get_act_fn, Encoder, Decoder, SimpleEncoder, SimpleDecoder, VAESplit, ImageLogCallback, PermutationCorrelationMetricsLogCallback
from models.shared import AutoregNormalizingFlow, gaussian_log_prob
from models.shared import create_interaction_prior, InteractionVisualizationCallback


class BISCUITVAE(pl.LightningModule):
    """ The main module implementing BISCUIT-VAE """

    def __init__(self, c_hid, num_latents, lr, action_size=-1,
                       warmup=100, max_iters=100000,
                       img_width=64,
                       c_in=3,
                       decoder_num_blocks=1,
                       act_fn='silu',
                       no_encoder_decoder=False,
                       linear_encoder_decoder=False,
                       use_flow_prior=True,
                       decoder_latents=-1,
                       prior_action_add_prev_state=False,
                       **kwargs):
        """
        Parameters
        ----------
        c_hid : int
                Hidden dimensionality to use in the network
        num_latents : int
                      Number of latent variables in the VAE
        lr : float
             Learning rate to use for training
        action_size : int
                      Size of the action space.
        warmup : int
                 Number of learning rate warmup steps
        max_iters : int
                    Number of max. training iterations. Needed for 
                    cosine annealing of the learning rate.
        img_width : int
                    Width of the input image (assumed to be equal to height)
        c_in : int
               Number of input channels (3 for RGB)
        decoder_num_blocks : int
                             Number of residual blocks to use per dimension in the decoder.
        act_fn : str
                 Activation function to use in the encoder and decoder network.
        no_encoder_decoder : bool
                             If True, no encoder or decoder are initialized. Used for BISCUIT-NF
        linear_encoder_decoder : bool
                                 If True, the encoder and decoder are simple MLPs.
        use_flow_prior : bool
                         If True, use a normalizing flow in the prior.
        decoder_latents : int
                          Number of latent variables to use as input to the decoder. If -1, equal to encoder.
                          Can be used when additional variables are added to the decoder, e.g. for the action.
        prior_action_add_prev_state : bool
                                      If True, we consider the interaction variables to potentially depend on
                                      the previous state and add it to the MLPs.
        """
        super().__init__()
        self.save_hyperparameters()
        self.hparams.num_latents = num_latents
        act_fn_func = get_act_fn(self.hparams.act_fn)
        if self.hparams.decoder_latents < 0:
            self.hparams.decoder_latents = self.hparams.num_latents

        # Encoder-Decoder init
        if self.hparams.no_encoder_decoder:
            self.encoder, self.decoder = nn.Identity(), nn.Identity()
        elif self.hparams.linear_encoder_decoder:
            nn_hid = max(512, 2 * self.hparams.c_hid)
            self.encoder = nn.Sequential(
                nn.Linear(self.hparams.c_in, nn_hid),
                nn.SiLU(),
                nn.Linear(nn_hid, nn_hid),
                nn.SiLU(),
                nn.Linear(nn_hid, 2 * self.hparams.num_latents),
                VAESplit(self.hparams.num_latents)
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.hparams.num_latents, nn_hid),
                nn.SiLU(),
                nn.Linear(nn_hid, nn_hid),
                nn.SiLU(),
                nn.Linear(nn_hid, self.hparams.c_in)
            )
        else:
            if self.hparams.img_width == 32:
                self.encoder = SimpleEncoder(c_in=self.hparams.c_in,
                                             c_hid=self.hparams.c_hid,
                                             num_latents=self.hparams.num_latents)
                self.decoder = SimpleDecoder(c_in=self.hparams.c_in,
                                             c_hid=self.hparams.c_hid,
                                             num_latents=self.hparams.num_latents)
            else:
                self.encoder = Encoder(num_latents=self.hparams.num_latents,
                                          c_hid=self.hparams.c_hid,
                                          c_in=self.hparams.c_in,
                                          width=self.hparams.img_width,
                                          act_fn=act_fn_func,
                                          variational=True)
                self.decoder = Decoder(num_latents=self.hparams.decoder_latents,
                                          c_hid=self.hparams.c_hid,
                                          c_out=self.hparams.c_in,
                                          width=self.hparams.img_width,
                                          num_blocks=self.hparams.decoder_num_blocks,
                                          act_fn=act_fn_func)
        # Transition prior
        self.prior_t1 = create_interaction_prior(num_latents=self.hparams.num_latents,
                                                 c_hid=self.hparams.c_hid,
                                                 add_prev_state=self.hparams.prior_action_add_prev_state,
                                                 action_size=self.hparams.action_size,
                                                 img_width=self.hparams.img_width,
                                                 extra_args=kwargs)

        if self.hparams.use_flow_prior:
            self.flow = AutoregNormalizingFlow(self.hparams.num_latents,
                                               num_flows=4,
                                               act_fn=nn.SiLU,
                                               hidden_per_var=16)
        # Logging
        self.all_val_dists = defaultdict(list)
        self.output_to_input = None
        self.register_buffer('last_target_assignment', torch.zeros(self.hparams.num_latents, 1))

    def forward(self, x):
        # Full encoding and decoding of samples
        z_mean, z_logstd = self.encoder(x)
        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        x_rec = self.decoder(z_sample)
        return x_rec, z_sample, z_mean, z_logstd

    def encode(self, x, random=True):
        # Map input to encoding, e.g. for correlation metrics
        z_mean, z_logstd = self.encoder(x)
        if random:
            z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        else:
            z_sample = z_mean
        if self.hparams.use_flow_prior:
            z_sample, _ = self.flow(z_sample)
        return z_sample

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.0)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def _get_loss(self, batch, mode='train'):
        """ Main training method for calculating the loss """
        if len(batch) == 2:
            imgs, action = batch
            labels = imgs
        else:
            imgs, labels, action = batch
        # En- and decode every element of the sequence, except first element no decoding
        z_mean, z_logstd = self.encoder(imgs.flatten(0, 1))
        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        decoder_inp = z_sample.unflatten(0, imgs.shape[:2])[:,1:].flatten(0, 1)
        if self.hparams.decoder_latents != self.hparams.num_latents:
            decoder_inp = torch.cat([decoder_inp, action.flatten(0, 1)], dim=-1)
        x_rec = self.decoder(decoder_inp)
        z_sample, z_mean, z_logstd, x_rec = [t.unflatten(0, (imgs.shape[0], -1)) for t in [z_sample, z_mean, z_logstd, x_rec]]

        if self.hparams.use_flow_prior:
            num_samples = 4
            z_sample = z_mean[:,:,None] + torch.randn_like(z_mean[:,:,None].expand(-1, -1, num_samples, -1)) * z_logstd.exp()[:,:,None]
            z_sample = torch.cat([z_mean[:,0:1,None].expand(-1, -1, z_sample.shape[2], -1),
                                  z_sample[:,1:]], dim=1)
            init_nll = -gaussian_log_prob(z_mean[:,:,None], z_logstd[:,:,None], z_sample).mean(dim=2).sum(dim=-1)
            z_sample, ldj = self.flow(z_sample.flatten(0, -2))
            z_sample = z_sample.unflatten(0, (imgs.shape[0], -1, num_samples))
            ldj = ldj.unflatten(0, (imgs.shape[0], -1, num_samples)).mean(dim=2)
            out_nll = self.prior_t1.sample_based_nll(z_t1=z_sample[:,1:].flatten(0, 1), 
                                                     action=action.flatten(0, 1), 
                                                     z_t=z_sample[:,:-1].flatten(0, 1))
            if isinstance(out_nll, tuple):
                out_nll, log_dict = out_nll
                for key in log_dict:
                    val = log_dict[key]
                    if isinstance(val, torch.Tensor):
                        val = val.mean()
                    self.log(f'{mode}_prior_{key}', val)
            out_nll = out_nll.unflatten(0, (imgs.shape[0], -1))
            p_z = out_nll 
            p_z_x = init_nll - ldj
            kld = -(p_z_x[:,1:] - p_z)
            kld_t1_all = kld.unflatten(0, (imgs.shape[0], -1)).sum(dim=1)
        else:
            # Calculate KL divergence between every pair of frames
            kld_t1_all = self.prior_t1.kl_divergence(z_t=z_mean[:,:-1].flatten(0, 1), 
                                                     action=action.flatten(0, 1), 
                                                     z_t1_mean=z_mean[:,1:].flatten(0, 1), 
                                                     z_t1_logstd=z_logstd[:,1:].flatten(0, 1), 
                                                     z_t1_sample=z_sample[:,1:].flatten(0, 1))
            if isinstance(kld_t1_all, tuple):
                kld_t1_all, log_dict = kld_t1_all
                for key in log_dict:
                    val = log_dict[key]
                    if isinstance(val, torch.Tensor):
                        val = val.mean()
                    self.log(f'{mode}_prior_{key}', val)
            kld_t1_all = kld_t1_all.unflatten(0, (imgs.shape[0], -1)).sum(dim=1)
        
        # Calculate reconstruction loss
        if isinstance(self.decoder, nn.Identity):
            rec_loss = z_mean.new_zeros(imgs.shape[0], imgs.shape[1])
        else:
            rec_loss = F.mse_loss(x_rec, labels[:,1:], reduction='none')
            rec_loss = rec_loss.sum(dim=[i for i in range(2, len(rec_loss.shape))])
        # Combine to full loss
        loss = (kld_t1_all + rec_loss.sum(dim=1)).mean()
        loss = loss / (imgs.shape[1] - 1)

        # Logging
        self.log(f'{mode}_kld_t1', kld_t1_all.mean() / (imgs.shape[1]-1))
        self.log(f'{mode}_rec_loss_t1', rec_loss.mean())

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='train')
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation/Testing with correlation matrices done via callbacks
        loss = self._get_loss(batch, mode='val')
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='test')
        self.log('test_loss', loss)
        return loss
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Need to do it explicitly since the size of the last_target_assignment might have changed
        if 'last_target_assignment' in checkpoint['state_dict']:
            self.last_target_assignment.data = checkpoint['state_dict']['last_target_assignment']

    @staticmethod
    def get_callbacks(exmp_inputs=None, dataset=None, cluster=False, correlation_dataset=None, correlation_test_dataset=None, action_data_loader=None, **kwargs):
        callbacks = [LearningRateMonitor('step')]
        if exmp_inputs is not None:
            img_callback = ImageLogCallback(exmp_inputs, dataset, every_n_epochs=10 if not cluster else 50, cluster=cluster)
            callbacks.append(img_callback)
        if correlation_dataset is not None:
            corr_callback = PermutationCorrelationMetricsLogCallback(correlation_dataset, cluster=cluster, test_dataset=correlation_test_dataset)
            callbacks.append(corr_callback)
        if action_data_loader is not None:
            actionvq_callback = InteractionVisualizationCallback(action_data_loader=action_data_loader)
            callbacks.append(actionvq_callback)
        return callbacks