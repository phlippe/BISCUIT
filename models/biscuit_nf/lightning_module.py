import torch
from pytorch_lightning.callbacks import LearningRateMonitor

import sys
sys.path.append('../')
from models.shared import get_act_fn, ImageLogCallback
from models.ae import Autoencoder
from models.biscuit_vae import BISCUITVAE
from models.shared import AutoregNormalizingFlow
from models.shared import InteractionVisualizationCallback, PermutationCorrelationMetricsLogCallback


class BISCUITNF(BISCUITVAE):
    """ 
    The main module implementing BISCUIT-NF.
    It is a subclass of BISCUIT-VAE to inherit several functionality.
    """

    def __init__(self, *args,
                        autoencoder_checkpoint=None,
                        num_flows=4,
                        hidden_per_var=16,
                        num_samples=8,
                        flow_act_fn='silu',
                        noise_level=-1,
                        **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs - see BISCUIT-VAE for the full list
        autoencoder_checkpoint : str
                                 Path to the checkpoint of the autoencoder
                                 which should be used for training the flow
                                 on.
        num_flows : int
                    Number of flow layers to use
        hidden_per_var : int
                         Hidden dimensionality per latent variable to use
                         in the autoregressive networks.
        num_samples : int
                      Number of samples to take from an input encoding
                      during training. Larger sample sizes give smoother
                      gradients.
        flow_act_fn : str
                      Activation function to use in the networks of the flow
        noise_level : float
                      Standard deviation of the added noise to the encodings.
                      If smaller than zero, the std of the autoencoder is used.
        """
        kwargs['no_encoder_decoder'] = True  # We do not need any additional en- or decoder
        super().__init__(*args, **kwargs)
        # Initialize the flow
        self.flow = AutoregNormalizingFlow(self.hparams.num_latents, 
                                           self.hparams.num_flows,
                                           act_fn=get_act_fn(self.hparams.flow_act_fn),
                                           hidden_per_var=self.hparams.hidden_per_var)
        # Setup autoencoder
        if self.hparams.autoencoder_checkpoint is not None:
            self.autoencoder = Autoencoder.load_from_checkpoint(self.hparams.autoencoder_checkpoint)
            for p in self.autoencoder.parameters():
                p.requires_grad_(False)
            
            if self.hparams.noise_level < 0.0:
                self.hparams.noise_level = self.autoencoder.hparams.noise_level
        else:
            self.autoencoder = None
            self.hparams.noise_level = 0.0

    def encode(self, x, random=True):
        # Map input to disentangled latents, e.g. for correlation metrics
        if random:
            x = x + torch.randn_like(x) * self.hparams.noise_level
        z, _ = self.flow(x)
        return z

    def _get_loss(self, batch, mode='train'):
        """ Main training method for calculating the loss """
        if len(batch) == 2:
            x_enc, action = batch
        else:
            x_enc, _, action = batch
        with torch.no_grad():
            # Expand encodings over samples and add noise to 'sample' from the autoencoder
            # latent distribution
            x_enc = x_enc[...,None,:].expand(-1, -1, self.hparams.num_samples, -1)
            batch_size, seq_len, num_samples, num_latents = x_enc.shape
            x_sample = x_enc + torch.randn_like(x_enc) * self.hparams.noise_level
            x_sample = x_sample.flatten(0, 2)
        # Execute the flow
        z_sample, ldj = self.flow(x_sample)
        z_sample = z_sample.unflatten(0, (batch_size, seq_len, num_samples))
        ldj = ldj.reshape(batch_size, seq_len, num_samples)
        # Calculate the negative log likelihood of the transition prior
        nll = self.prior_t1.sample_based_nll(z_t=z_sample[:,:-1].flatten(0, 1),
                                             z_t1=z_sample[:,1:].flatten(0, 1),
                                             action=action.flatten(0, 1))
        # Add LDJ and prior NLL for full loss
        ldj = ldj[:,1:].flatten(0, 1).mean(dim=-1)  # Taking the mean over samples
        loss = nll + ldj
        loss = (loss * (seq_len - 1)).mean()

        # Logging
        self.log(f'{mode}_nll', nll.mean())
        self.log(f'{mode}_ldj', ldj.mean())

        return loss

    @staticmethod
    def get_callbacks(exmp_inputs=None, dataset=None, cluster=False, correlation_dataset=False, correlation_test_dataset=None, action_data_loader=None, **kwargs):
        img_callback = ImageLogCallback([None, None], dataset, every_n_epochs=10, cluster=cluster)
        # Create learning rate callback
        lr_callback = LearningRateMonitor('step')
        callbacks = [lr_callback, img_callback]
        corr_callback = PermutationCorrelationMetricsLogCallback(correlation_dataset, 
                                                                 cluster=cluster, 
                                                                 test_dataset=correlation_test_dataset)
        callbacks.append(corr_callback)
        if action_data_loader is not None:
            actionvq_callback = InteractionVisualizationCallback(action_data_loader=action_data_loader)
            callbacks.append(actionvq_callback)
        return callbacks