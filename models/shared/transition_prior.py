import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append('../../')
from models.shared.utils import kl_divergence, gaussian_log_prob
from models.shared.modules import MultivarLinear, MultivarSequential, LinearScheduler, SinusoidalEncoding


def create_interaction_prior(*args, extra_args=None, **kwargs):
    if extra_args is None:
        extra_args = dict()
    for key in extra_args:
        if key not in kwargs:
            kwargs[key] = extra_args[key]
    ModelClass = InteractionTransitionPrior
    return ModelClass(*args, **kwargs)


class InteractionTransitionPrior(nn.Module):
    """
    Transition prior of BISCUIT
    """

    def __init__(self, num_latents, c_hid,
                 action_size=-1,
                 logit_reg_factor=0.004,
                 temp_schedule=[(1.0, 0), (2.5, 50000), (5.0, 100000)],
                 add_prev_state=False,
                 img_width=32,
                 **kwargs):
        """
        Parameters
        ----------
        num_latents : int
                      Number of latents in the model
        c_hid : int
                Hidden dimensionality of the networks
        action_size : int, default -1
                      Dimensionality of the action/regime variable
        logit_reg_factor : float, default 0.004
                           Regularization factor for the logits of the action variable
        temp_schedule : list of tuples, default [(1.0, 0), (2.5, 50000), (5.0, 100000)]
                        Schedule for the temperature factor of the Tanh activation function
                        of the interaction variable. The first value of each tuple is the
                        temperature factor, the second value is the number of training steps
                        after which the temperature factor is reached. We interpolate linearly
                        between the steps.
        add_prev_state : bool, default False
                         Whether to add the previous state to the interaction variable prediction.
        img_width : int, default 32
                    Width of the images in the dataset. Only used if action_size == 2.
        """
        super().__init__()
        self.num_latents = num_latents
        self.c_hid = c_hid
        self.action_size = action_size
        self.logit_reg_factor = logit_reg_factor
        self.temp_scheduler = LinearScheduler(temp_schedule)
        self.step = 0
        self.add_prev_state = add_prev_state
        self.last_batch_prev_state = None
        self.variable_alignments = None

        # Create prior network. The prior network is a simple MLP with
        # the input of the previous latent variable z_i^t, all variables z^t,
        # and the predicted interaction variable I_i^t separated.
        self.inp_layer = MultivarLinear(1,
                                        c_hid, [self.num_latents])
        self.action_layer = MultivarLinear(1,
                                           self.c_hid,
                                          [self.num_latents])
        self.action_layer.weight.data.fill_(0.0)
        self.context_layer = MultivarSequential(
            MultivarLinear(self.num_latents, 
                           self.c_hid,
                           [self.num_latents]),
            nn.Tanh(),
            MultivarLinear(self.c_hid, 
                           self.c_hid,
                           [self.num_latents])
        )
        self.out_layer = MultivarSequential(
            nn.SiLU(),
            MultivarLinear(c_hid, c_hid, [self.num_latents]),
            nn.SiLU(),
            MultivarLinear(c_hid, 2, 
                           [self.num_latents])
        )
        
        # Action preprocessing network, i.e. the MLP for interaction variable prediction.
        if self.action_size == 2:
            # Encoding x-y coordinates via sinusoidal embeddings
            encoding_layer = SinusoidalEncoding(grid_size=img_width, 
                                                out_dim=self.c_hid // 4,
                                                input_dim=self.action_size)
            enc_dim = encoding_layer.get_output_dim(self.action_size)
        else:
            encoding_layer = nn.Identity()
            enc_dim = self.action_size
        self.action_preprocess = MultivarSequential(
            encoding_layer,
            MultivarLinear(enc_dim + (self.num_latents if self.add_prev_state else 0), 
                           self.c_hid,
                           [self.num_latents]),
            nn.SiLU(),
            MultivarLinear(self.c_hid, 
                           self.c_hid,
                           [self.num_latents]),
            nn.SiLU(),
            MultivarLinear(self.c_hid, 
                           1,
                           [self.num_latents])
        )
        if self.add_prev_state:
            # If we add the previous state, it is better to initialize the weights
            # to put equal importance on actions and previous states, since the
            # previous state is usually much higher dimensional (e.g. 2/16 vs 40 in iTHOR)
            self.action_preprocess[1].weight.data[...,enc_dim:] *= np.sqrt(enc_dim / self.num_latents)
            self.action_preprocess[1].weight.data[...,:enc_dim] *= np.sqrt(self.num_latents / enc_dim)

    def _prepare_prior_input(self):
        if self.training:
            temp_factor = self.temp_scheduler.get_factor(self.step)
            self.step += 1
        else:
            temp_factor = 1.0
        return {'temp_factor': temp_factor}

    def requires_prev_state(self):
        return self.add_prev_state

    def _get_prior_params(self, z_t, action, detach_weights=False, **kwargs):
        """
        Abstracting the execution of the networks for estimating the prior parameters.

        Parameters
        ----------
        z_t : torch.FloatTensor, shape [batch_size, num_latents]
              Latents at time step t, i.e. the input to the prior
        action : torch.FloatTensor, shape [batch_size, action_size]
        detach_weights : bool, default False
                         Whether to detach the weights of the prior network.
                         Not strictly used anymore, but can be useful for debugging.
        """
        action_feats, extra_info = self._get_action_feats(action, 
                z_t=z_t.reshape(action.shape[0], -1, z_t.shape[-1])[:,0], 
                detach_weights=detach_weights,
                **kwargs)
        
        if z_t.shape[0] > action_feats.shape[0]:
            sample_ext = int(z_t.shape[0] / action_feats.shape[0])
            action_feats = action_feats[:,None].expand(-1, sample_ext, -1, -1, -1).flatten(0, 1)

        net_inp = z_t
        context = self.context_layer(net_inp[...,None,:].expand(-1, self.num_latents, -1),
                                     detach_weights=detach_weights)
        net_inp_exp = net_inp.unflatten(-1, (self.num_latents, -1))
        block_inp = self.inp_layer(net_inp_exp,
                                   detach_weights=detach_weights)

        prior_inp = context[:, None, :, :] + block_inp[:, None, :, :] + action_feats
        prior_inp = prior_inp.flatten(0, 1)
        prior_params = self.out_layer(prior_inp, detach_weights=detach_weights)
        prior_params = prior_params.unflatten(0, action_feats.shape[:2])
        prior_params = prior_params.chunk(2, dim=-1)
        prior_params = [p.flatten(-2, -1) for p in prior_params]
        return prior_params, extra_info

    def _get_action_feats(self, action, temp_factor=1.0, z_t=None, detach_weights=False, **kwargs):
        """
        Determining the interaction variables from the action and previous time step.

        Parameters
        ----------
        action : torch.FloatTensor, shape [batch_size, action_size]
                 Action/Regime variable at time step t+1 (i.e. causing change from t -> t+1)
        temp_factor: float, default 1.0
                     Temperature factor for the Tanh activation function
        z_t : torch.FloatTensor, shape [batch_size, num_latents]
              Latents at time step t, i.e. previous time step to condition interaction vars on
        detach_weights : bool, default False
                         Whether to detach the weights of the action preprocessing network.
                         Not strictly used anymore, but can be useful for debugging.
        """
        if self.add_prev_state:
            if action.ndim == 3:
                z_t = z_t[:,None,:].expand(-1, action.shape[1], -1)
            action = torch.cat([action, z_t], dim=-1)
            if self.training:
                self.last_batch_prev_state = z_t.detach()
        if action.ndim == 2:
            action = action[...,None,:].expand(-1, self.num_latents, -1)
        action = self.action_preprocess(action, detach_weights=detach_weights)
        abs_logits = torch.abs(action)
        action_logits = action
        action = torch.tanh(action * temp_factor)

        action_feats = self.action_layer(action, detach_weights=detach_weights)
        action_feats = action_feats.unsqueeze(dim=1)
        return action_feats, {'action': action, 'abs_logits': abs_logits, 'action_logits': action_logits}

    def get_interaction_quantization(self, action, prev_state=None, soft=False):
        """
        Return (binarized) interaction variables.

        Parameters
        ----------
        action : torch.FloatTensor, shape [batch_size, action_size]
                 Action/Regime variable at time step t+1 (i.e. causing change from t -> t+1)
        prev_state : torch.FloatTensor, shape [batch_size, num_latents], default None
                     Latents at time step t, i.e. previous time step to condition interaction vars on.
                     If None, the last batch of latents is used if needed.
        soft : bool, default False
               Whether to return the soft (i.e. tanh output) or hard binarization.
        """
        if self.add_prev_state:
            if prev_state is None:
                if self.last_batch_prev_state is None:
                    prev_state = action.new_zeros(*action.shape[:-1], self.num_latents)
                else:
                    prev_state = self.last_batch_prev_state[0:1]
                    for _ in range(action.ndim - prev_state.ndim):
                        prev_state = prev_state.unsqueeze(0)
                    prev_state = prev_state.expand(*action.shape[:-1], self.num_latents)
            action = torch.cat([action, prev_state], dim=-1)
        action = self.action_preprocess(action[...,None,:].expand(-1, self.num_latents, -1)).squeeze(dim=-1)
        if soft:
            action_idx = torch.tanh(action)
        else:
            action_idx = (action > 0).long()
        return action_idx

    def kl_divergence(self, z_t, action, z_t1_mean, z_t1_logstd, z_t1_sample):
        return self.sample_based_nll(z_t=z_t[:,None,:], 
                                     z_t1=z_t1_sample[:,None,:],
                                     action=action,
                                     use_KLD=True,
                                     z_t1_logstd=z_t1_logstd,
                                     z_t1_mean=z_t1_mean)

    def sample_based_nll(self, z_t, z_t1, action, use_KLD=False, z_t1_logstd=None, z_t1_mean=None):
        """
        Calculate the negative log likelihood of p(z^t1|z^t,I^t+1) in BISCUIT.
        For the NF, we cannot make use of the KL divergence since the normalizing flow 
        transforms the autoencoder distribution in a per-sample fashion. Nonetheless, to 
        improve stability and smooth gradients, we allow z^t and z^t1 to be multiple 
        samples for the same batch elements. 

        Parameters
        ----------
        z_t : torch.FloatTensor, shape [batch_size, num_samples, num_latents]
              Latents at time step t, i.e. the input to the prior, with potentially
              multiple samples over the second dimension.
        z_t1 : torch.FloatTensor, shape [batch_size, num_samples, num_latents]
               Latents at time step t+1, i.e. the samples to estimate the prior for.
               Multiple samples again over the second dimension.
        action : torch.FloatTensor, shape [batch_size, action_size]
                 Action/Regime variable at time step t+1 (i.e. causing change from t -> t+1)
        """
        batch_size, num_samples, _ = z_t.shape

        if len(action.shape) == 3:
            action = action.flatten(0, 1)
        
        extra_params = self._prepare_prior_input()

        # Obtain estimated prior parameters for p(z^t1|z^t,I^t+1)
        prior_params, extra_info = self._get_prior_params(z_t.flatten(0, 1), 
                                                          action=action,
                                                          **extra_params)

        if not use_KLD:
            prior_mean, prior_logstd = [p.unflatten(0, (batch_size, num_samples)) for p in prior_params]
            # prior_mean - shape [batch_size, num_samples, 1+num_blocks, num_latents]
            # Sample-based NLL
            nll = -gaussian_log_prob(prior_mean[:,None,:,:,:], prior_logstd[:,None,:,:,:], z_t1[:,:,None,None,:])
            nll = nll.mean(dim=[1, 2])  # Averaging over input and output samples - shape [batch_size, 1, num_latents]
            nll = nll.sum(dim=[1, 2])  # Summing over latents - shape [batch_size]
        else:
            prior_mean, prior_logstd = prior_params
            # KL-Divergence
            kld = kl_divergence(z_t1_mean[:,None,:], z_t1_logstd[:,None,:], prior_mean, prior_logstd)
            nll = kld.sum(dim=[1, 2])

        loss = self._calculate_loss(nll, **extra_info, **extra_params)

        return loss
    
    def sample(self, z_t, action, num_samples=1, **kwargs):
        """
        Sample from the prior distribution p(z^t1|z^t,I^t+1) in BISCUIT.

        Parameters
        ----------
        z_t : torch.FloatTensor, shape [batch_size, num_latents]
              Latents at time step t, i.e. the input to the prior
        action : torch.FloatTensor, shape [batch_size, action_size]
                 Action/Regime variable at time step t+1 (i.e. causing change from t -> t+1)
        num_samples : int, default 1
                      Number of samples to draw from the prior
        """
        extra_params = self._prepare_prior_input()
        prior_params, extra_info = self._get_prior_params(z_t, action=action, **extra_params)
        prior_mean, prior_logstd = prior_params
        z_t1 = torch.randn(z_t.shape[0], num_samples, self.num_latents, device=z_t.device)
        z_t1 = z_t1 * torch.exp(prior_logstd) + prior_mean
        return z_t1, (prior_mean, prior_logstd)

    def _calculate_loss(self, nll, action_logits, **kwargs):
        """
        Calculate the loss of the prior network, combining NLL and regularizer.

        Parameters
        ----------
        nll : torch.FloatTensor, shape [batch_size]
              Negative log likelihood of the prior
        action_logits : torch.FloatTensor, shape [batch_size, action_size]
                        Logits of the action variable
        """
        loss = nll

        # Small regularizer on the logits of actions
        # In general, this is only needed to prevent the logits to explode in the first
        # iterations and giving them a negative bias (i.e. 1 for interventional, -1/0 for observational)
        action_logits = action_logits.squeeze(dim=-1)
        logit_reg = self.logit_reg_factor * ((1 + torch.sign(action_logits) + action_logits - action_logits.detach()) ** 2)
        logit_reg = logit_reg.sum(dim=-1)
        logit_reg += self.logit_reg_factor * 0.01 * (action_logits ** 2).mean(dim=-1)
        loss = loss + logit_reg - logit_reg.detach()
        return loss

    def set_variable_alignments(self, variable_alignments):
        if isinstance(variable_alignments, np.ndarray):
            variable_alignments = torch.from_numpy(variable_alignments)
        self.variable_alignments = variable_alignments

    def get_variable_alignments(self):
        return self.variable_alignments

    def allow_sign_flip(self):
        return True