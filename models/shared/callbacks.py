import torch
import torch.nn.functional as F
import torch.utils.data as data
import pytorch_lightning as pl
from torchmetrics.functional.classification import binary_f1_score
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
import os
import optuna
from collections import OrderedDict
from tqdm.auto import tqdm
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment

import sys
sys.path.append('../../')
from models.shared.visualization import visualize_reconstruction
from models.shared.utils import log_matrix
from models.shared.causal_encoder import CausalEncoder
from experiments.datasets import iTHORDataset, CausalWorldDataset


class ImageLogCallback(pl.Callback):
    """ Callback for creating visualizations for logging """

    def __init__(self, exmp_inputs, dataset, every_n_epochs=10, cluster=False, prefix=''):
        super().__init__()
        self.imgs = exmp_inputs[0]
        if len(exmp_inputs) > 2 and len(exmp_inputs[1].shape) == len(self.imgs.shape):
            self.labels = exmp_inputs[1]
            self.extra_inputs = exmp_inputs[2:]
        else:
            self.labels = self.imgs
            self.extra_inputs = exmp_inputs[1:]
        self.dataset = dataset
        self.every_n_epochs = every_n_epochs
        self.prefix = prefix
        self.cluster = cluster

    def on_train_epoch_end(self, trainer, pl_module):
        def log_fig(tag, fig):
            if fig is None:
                return
            trainer.logger.experiment.add_figure(f'{self.prefix}{tag}', fig, global_step=trainer.global_step)
            plt.close(fig)

        if self.imgs is not None and (trainer.current_epoch+1) % self.every_n_epochs == 0:
            trainer.model.eval()
            images = self.imgs.to(trainer.model.device)
            labels = self.labels.to(trainer.model.device)
            if len(images.shape) == 5:
                full_imgs, full_labels = images, labels
                images = images[:,0]
                labels = labels[:,0]
            else:
                full_imgs, full_labels = None, None

            for i in range(min(4, images.shape[0])):
                log_fig(f'reconstruction_{i}', visualize_reconstruction(trainer.model, images[i], labels[i], self.dataset))
            
            trainer.model.train()


class CorrelationMetricsLogCallback(pl.Callback):
    """ Callback for extracting correlation metrics (R^2 and Spearman) """

    def __init__(self, dataset, every_n_epochs=10, num_train_epochs=100, cluster=False, test_dataset=None):
        super().__init__()
        assert dataset is not None, "Dataset for correlation metrics cannot be None."
        self.dataset = dataset
        self.val_dataset = dataset
        self.test_dataset = test_dataset
        self.every_n_epochs = every_n_epochs
        self.num_train_epochs = num_train_epochs
        self.cluster = cluster
        self.log_postfix = ''
        self.extra_postfix = ''

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if isinstance(self.dataset, dict):
            dataset_dict = self.dataset
            sum([len(dataset_dict[key]) for key in dataset_dict])
            for key in dataset_dict:
                self.dataset = dataset_dict[key]
                self.log_postfix = f'{self.extra_postfix}_{key}'
                self.test_model(trainer, pl_module)
            self.log_postfix = ''
            self.dataset = dataset_dict
        else:
            self.test_model(trainer, pl_module)

        results = trainer._results
        if 'validation_step.val_loss' in results:
            val_comb_loss = results['validation_step.val_loss'].value / results['validation_step.val_loss'].cumulated_batch_size
            new_val_dict = {'val_loss': val_comb_loss}
            for key in ['on_validation_epoch_end.corr_callback_r2_matrix_diag',
                        'on_validation_epoch_end.corr_callback_spearman_matrix_diag',
                        'on_validation_epoch_end.corr_callback_r2_matrix_max_off_diag',
                        'on_validation_epoch_end.corr_callback_spearman_matrix_max_off_diag']:
                if key in results:
                    val = results[key].value
                    new_val_dict[key.split('_',5)[-1]] = val
            new_val_dict = {key: (val.item() if isinstance(val, torch.Tensor) else val) for key, val in new_val_dict.items()}
            if self.cluster:
                s = f'[Epoch {trainer.current_epoch}] ' + ', '.join([f'{key}: {new_val_dict[key]:5.3f}' for key in sorted(list(new_val_dict.keys()))])
                print(s)

    @torch.no_grad()
    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_dataset is None:
            print('Skipping correlation metrics testing due to missing dataset...')
        else:
            val_dataset = self.dataset
            self.dataset = self.test_dataset
            self.log_postfix = '_test'
            self.extra_postfix = '_test'
            self.on_validation_epoch_end(trainer, pl_module)
            self.dataset = val_dataset
            self.log_postfix = ''
            self.extra_postfix = ''

    @torch.no_grad()
    def test_model(self, trainer, pl_module):
        # Encode whole dataset with pl_module
        is_training = pl_module.training
        pl_module = pl_module.eval()
        loader = data.DataLoader(self.dataset, batch_size=256, drop_last=False, shuffle=False)
        all_encs, all_latents = [], []
        for batch in loader:
            inps, *_, latents = batch
            encs = pl_module.encode(inps.to(pl_module.device)).cpu()
            all_encs.append(encs)
            all_latents.append(latents)
        all_encs = torch.cat(all_encs, dim=0)
        all_latents = torch.cat(all_latents, dim=0)
        # Normalize latents for stable gradient signals
        all_encs = (all_encs - all_encs.mean(dim=0, keepdim=True)) / all_encs.std(dim=0, keepdim=True).clamp(min=1e-2)
        # Create new tensor dataset for training (50%) and testing (50%)
        full_dataset = data.TensorDataset(all_encs, all_latents)
        train_size = int(0.5 * all_encs.shape[0])
        test_size = all_encs.shape[0] - train_size
        train_dataset, test_dataset = data.random_split(full_dataset, 
                                                        lengths=[train_size, test_size], 
                                                        generator=torch.Generator().manual_seed(42))
        # Train network to predict causal factors from latent variables
        if hasattr(pl_module, 'target_assignment') and pl_module.target_assignment is not None:
            target_assignment = pl_module.target_assignment.clone()
        else:
            target_assignment = torch.eye(all_encs.shape[-1])
        encoder = self.train_network(pl_module, train_dataset, target_assignment)
        encoder.eval()
        # Record predictions of model on test and calculate distances
        test_inps, test_labels = all_encs[test_dataset.indices], all_latents[test_dataset.indices]
        test_exp_inps, test_exp_labels = self._prepare_input(test_inps, target_assignment.cpu(), test_labels, flatten_inp=False)
        pred_dict = encoder.forward(test_exp_inps.to(pl_module.device))
        for key in pred_dict:
            pred_dict[key] = pred_dict[key].cpu()
        _, dists, norm_dists = encoder.calculate_loss_distance(pred_dict, test_exp_labels)
        # Calculate statistics (R^2, pearson, etc.)
        avg_norm_dists, r2_matrix = self.log_R2_statistic(trainer, encoder, pred_dict, test_labels, norm_dists, pl_module=pl_module)
        self.log_Spearman_statistics(trainer, encoder, pred_dict, test_labels, pl_module=pl_module)
        if is_training:
            pl_module = pl_module.train()
        return r2_matrix

    @torch.enable_grad() 
    @torch.inference_mode(False)
    def train_network(self, pl_module, train_dataset, target_assignment):
        device = pl_module.device
        if hasattr(pl_module, 'causal_encoder') and pl_module.causal_encoder is not None:
            causal_var_info = pl_module.causal_encoder.hparams.causal_var_info
        else:
            causal_var_info = pl_module.hparams.causal_var_info
        # We use one, sufficiently large network that predicts for any input all causal variables
        # To iterate over the different sets, we use a mask which is an extra input to the model
        # This is more efficient than using N networks and showed same results with large hidden size
        encoder = CausalEncoder(c_hid=128,
                                lr=4e-3,
                                causal_var_info=causal_var_info,
                                single_linear=True,
                                c_in=pl_module.hparams.num_latents*2,
                                warmup=0)
        optimizer, _ = encoder.configure_optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]

        train_loader = data.DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=512)
        target_assignment = target_assignment.to(device)
        encoder.to(device)
        encoder.train()
        range_iter = range(self.num_train_epochs)
        if not self.cluster:
            range_iter = tqdm(range_iter, leave=False, desc=f'Training correlation encoder {self.log_postfix}')
        for epoch_idx in range_iter:
            avg_loss = 0.0
            for inps, latents in train_loader:
                inps = inps.to(device)
                latents = latents.to(device)
                inps, latents = self._prepare_input(inps, target_assignment, latents)
                loss = encoder._get_loss([inps, latents], mode=None)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
        return encoder

    def _prepare_input(self, inps, target_assignment, latents, flatten_inp=True):
        ta = target_assignment.detach()[None,:,:].expand(inps.shape[0], -1, -1)
        inps = torch.cat([inps[:,:,None] * ta, ta], dim=-2).permute(0, 2, 1)
        latents = latents[:,None].expand(-1, inps.shape[1], -1)
        if flatten_inp:
            inps = inps.flatten(0, 1)
            latents = latents.flatten(0, 1)
        return inps, latents

    def log_R2_statistic(self, trainer, encoder, pred_dict, test_labels, norm_dists, pl_module=None):
        avg_pred_dict = OrderedDict()
        for i, var_key in enumerate(encoder.hparams.causal_var_info):
            var_info = encoder.hparams.causal_var_info[var_key]
            gt_vals = test_labels[...,i]
            if var_info.startswith('continuous'):
                avg_pred_dict[var_key] = gt_vals.mean(dim=0, keepdim=True).expand(gt_vals.shape[0],)
            elif var_info.startswith('angle'):
                avg_angle = torch.atan2(torch.sin(gt_vals).mean(dim=0, keepdim=True), 
                                        torch.cos(gt_vals).mean(dim=0, keepdim=True)).expand(gt_vals.shape[0],)
                avg_angle = torch.where(avg_angle < 0.0, avg_angle + 2*np.pi, avg_angle)
                avg_pred_dict[var_key] = torch.stack([torch.sin(avg_angle), torch.cos(avg_angle)], dim=-1)
            elif var_info.startswith('categ'):
                gt_vals = gt_vals.long()
                mode = torch.mode(gt_vals, dim=0, keepdim=True).values
                avg_pred_dict[var_key] = F.one_hot(mode, int(var_info.split('_')[-1])).float().expand(gt_vals.shape[0], -1)
            else:
                assert False, f'Do not know how to handle key \"{var_key}\" in R2 statistics.'
        _, _, avg_norm_dists = encoder.calculate_loss_distance(avg_pred_dict, test_labels, keep_sign=True)

        r2_matrix = []
        for var_key in encoder.hparams.causal_var_info:
            ss_res = (norm_dists[var_key] ** 2).mean(dim=0)
            ss_tot = (avg_norm_dists[var_key] ** 2).mean(dim=0, keepdim=True)
            ss_tot = np.where(ss_tot == 0.0, 1.0, ss_tot)
            r2 = 1 - ss_res / ss_tot
            r2_matrix.append(r2)
        r2_matrix = torch.stack(r2_matrix, dim=-1).cpu().numpy()
        log_matrix(r2_matrix, trainer, 'r2_matrix' + self.log_postfix)
        self._log_heatmap(trainer=trainer, 
                          values=r2_matrix, 
                          tag='r2_matrix',
                          title='R^2 Matrix',
                          xticks=[key for key in encoder.hparams.causal_var_info],
                          pl_module=pl_module)
        return avg_norm_dists, r2_matrix

    def log_pearson_statistic(self, trainer, encoder, pred_dict, test_labels, norm_dists, avg_gt_norm_dists, pl_module=None):
        avg_pred_dict = OrderedDict()
        for i, var_key in enumerate(encoder.hparams.causal_var_info):
            var_info = encoder.hparams.causal_var_info[var_key]
            pred_vals = pred_dict[var_key]
            if var_info.startswith('continuous'):
                pred_vals = pred_vals.squeeze(dim=-1)
                avg_pred_dict[var_key] = pred_vals.mean(dim=0, keepdim=True).expand(pred_vals.shape[0], -1)
            elif var_info.startswith('angle'):
                angles = torch.atan(pred_vals[...,0] / pred_vals[...,1])
                avg_angle = torch.atan2(torch.sin(angles).mean(dim=0, keepdim=True), 
                                        torch.cos(angles).mean(dim=0, keepdim=True)).expand(pred_vals.shape[0], -1)
                avg_angle = torch.where(avg_angle < 0.0, avg_angle + 2*np.pi, avg_angle)
                avg_pred_dict[var_key] = avg_angle
            elif var_info.startswith('categ'):
                pred_vals = pred_vals.argmax(dim=-1)
                mode = torch.mode(pred_vals, dim=0, keepdim=True).values
                avg_pred_dict[var_key] = mode.expand(pred_vals.shape[0], -1)
            else:
                assert False, f'Do not know how to handle key \"{var_key}\" in Pearson statistics.'
        _, _, avg_pred_norm_dists = encoder.calculate_loss_distance(pred_dict, gt_vec=torch.stack([avg_pred_dict[key] for key in avg_pred_dict], dim=-1), keep_sign=True)

        pearson_matrix = []
        for var_key in encoder.hparams.causal_var_info:
            var_info = encoder.hparams.causal_var_info[var_key]
            pred_dist, gt_dist = avg_pred_norm_dists[var_key], avg_gt_norm_dists[var_key]
            nomin = (pred_dist * gt_dist[:,None]).sum(dim=0)
            denom = torch.sqrt((pred_dist**2).sum(dim=0) * (gt_dist[:,None]**2).sum(dim=0))
            p = nomin / denom.clamp(min=1e-5)
            pearson_matrix.append(p)
        pearson_matrix = torch.stack(pearson_matrix, dim=-1).cpu().numpy()
        log_matrix(pearson_matrix, trainer, 'pearson_matrix' + self.log_postfix)
        self._log_heatmap(trainer=trainer, 
                          values=pearson_matrix, 
                          tag='pearson_matrix',
                          title='Pearson Matrix',
                          xticks=[key for key in encoder.hparams.causal_var_info],
                          pl_module=pl_module)

    def log_Spearman_statistics(self, trainer, encoder, pred_dict, test_labels, pl_module=None):
        spearman_matrix = []
        for i, var_key in enumerate(encoder.hparams.causal_var_info):
            var_info = encoder.hparams.causal_var_info[var_key]
            gt_vals = test_labels[...,i]
            pred_val = pred_dict[var_key]
            if var_info.startswith('continuous'):
                spearman_preds = pred_val.squeeze(dim=-1)  # Nothing needs to be adjusted
            elif var_info.startswith('angle'):
                spearman_preds = F.normalize(pred_val, p=2.0, dim=-1)
                gt_vals = torch.stack([torch.sin(gt_vals), torch.cos(gt_vals)], dim=-1)
                # angles = torch.atan(pred_val[...,0] / pred_val[...,1])
                # angles = torch.where(angles < 0.0, angles + 2*np.pi, angles)
                # spearman_preds = angles
            elif var_info.startswith('categ'):
                spearman_preds = pred_val.argmax(dim=-1).float()
            else:
                assert False, f'Do not know how to handle key \"{var_key}\" in Spearman statistics.'

            gt_vals = gt_vals.cpu().numpy()
            spearman_preds = spearman_preds.cpu().numpy()
            results = torch.zeros(spearman_preds.shape[1],)
            for j in range(spearman_preds.shape[1]):
                if len(spearman_preds.shape) == 2:
                    if np.unique(spearman_preds[:,j]).shape[0] == 1:
                        results[j] = 0.0
                    else:
                        results[j] = spearmanr(spearman_preds[:,j], gt_vals).correlation
                elif len(spearman_preds.shape) == 3:
                    num_dims = spearman_preds.shape[-1]
                    for k in range(num_dims):
                        if np.unique(spearman_preds[:,j,k]).shape[0] == 1:
                            results[j] = 0.0
                        else:
                            results[j] += spearmanr(spearman_preds[:,j,k], gt_vals[...,k]).correlation
                    results[j] /= num_dims
                
            spearman_matrix.append(results)
        
        spearman_matrix = torch.stack(spearman_matrix, dim=-1).cpu().numpy()
        log_matrix(spearman_matrix, trainer, 'spearman_matrix' + self.log_postfix)
        self._log_heatmap(trainer=trainer, 
                          values=spearman_matrix, 
                          tag='spearman_matrix',
                          title='Spearman\'s Rank Correlation Matrix',
                          xticks=[key for key in encoder.hparams.causal_var_info],
                          pl_module=pl_module)

    def _log_heatmap(self, trainer, values, tag, title=None, xticks=None, yticks=None, xlabel=None, ylabel=None, pl_module=None):
        if ylabel is None:
            ylabel = 'Target dimension'
        if xlabel is None:
            xlabel = 'True causal variable'
        if yticks is None:
            yticks = [f'Dim {i+1}' for i in range(values.shape[0])]
        if xticks is None:
            xticks = self.dataset.target_names()
        fig = plt.figure(figsize=(min(6, max(4, values.shape[1]/1.25)), 
                                  min(6, max(4, values.shape[0]/1.25))), 
                         dpi=150)
        sns.heatmap(values, annot=min(values.shape) < 10,
                    yticklabels=yticks,
                    xticklabels=xticks,
                    vmin=0.0,
                    vmax=1.0,
                    fmt='3.2f')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title is not None:
            plt.title(title)
        fig.tight_layout()

        trainer.logger.experiment.add_figure(tag + self.log_postfix, fig, global_step=trainer.global_step)
        plt.close(fig)

        if values.shape[0] == values.shape[1] + 1:
            values = values[:-1]

        if values.shape[0] == values.shape[1]:
            avg_diag = np.diag(values).mean()
            max_off_diag = (values - np.eye(values.shape[0]) * 10).max(axis=-1).mean()
            if pl_module is None:
                trainer.logger.experiment.add_scalar(f'corr_callback_{tag}_diag{self.log_postfix}', avg_diag, global_step=trainer.global_step)
                trainer.logger.experiment.add_scalar(f'corr_callback_{tag}_max_off_diag{self.log_postfix}', max_off_diag, global_step=trainer.global_step)
            else:
                pl_module.log(f'corr_callback_{tag}_diag{self.log_postfix}', avg_diag)
                pl_module.log(f'corr_callback_{tag}_max_off_diag{self.log_postfix}', max_off_diag)


class PermutationCorrelationMetricsLogCallback(CorrelationMetricsLogCallback):
    """ 
    Adapting the correlation metrics callback by first running 
    the correlation estimation for every single latent variable, and then grouping
    them according to the highest correlation with a causal variable.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.test_dataset is None:
            self.test_dataset = self.val_dataset

    @torch.no_grad()
    def on_validation_epoch_start(self, trainer, pl_module, is_test=False):
        self.log_postfix = '_all_latents' + ('_test' if is_test else '')
        self.extra_postfix = '_test' if is_test else ''
        pl_module.target_assignment = None
        r2_matrix = self.test_model(trainer, pl_module)
        r2_matrix = torch.from_numpy(r2_matrix)
        # Assign each latent to the causal variable with the highest (relative) correlation
        r2_matrix = r2_matrix / r2_matrix.abs().max(dim=0, keepdims=True).values.clamp(min=0.1)
        max_r2 = r2_matrix.argmax(dim=-1)
        ta = F.one_hot(max_r2, num_classes=r2_matrix.shape[-1]).float()
        # Group multi-dimensional causal variables together
        if isinstance(self.dataset, iTHORDataset):
            ta = torch.cat([ta[:,:1],
                            ta[:,1:7].sum(dim=-1, keepdims=True),
                            ta[:,7:9],
                            ta[:,9:13].sum(dim=-1, keepdims=True),
                            ta[:,13:]], dim=-1)
        elif isinstance(self.dataset, CausalWorldDataset):
            ta = torch.cat([ta[:,:6],
                            ta[:,6:].sum(dim=-1, keepdims=True)], dim=-1)
        if trainer.current_epoch == 0:
            ta[:,0] = 1
            ta[:,1:] = 0
        pl_module.target_assignment = ta
        pl_module.last_target_assignment.data = ta

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module, is_test=False):
        self.log_postfix = '_grouped_latents' + ('_test' if is_test else '')
        self.test_model(trainer, pl_module)
        
        if not is_test:
            results = trainer._results
            if 'validation_step.val_loss' in results:
                val_comb_loss = results['validation_step.val_loss'].value / results['validation_step.val_loss'].cumulated_batch_size
                new_val_dict = {'val_loss': val_comb_loss}
                for key in ['on_validation_epoch_end.corr_callback_r2_matrix_diag_grouped_latents',
                            'on_validation_epoch_end.corr_callback_spearman_matrix_diag_grouped_latents',
                            'on_validation_epoch_end.corr_callback_r2_matrix_max_off_diag_grouped_latents',
                            'on_validation_epoch_end.corr_callback_spearman_matrix_max_off_diag_grouped_latents']:
                    if key in results:
                        val = results[key].value
                        new_val_dict[key.split('_',5)[-1]] = val
                new_val_dict = {key: (val.item() if isinstance(val, torch.Tensor) else val) for key, val in new_val_dict.items()}
                if self.cluster:
                    s = f'[Epoch {trainer.current_epoch}] ' + ', '.join([f'{key}: {new_val_dict[key]:5.3f}' for key in sorted(list(new_val_dict.keys()))])
                    print(s)

    @torch.no_grad()
    def on_test_epoch_start(self, trainer, pl_module):
        self.dataset = self.test_dataset
        self.on_validation_epoch_start(trainer, pl_module, is_test=True)

    @torch.no_grad()
    def on_test_epoch_end(self, trainer, pl_module):
        self.on_validation_epoch_end(trainer, pl_module, is_test=True)
        self.dataset = self.val_dataset


class InteractionVisualizationCallback(pl.Callback):

    def __init__(self, action_data_loader, prefix='', trial=None):
        super().__init__()
        self.action_data_loader = action_data_loader
        self.prefix = prefix
        self.all_match_accuracies = []
        self.trial = trial

    @torch.no_grad()
    def on_validation_epoch_start(self, trainer, pl_module):
        all_true_intvs = []
        all_pred_intvs = []
        prior_module = pl_module.prior_t1
        for batch in self.action_data_loader:
            img_inp, *_, action_inp, true_intv = batch
            action_inp = action_inp.to(pl_module.device)
            true_intv = true_intv.to(pl_module.device)
            if len(action_inp.shape) > 2:
                action_inp = action_inp.flatten(0, -2)
            if len(true_intv.shape) > 2:
                true_intv = true_intv.flatten(0, -2)
            all_true_intvs.append(true_intv)

            if action_inp.shape[-1] != prior_module.action_size:
                return

            prev_state = None
            if prior_module.requires_prev_state():
                img_inp = img_inp.to(pl_module.device)
                prev_state = pl_module.encode(img_inp[:,0])
            pred_intv = prior_module.get_interaction_quantization(action_inp, prev_state=prev_state)
            all_pred_intvs.append(pred_intv)

        true_intv = torch.cat(all_true_intvs, dim=0).long()
        pred_intv = torch.cat(all_pred_intvs, dim=0)
        num_true_targets = true_intv.shape[1]
        num_pred_targets = pred_intv.shape[1]

        if true_intv.sum() > 0:
            true_intv = true_intv[:,None,:].expand(-1, num_pred_targets, -1).permute(1, 2, 0).flatten(0, 1)
            pred_intv = pred_intv[:,:,None].expand(-1, -1, num_true_targets).permute(1, 2, 0).flatten(0, 1)
            f1_scores_pos = binary_f1_score(pred_intv, true_intv, multidim_average='samplewise')
            if prior_module.allow_sign_flip():
                f1_scores_neg = binary_f1_score(1-pred_intv, true_intv, multidim_average='samplewise')
                f1_scores = torch.maximum(f1_scores_pos, f1_scores_neg)
            else:
                f1_scores = f1_scores_pos
            accs = f1_scores.unflatten(0, (num_pred_targets, num_true_targets))
            accs = accs.cpu().numpy()

            # Log optimal assignment
            if num_true_targets == num_pred_targets:
                row_ind, col_ind = linear_sum_assignment(accs, maximize=True)
                best_acc = accs[row_ind,col_ind].mean()
                var_assignments = row_ind[np.argsort(col_ind)]
            else:
                best_acc = accs.max(axis=0).mean()
                var_assignments = np.argmax(accs, axis=1)
            trainer.logger.experiment.add_scalar(f'{self.prefix}interaction_match_f1', best_acc, global_step=trainer.global_step)
            self.all_match_accuracies.append(best_acc)
            self.report_to_trial(trainer, best_acc)
            prior_module.set_variable_alignments(var_assignments)

            # Visualize
            fig = plt.figure(figsize=(max(4, num_pred_targets), max(4, num_true_targets)), dpi=200)
            sns.heatmap(accs, annot=True,
                        fmt='.2%',
                        vmin=0.0,
                        vmax=1.0,
                        cmap='cividis')
            plt.title('F1 Scores between predicted and true interaction variables')
            plt.xlabel('True Causal Variable Index')
            plt.ylabel('Predicted Interaction Variable Index')
            fig.tight_layout()
            trainer.logger.experiment.add_figure(f'{self.prefix}interaction_matches', fig, global_step=trainer.global_step)
            plt.close(fig)

        if action_inp.shape[-1] == 2:
            resolution = 256 if pl_module.hparams.num_latents < 16 else 64
            x, y = torch.meshgrid(torch.linspace(-1, 1, steps=resolution),
                                  torch.linspace(-1, 1, steps=resolution),
                                  indexing='xy')
            xy = torch.stack([x, y], dim=-1).flatten(0, 1)
            xy = xy.to(pl_module.device)
            outs = []
            for i in range(0, xy.shape[0], 4096):
                outs.append(prior_module.get_interaction_quantization(xy[i:i+4096], soft=True))
            pred_intv = torch.cat(outs, dim=0)
            pred_intv = pred_intv.unflatten(0, (x.shape[0], y.shape[0])).cpu().numpy()
            os.makedirs(os.path.join(trainer.logger.log_dir, 'interaction_matches'), exist_ok=True)
            extra_save_args = {}
            if hasattr(pl_module.prior_t1, 'last_batch_prev_state') and pl_module.prior_t1.last_batch_prev_state is not None:
                prev_state = pl_module.prior_t1.last_batch_prev_state[0:1]
                extra_save_args['prev_state'] = prev_state[0].cpu().numpy()
                if hasattr(pl_module, 'autoencoder'):
                    ae_latents = pl_module.flow.reverse(prev_state)
                    ae_rec = pl_module.autoencoder.decoder(ae_latents).cpu().numpy()
                    ae_rec = (ae_rec + 1.0) / 2.0
                    ae_rec = ae_rec[0].transpose(1, 2, 0)
                    extra_save_args['prev_img'] = ae_rec
            np.savez_compressed(os.path.join(trainer.logger.log_dir, 'interaction_matches',
                                             f'{self.prefix}interaction_matches_{str(trainer.current_epoch).zfill(5)}.npz'), 
                                pred_intv=pred_intv, **extra_save_args)
            pred_intv = (pred_intv > 0).astype(np.int32)
            
            num_vars = pred_intv.shape[-1]
            img = np.zeros((resolution, resolution, 3), dtype=np.float32)
            counts = np.zeros((resolution, resolution, num_vars))
            hues = [hsv_to_rgb([i/num_vars*0.9, 1.0, 1.0]) for i in range(num_vars)]
            for i in range(num_vars):
                ind = pred_intv[...,i]
                if prior_module.allow_sign_flip() and ind.mean() > 0.5:
                    ind = 1 - ind
                img += ind[...,None] * hues[i][None,None]
                counts[:,:,i] = ind

            counts_sum = counts.sum(axis=-1, keepdims=True)
            img = img / np.maximum(counts_sum, 1)
            img += (counts_sum == 0) * np.array([[[0.9, 0.9, 0.9]]])

            unique_combs, unq_counts = np.unique(counts.reshape(-1, num_vars), axis=0, return_counts=True)
            unique_combs = unique_combs[np.argsort(-unq_counts)]
            unique_combs = unique_combs[unique_combs.sum(axis=-1) > 1]

            handle_names = [f'Latent {i}' for i in range(len(hues)) if counts[:,:,i].sum() > 0]
            for comb in unique_combs[:10]:
                latent_idxs = np.where(comb == 1)[0]
                if latent_idxs.shape[0] > 5:
                    continue
                new_hue = sum([hues[i] for i in latent_idxs]) / comb.sum()
                hues.append(new_hue)
                handle_names.append(f'Latents {"&".join([str(i) for i in latent_idxs])}')

            fig = plt.figure(figsize=(6, 6), dpi=150)
            if len(handle_names) < 10:
                plt.legend(handles=[Patch(facecolor=color, label=label) for color, label in zip(hues, handle_names)], loc='center')
            img = img.clip(0.0, 1.0)
            plt.imshow(img)
            plt.axis('off')
            plt.title('Learned space partitioning')
            fig.tight_layout()
            trainer.logger.experiment.add_figure(f'{self.prefix}interaction_partitioning', fig, global_step=trainer.global_step)
            trainer.logger.experiment.add_image(f'{self.prefix}interaction_partitioning_clean', img, global_step=trainer.global_step, dataformats='HWC')
            plt.close(fig)

    def on_test_epoch_start(self, trainer, pl_module):
        old_prefix = self.prefix
        self.prefix = 'test_' + self.prefix
        self.on_validation_epoch_start(trainer, pl_module)
        self.prefix = old_prefix

    def get_max_accuracy(self):
        return max(self.all_match_accuracies)

    def set_trial(self, trial):
        self.trial = trial

    def report_to_trial(self, trainer, score):
        if trainer.sanity_checking or self.trial is None:
            return

        epoch = trainer.current_epoch
        self.trial.report(score, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f'Trial was pruned at epoch {epoch}')