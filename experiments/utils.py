"""
General training function with PyTorch Lightning
"""

import os
import argparse
import json
import torch
import torch.utils.data as data
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint
from shutil import copyfile

import sys
sys.path.append('../')
from experiments.datasets import VoronoiDataset, CausalWorldDataset, iTHORDataset

def get_device():
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--cluster', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_img_width', type=int, default=-1)
    parser.add_argument('--seq_len', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=-1)
    parser.add_argument('--logger_name', type=str, default='')
    parser.add_argument('--files_to_save', type=str, nargs='+', default='')
    parser.add_argument('--compile', action='store_true')
    return parser

def load_datasets(args):
    pl.seed_everything(args.seed)
    print('Loading datasets...')
    if 'voronoi' in args.data_dir:
        data_name = 'voronoi' + args.data_dir.split('voronoi')[-1].replace('/','')
        DataClass = VoronoiDataset
        dataset_args = {'return_targets': False, 'return_robot_state': True}
    elif 'causal_world' in args.data_dir:
        data_name = 'causalworld' + args.data_dir.split('causal_world')[-1].replace('/','')
        DataClass = CausalWorldDataset
        dataset_args = {'downsample_images': True}
    elif 'ithor' in args.data_dir:
        data_name = 'ithor' + args.data_dir.split('ithor')[-1].replace('/','')
        DataClass = iTHORDataset
        dataset_args = {}
    else:
        assert False, f'Unknown data class for {args.data_dir}'
    if hasattr(args, 'try_encodings'):
        dataset_args['try_encodings'] = args.try_encodings
    train_dataset = DataClass(
        data_folder=args.data_dir, split='train', single_image=False, triplet=False, seq_len=args.seq_len, cluster=args.cluster, **dataset_args)
    val_dataset = DataClass(
        data_folder=args.data_dir, split='val_indep', single_image=True, triplet=False, return_latents=True, cluster=args.cluster, **dataset_args)
    val_seq_dataset = DataClass(
        data_folder=args.data_dir, split='val', single_image=False, triplet=False, seq_len=args.seq_len, cluster=args.cluster, **dataset_args)
    test_dataset = DataClass(
        data_folder=args.data_dir, split='test_indep', single_image=True, triplet=False, return_latents=True, cluster=args.cluster, **dataset_args)
    test_seq_dataset = DataClass(
        data_folder=args.data_dir, split='test', single_image=False, triplet=False, seq_len=args.seq_len, cluster=args.cluster, **dataset_args)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    val_seq_loader = data.DataLoader(val_seq_dataset, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=args.num_workers)
    test_seq_loader = data.DataLoader(test_seq_dataset, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=args.num_workers)

    print(f'Training dataset size: {len(train_dataset)} / {len(train_loader)}')
    print(f'Val sequence dataset size: {len(val_seq_dataset)} / {len(val_seq_loader)}')
    if isinstance(val_dataset, dict):
        print(f'Val correlation dataset sizes: { {key: len(val_dataset[key]) for key in val_dataset} }')
    else:
        print(f'Val correlation dataset size: {len(val_dataset)}')
    print(f'Test sequence dataset size: {len(test_seq_dataset)} / {len(test_seq_loader)}')
    if isinstance(test_dataset, dict):
        print(f'Test correlation dataset sizes: { {key: len(test_dataset[key]) for key in test_dataset} }')
    else:
        print(f'Test correlation dataset size: {len(test_dataset)}')

    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'val_seq': val_seq_dataset,
        'test': test_dataset,
        'test_seq': test_seq_dataset
    }
    data_loaders = {
        'train': train_loader,
        'val_seq': val_seq_loader,
        'test_seq': test_seq_loader
    }

    if hasattr(datasets['train'], 'action_size'):
        for key in ['return_robot_state', 'return_targets']:
            if key in dataset_args:
                dataset_args.pop(key)
        action_dataset = DataClass(data_folder=args.data_dir,
                                   split='val',
                                   return_robot_state=True,
                                   triplet=False,
                                   return_targets=True,
                                   **dataset_args
                                   )
        action_loader = data.DataLoader(action_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=False)
        datasets['action'] = action_dataset
        data_loaders['action'] = action_loader

    return datasets, data_loaders, data_name

def print_params(logger_name, model_args):
    num_chars = max(50, 11+len(logger_name))
    print('=' * num_chars)
    print(f'Experiment {logger_name}')
    print('-' * num_chars)
    for key in sorted(list(model_args.keys())):
        print(f'-> {key}: {model_args[key]}')
    print('=' * num_chars)

def train_model(model_class, train_loader, val_loader, 
                test_loader=None,
                logger_name=None,
                max_epochs=200,
                progress_bar_refresh_rate=1,
                check_val_every_n_epoch=1,
                debug=False,
                offline=False,
                op_before_running=None,
                load_pretrained=False,
                root_dir=None,
                files_to_save=None,
                gradient_clip_val=1.0,
                cluster=False,
                callback_kwargs=None,
                seed=42,
                save_last_model=False,
                val_track_metric='val_loss',
                data_dir=None,
                compile=False,
                **kwargs):
    torch.set_float32_matmul_precision('medium')
    trainer_args = {}
    if root_dir is None or root_dir == '':
        root_dir = os.path.join('checkpoints/', model_class.__name__)
    if not (logger_name is None or logger_name == ''):
        logger_name = logger_name.split('/')
        logger = pl.loggers.TensorBoardLogger(root_dir, 
                                              name=logger_name[0], 
                                              version=logger_name[1] if len(logger_name) > 1 else None)
        trainer_args['logger'] = logger
    if progress_bar_refresh_rate == 0:
        trainer_args['enable_progress_bar'] = False

    if callback_kwargs is None:
        callback_kwargs = dict()
    callbacks = model_class.get_callbacks(exmp_inputs=next(iter(val_loader)), cluster=cluster, 
                                          **callback_kwargs)
    if not debug:
        callbacks.append(
                ModelCheckpoint(save_weights_only=True, 
                                mode="min", 
                                monitor=val_track_metric,
                                save_last=save_last_model,
                                every_n_epochs=check_val_every_n_epoch)
            )
    if debug:
        torch.autograd.set_detect_anomaly(True) 
    trainer = pl.Trainer(default_root_dir=root_dir,
                         accelerator='gpu' if torch.cuda.is_available() else 'cpu', 
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=callbacks,
                         check_val_every_n_epoch=check_val_every_n_epoch,
                         gradient_clip_val=gradient_clip_val,
                         **trainer_args)
    trainer.logger._default_hp_metric = None

    if files_to_save is not None:
        log_dir = trainer.logger.log_dir
        os.makedirs(log_dir, exist_ok=True)
        for file in files_to_save:
            if os.path.isfile(file):
                filename = file.split('/')[-1]
                copyfile(file, os.path.join(log_dir, filename))
                print(f'=> Copied {filename}')
            else:
                print(f'=> File not found: {file}')

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(
        'checkpoints/', model_class.__name__ + ".ckpt")
    if load_pretrained and os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        model = model_class.load_from_checkpoint(pretrained_filename)
    else:
        if load_pretrained:
            print("Warning: Could not load any pretrained models despite", load_pretrained)
        pl.seed_everything(seed)  # To be reproducable
        model = model_class(**kwargs)
        if op_before_running is not None:
            model.to(get_device())
            op_before_running(model)
        if compile:
            if hasattr(torch, 'compile'):
                model = torch.compile(model)
            else:
                print('Warning: PyTorch version does not support compilation. Skipping...')
        trainer.fit(model, train_loader, val_loader)
        model = model_class.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training

    if test_loader is not None:
        model_paths = [(trainer.checkpoint_callback.best_model_path, "best")]
        if save_last_model:
            model_paths += [(trainer.checkpoint_callback.last_model_path, "last")]
        for file_path, prefix in model_paths:
            model = model_class.load_from_checkpoint(file_path)
            test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
            test_result = test_result[0]
            print('='*50)
            print(f'Test results ({prefix}):')
            print('-'*50)
            for key in test_result:
                print(key + ':', test_result[key])
            print('='*50)

            log_file = os.path.join(trainer.logger.log_dir, f'test_results_{prefix}.json')
            with open(log_file, 'w') as f:
                json.dump(test_result, f, indent=4)