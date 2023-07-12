"""
Run file to train BISCUIT-VAE
"""

import sys
sys.path.append('../')
from models.biscuit_vae import BISCUITVAE
from experiments.utils import train_model, load_datasets, get_default_parser, print_params


if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument('--c_hid', type=int, default=64)
    parser.add_argument('--decoder_num_blocks', type=int, default=1)
    parser.add_argument('--act_fn', type=str, default='silu')
    parser.add_argument('--num_latents', type=int, default=16)
    parser.add_argument('--prior_action_add_prev_state', action='store_true')
    parser.add_argument('--logit_reg_factor', type=float, default=0.0005)

    args = parser.parse_args()
    model_args = vars(args)

    datasets, data_loaders, data_name = load_datasets(args)

    model_args['data_folder'] = [s for s in args.data_dir.split('/') if len(s) > 0][-1]
    model_args['img_width'] = datasets['train'].get_img_width()
    model_args['max_iters'] = args.max_epochs * len(data_loaders['train'])
    if hasattr(datasets['train'], 'get_inp_channels'):
        model_args['c_in'] = datasets['train'].get_inp_channels()
    model_name = 'BISCUITVAE'
    model_class = BISCUITVAE
    logger_name = f'{model_name}_{args.num_latents}l_{datasets["train"].num_vars()}b_{args.c_hid}hid_{data_name}'
    args_logger_name = model_args.pop('logger_name')
    if len(args_logger_name) > 0:
        logger_name += '/' + args_logger_name

    print_params(logger_name, model_args)
    
    check_val_every_n_epoch = model_args.pop('check_val_every_n_epoch')
    if check_val_every_n_epoch <= 0:
        check_val_every_n_epoch = 1 if not args.cluster else 5
    train_model(model_class=model_class,
                train_loader=data_loaders['train'],
                val_loader=data_loaders['val_seq'],
                test_loader=data_loaders['test_seq'],
                logger_name=logger_name,
                check_val_every_n_epoch=check_val_every_n_epoch,
                progress_bar_refresh_rate=0 if args.cluster else 1,
                callback_kwargs={'dataset': datasets['train'], 
                                 'correlation_dataset': datasets['val'],
                                 'correlation_test_dataset': datasets['test'],
                                 'action_data_loader': data_loaders['action']},
                save_last_model=True,
                action_size=datasets['train'].action_size(),
                causal_var_info=datasets['train'].get_causal_var_info(),
                **model_args)
