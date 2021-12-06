# coding: utf-8
# author: wcc
# create date: 2021-01-10 20:35

config_wikipedia = {
    'dataset': 'wikipedia',

    'use_cuda': True,

    # model setting
    'embedding_dim': 64,
    'time_embedding_dim': 64,
    'k_shots': 6,
    'interval': 2,

    'phi_update': 1,
    'omega_update': 1,

    'lr': 1e-3,
    'base_lr': 25e-3,
    'encoder_lr': 25e-3,
    'local_lr': 25e-3,

    'num_neighbors': 16,
    'batch_size': 48,  # for each batch, the number of tasks
    'num_epoch': 24,
    'attn_mode': 'simple',
    'use_time': 'time',
    'agg_method': 'attn',
    'num_layers': 2,
    'n_head': 2,
    'drop_out': 0.5,
}


config_reddit = {
    'dataset': 'reddit',

    'use_cuda': True,

    # model setting
    'embedding_dim': 64,
    'time_embedding_dim': 64,
    'k_shots': 6,
    'interval': 2,

    'phi_update': 1,
    'omega_update': 1,

    'lr': 1e-3,
    'base_lr': 2e-2,
    'encoder_lr': 2e-4,
    'local_lr': 25e-3,

    'num_neighbors': 16,
    'batch_size': 64,  # for each batch, the number of tasks
    'num_epoch': 20,
    'attn_mode': 'multi',
    'use_time': 'time',
    'agg_method': 'attn',
    'num_layers': 2,
    'n_head': 2,
    'drop_out': 0.5,
}


config_dblp = {
    'dataset': 'dblp',

    'use_cuda': True,

    # model setting
    'embedding_dim': 64,
    'time_embedding_dim': 64,
    'k_shots': 6,
    'interval': 2,

    'phi_update': 1,
    'omega_update': 1,

    'lr': 1e-3,
    'base_lr': 2e-2,
    'encoder_lr': 2e-4,
    'local_lr': 25e-3,

    'num_neighbors': 16,
    'batch_size': 32,  # for each batch, the number of tasks
    'num_epoch': 20,
    'attn_mode': 'multi',
    'use_time': 'time',
    'agg_method': 'attn',
    'num_layers': 2,
    'n_head': 2,
    'drop_out': 0.5,
}
