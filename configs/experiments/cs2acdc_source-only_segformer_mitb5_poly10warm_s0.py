# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/segformer_b5.py',
    '../_base_/datasets/cityscapes_to_acdc_512x512.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')
# Meta Information for Result Analysis
name = 'cs2acdc_source-only_segformer_mitb5_poly10warm_s0'
exp = 'basic'
name_dataset = 'cityscapes2acdc'
name_architecture = 'segformer_mitb5'
name_encoder = 'mitb5'
name_decoder = 'segformer'
name_uda = 'source-only'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
