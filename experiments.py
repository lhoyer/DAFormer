# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import itertools
import logging
import math


def get_model_base(architecture, backbone):
    architecture = architecture.replace('sfa_', '')
    architecture = architecture.replace('_nodbn', '')
    if 'segformer' in architecture:
        return {
            'mitb5': f'_base_/models/{architecture}_b5.py',
            # It's intended that <=b4 refers to b5 config
            'mitb4': f'_base_/models/{architecture}_b5.py',
            'mitb3': f'_base_/models/{architecture}_b5.py',
            'r101v1c': f'_base_/models/{architecture}_r101.py',
        }[backbone]
    if 'daformer_' in architecture and 'mitb5' in backbone:
        return f'_base_/models/{architecture}_mitb5.py'
    if 'upernet' in architecture and 'mit' in backbone:
        return f'_base_/models/{architecture}_mit.py'
    assert 'mit' not in backbone or '-del' in backbone
    return {
        'dlv2': '_base_/models/deeplabv2_r50-d8.py',
        'dlv2red': '_base_/models/deeplabv2red_r50-d8.py',
        'dlv3p': '_base_/models/deeplabv3plus_r50-d8.py',
        'da': '_base_/models/danet_r50-d8.py',
        'isa': '_base_/models/isanet_r50-d8.py',
        'uper': '_base_/models/upernet_r50.py',
    }[architecture]


def get_pretraining_file(backbone):
    if 'mitb5' in backbone:
        return 'pretrained/mit_b5.pth'
    if 'mitb4' in backbone:
        return 'pretrained/mit_b4.pth'
    if 'mitb3' in backbone:
        return 'pretrained/mit_b3.pth'
    if 'r101v1c' in backbone:
        return 'open-mmlab://resnet101_v1c'
    return {
        'r50v1c': 'open-mmlab://resnet50_v1c',
        'x50-32': 'open-mmlab://resnext50_32x4d',
        'x101-32': 'open-mmlab://resnext101_32x4d',
        's50': 'open-mmlab://resnest50',
        's101': 'open-mmlab://resnest101',
        's200': 'open-mmlab://resnest200',
    }[backbone]


def get_backbone_cfg(backbone):
    for i in [1, 2, 3, 4, 5]:
        if backbone == f'mitb{i}':
            return dict(type=f'mit_b{i}')
        if backbone == f'mitb{i}-del':
            return dict(_delete_=True, type=f'mit_b{i}')
    return {
        'r50v1c': {
            'depth': 50
        },
        'r101v1c': {
            'depth': 101
        },
        'x50-32': {
            'type': 'ResNeXt',
            'depth': 50,
            'groups': 32,
            'base_width': 4,
        },
        'x101-32': {
            'type': 'ResNeXt',
            'depth': 101,
            'groups': 32,
            'base_width': 4,
        },
        's50': {
            'type': 'ResNeSt',
            'depth': 50,
            'stem_channels': 64,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True
        },
        's101': {
            'type': 'ResNeSt',
            'depth': 101,
            'stem_channels': 128,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True
        },
        's200': {
            'type': 'ResNeSt',
            'depth': 200,
            'stem_channels': 128,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True,
        },
    }[backbone]


def update_decoder_in_channels(cfg, architecture, backbone):
    cfg.setdefault('model', {}).setdefault('decode_head', {})
    if 'dlv3p' in architecture and 'mit' in backbone:
        cfg['model']['decode_head']['c1_in_channels'] = 64
    if 'sfa' in architecture:
        cfg['model']['decode_head']['in_channels'] = 512
    return cfg


def setup_rcs(cfg, temperature):
    cfg.setdefault('data', {}).setdefault('train', {})
    cfg['data']['train']['rare_class_sampling'] = dict(
        min_pixels=3000, class_temp=temperature, min_crop_ratio=0.5)
    return cfg


def generate_experiment_cfgs(id):

    def config_from_vars():
        cfg = {'_base_': ['_base_/default_runtime.py'], 'n_gpus': n_gpus}
        if seed is not None:
            cfg['seed'] = seed

        # Setup model config
        architecture_mod = architecture
        model_base = get_model_base(architecture_mod, backbone)
        cfg['_base_'].append(model_base)
        cfg['model'] = {
            'pretrained': get_pretraining_file(backbone),
            'backbone': get_backbone_cfg(backbone),
        }
        if 'sfa_' in architecture_mod:
            cfg['model']['neck'] = dict(type='SegFormerAdapter')
        if '_nodbn' in architecture_mod:
            cfg['model'].setdefault('decode_head', {})
            cfg['model']['decode_head']['norm_cfg'] = None
        cfg = update_decoder_in_channels(cfg, architecture_mod, backbone)

        # Setup UDA config
        if uda == 'target-only':
            cfg['_base_'].append(f'_base_/datasets/{target}_half_{crop}.py')
        elif uda == 'source-only':
            cfg['_base_'].append(
                f'_base_/datasets/{source}_to_{target}_{crop}.py')
        else:
            cfg['_base_'].append(
                f'_base_/datasets/uda_{source}_to_{target}_{crop}.py')
            cfg['_base_'].append(f'_base_/uda/{uda}.py')
        if 'dacs' in uda and plcrop:
            cfg.setdefault('uda', {})
            cfg['uda']['pseudo_weight_ignore_top'] = 15
            cfg['uda']['pseudo_weight_ignore_bottom'] = 120
        cfg['data'] = dict(
            samples_per_gpu=batch_size,
            workers_per_gpu=workers_per_gpu,
            train={})
        if 'dacs' in uda and rcs_T is not None:
            cfg = setup_rcs(cfg, rcs_T)

        # Setup optimizer and schedule
        if 'dacs' in uda:
            cfg['optimizer_config'] = None  # Don't use outer optimizer

        cfg['_base_'].extend(
            [f'_base_/schedules/{opt}.py', f'_base_/schedules/{schedule}.py'])
        cfg['optimizer'] = {'lr': lr}
        cfg['optimizer'].setdefault('paramwise_cfg', {})
        cfg['optimizer']['paramwise_cfg'].setdefault('custom_keys', {})
        opt_param_cfg = cfg['optimizer']['paramwise_cfg']['custom_keys']
        if pmult:
            opt_param_cfg['head'] = dict(lr_mult=10.)
        if 'mit' in backbone:
            opt_param_cfg['pos_block'] = dict(decay_mult=0.)
            opt_param_cfg['norm'] = dict(decay_mult=0.)

        # Setup runner
        cfg['runner'] = dict(type='IterBasedRunner', max_iters=iters)
        cfg['checkpoint_config'] = dict(
            by_epoch=False, interval=iters, max_keep_ckpts=1)
        cfg['evaluation'] = dict(interval=iters // 10, metric='mIoU')

        # Construct config name
        uda_mod = uda
        if 'dacs' in uda and rcs_T is not None:
            uda_mod += f'_rcs{rcs_T}'
        if 'dacs' in uda and plcrop:
            uda_mod += '_cpl'
        cfg['name'] = f'{source}2{target}_{uda_mod}_{architecture_mod}_' \
                      f'{backbone}_{schedule}'
        cfg['exp'] = id
        cfg['name_dataset'] = f'{source}2{target}'
        cfg['name_architecture'] = f'{architecture_mod}_{backbone}'
        cfg['name_encoder'] = backbone
        cfg['name_decoder'] = architecture_mod
        cfg['name_uda'] = uda_mod
        cfg['name_opt'] = f'{opt}_{lr}_pm{pmult}_{schedule}' \
                          f'_{n_gpus}x{batch_size}_{iters // 1000}k'
        if seed is not None:
            cfg['name'] += f'_s{seed}'
        cfg['name'] = cfg['name'].replace('.', '').replace('True', 'T') \
            .replace('False', 'F').replace('cityscapes', 'cs') \
            .replace('synthia', 'syn') \
            .replace('darkzurich', 'dzur')
        return cfg

    # -------------------------------------------------------------------------
    # Set some defaults
    # -------------------------------------------------------------------------
    cfgs = []
    n_gpus = 1
    batch_size = 2
    iters = 40000
    opt, lr, schedule, pmult = 'adamw', 0.00006, 'poly10warm', True
    crop = '512x512'
    datasets = [
        ('gta', 'cityscapes'),
    ]
    architecture = None
    workers_per_gpu = 4
    rcs_T = None
    plcrop = False
    # -------------------------------------------------------------------------
    # UDA Architecture Comparison (Table 1)
    # -------------------------------------------------------------------------
    if id == 1:
        seeds = [0, 1, 2]
        models = [
            # Note: For the DeepLabV2 decoder, we follow AdaptSegNet as well as
            # many follow-up works using the same source code for the network
            # architecture (e.g. DACS or ProDA) and use only the dilation rates
            # 6 and 12. We point this out as it is hidden in the source code by
            # a return statement within a loop:
            # https://github.com/wasidennis/AdaptSegNet/blob/fca9ff0f09dab45d44bf6d26091377ac66607028/model/deeplab.py#L116
            ('dlv2red', 'r101v1c'),
            # Note: For the decoders used in combination with CNN encoders, we
            # do not apply BatchNorm in the *decoder* as it decreases
            # the UDA performance. In the encoder, BatchNorm is still applied.
            # The decoder of DeepLabV2 has no BatchNorm layer by default.
            ('da_nodbn', 'r101v1c'),
            ('isa_nodbn', 'r101v1c'),
            ('dlv3p_nodbn', 'r101v1c'),
            ('segformer', 'mitb5'),
        ]
        udas = [
            'source-only',
            'dacs',
            'target-only',
        ]
        for (source, target), (architecture, backbone), uda, seed in \
                itertools.product(datasets, models, udas, seeds):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # SegFormer Encoder / Decoder Ablation (Table 2)
    # -------------------------------------------------------------------------
    elif id == 2:
        seeds = [0, 1, 2]
        models = [
            # ('segformer', 'mitb5'),  # already run in exp 1
            ('sfa_dlv3p_nodbn', 'mitb5-del'),
            ('segformer', 'r101v1c'),
            # ('dlv3p_nodbn', 'r101v1c'),  # already run in exp 1
        ]
        udas = [
            'dacs',
            'target-only',
        ]
        for (source, target), (architecture, backbone), uda, seed in \
                itertools.product(datasets, models, udas, seeds):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Encoder Study (Table 3)
    # -------------------------------------------------------------------------
    elif id == 3:
        seeds = [0]
        models = [
            ('dlv2red', 'r50v1c'),
            # ('dlv2red', 'r101v1c'),  # already run in exp 1
            ('dlv2red', 's50'),
            ('dlv2red', 's101'),
            ('dlv2red', 's200'),
            ('segformer', 'mitb3'),
            ('segformer', 'mitb4'),
            # ('segformer', 'mitb5'),  # already run in exp 1
        ]
        udas = [
            'source-only',
            'dacs',
            'target-only',
        ]
        for (source, target), (architecture, backbone), uda, seed in \
                itertools.product(datasets, models, udas, seeds):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Learning Rate Warmup Ablation (Table 4)
    # -------------------------------------------------------------------------
    elif id == 4:
        seeds = [0]
        models = [
            ('dlv2red', 'r101v1c'),
            ('segformer', 'mitb5'),
        ]
        udas = ['dacs', 'target-only']
        opts = [
            ('adamw', 0.00006, 'poly10', True),
            # ('adamw', 0.00006, 'poly10warm', True),  # already run in exp 1
        ]
        for (source, target), (architecture, backbone), \
            (opt, lr, schedule, pmult), uda, seed in \
                itertools.product(datasets, models, opts, udas, seeds):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # RCS and FD (Table 5)
    # -------------------------------------------------------------------------
    elif id == 5:
        seeds = [0, 1, 2]
        for architecture, backbone, uda, rcs_T, plcrop in [
            ('segformer', 'mitb5', 'dacs', math.inf, False),
            ('segformer', 'mitb5', 'dacs', 0.01, False),
            ('segformer', 'mitb5', 'dacs_fd', None, False),
            ('segformer', 'mitb5', 'dacs_fdthings', None, False),
            ('segformer', 'mitb5', 'dacs_fdthings', 0.01, False),
            ('segformer', 'mitb5', 'dacs_a999_fdthings', 0.01, True),
            ('dlv2red', 'r101v1c', 'dacs_a999_fdthings', 0.01, True),
        ]:
            for (source, target), seed in \
                    itertools.product(datasets, seeds):
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Decoder Study (Table 7)
    # -------------------------------------------------------------------------
    elif id == 6:
        seeds = [0, 1, 2]
        udas = [
            'dacs_a999_fdthings',
            'target-only',
        ]
        rcs_T = 0.01
        plcrop = True
        models = [
            # ('segformer', 'mitb5'),  # already run in exp 5
            ('daformer_conv1', 'mitb5'),  # this is segformer with 256 channels
            ('upernet', 'mitb5'),
            ('upernet_ch256', 'mitb5'),
            ('daformer_isa', 'mitb5'),
            ('daformer_sepaspp_bottleneck', 'mitb5'),  # Context only at F4
            ('daformer_aspp', 'mitb5'),  # DAFormer w/o DSC
            ('daformer_sepaspp', 'mitb5'),  # DAFormer
        ]
        for (source, target), (architecture, backbone), uda, seed in \
                itertools.product(datasets, models, udas, seeds):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Final DAFormer (Table 6)
    # -------------------------------------------------------------------------
    elif id == 7:
        seeds = [0, 1, 2]
        datasets = [
            # ('gta', 'cityscapes'),  # already run in exp 6
            ('synthia', 'cityscapes'),
        ]
        architecture, backbone = ('daformer_sepaspp', 'mitb5')
        uda = 'dacs_a999_fdthings'
        rcs_T = 0.01
        plcrop = True
        for (source, target), seed in \
                itertools.product(datasets, seeds):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Further Datasets
    # -------------------------------------------------------------------------
    elif id == 8:
        seeds = [0, 1, 2]
        datasets = [
            ('cityscapes', 'acdc'),
            ('cityscapes', 'darkzurich'),
        ]
        architecture, backbone = ('daformer_sepaspp', 'mitb5')
        uda = 'dacs_a999_fdthings'
        rcs_T = 0.01
        plcrop = True
        for (source, target), seed in \
                itertools.product(datasets, seeds):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Architecture Startup Test
    # -------------------------------------------------------------------------
    elif id == 100:
        iters = 2
        seeds = [0]
        models = [
            ('dlv2red', 'r101v1c'),
            ('dlv3p_nodbn', 'r101v1c'),
            ('da_nodbn', 'r101v1c'),
            ('segformer', 'mitb5'),
            ('isa_nodbn', 'r101v1c'),
            ('dlv2red', 'r50v1c'),
            ('dlv2red', 's50'),
            ('dlv2red', 's101'),
            ('dlv2red', 's200'),
            ('dlv2red', 'x50-32'),
            ('dlv2red', 'x101-32'),
            ('segformer', 'mitb4'),
            ('segformer', 'mitb3'),
            ('sfa_dlv3p_nodbn', 'mitb5-del'),
            ('segformer', 'r101v1c'),
            ('daformer_conv1', 'mitb5'),
            ('daformer_isa', 'mitb5'),
            ('daformer_sepaspp_bottleneck', 'mitb5'),
            ('daformer_aspp', 'mitb5'),
            ('daformer_sepaspp', 'mitb5'),
            ('upernet', 'mitb5'),
            ('upernet_ch256', 'mitb5'),
        ]
        udas = ['target-only']
        for (source, target), (architecture, backbone), uda, seed in \
                itertools.product(datasets, models, udas, seeds):
            cfg = config_from_vars()
            cfg['log_level'] = logging.ERROR
            cfg['evaluation']['interval'] = 100
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # UDA Training Startup Test
    # -------------------------------------------------------------------------
    elif id == 101:
        iters = 2
        seeds = [0]
        for architecture, backbone, uda, rcs_T, plcrop in [
            ('segformer', 'mitb5', 'source-only', None, False),
            ('segformer', 'mitb5', 'target-only', None, False),
            ('segformer', 'mitb5', 'dacs', None, False),
            ('segformer', 'mitb5', 'dacs', math.inf, False),
            ('segformer', 'mitb5', 'dacs', 0.01, False),
            ('segformer', 'mitb5', 'dacs_fd', None, False),
            ('segformer', 'mitb5', 'dacs_fdthings', None, False),
            ('segformer', 'mitb5', 'dacs_fdthings', 0.01, False),
            ('segformer', 'mitb5', 'dacs_a999_fdthings', 0.01, True),
        ]:
            for (source, target), seed in \
                    itertools.product(datasets, seeds):
                cfg = config_from_vars()
                cfg['log_level'] = logging.ERROR
                cfg['evaluation']['interval'] = 100
                cfgs.append(cfg)
    else:
        raise NotImplementedError('Unknown id {}'.format(id))

    return cfgs
