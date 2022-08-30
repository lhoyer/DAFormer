# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# ISA Fusion in Tab. 7

_base_ = ['daformer_conv1_mitb5.py']

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    decode_head=dict(
        decoder_params=dict(
            fusion_cfg=dict(
                _delete_=True,
                type='isa',
                isa_channels=256,
                key_query_num_convs=1,
                down_factor=(8, 8),
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg))))
