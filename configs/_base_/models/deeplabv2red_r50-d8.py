# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = ['deeplabv2_r50-d8.py']
# Previous UDA methods only use the dilation rates 6 and 12 for DeepLabV2.
# This might be a bit hidden as it is caused by a return statement WITHIN
# a loop over the dilation rates:
# https://github.com/wasidennis/AdaptSegNet/blob/fca9ff0f09dab45d44bf6d26091377ac66607028/model/deeplab.py#L116
model = dict(decode_head=dict(dilations=(6, 12)))
