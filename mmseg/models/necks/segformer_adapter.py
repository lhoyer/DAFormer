# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
import torch.nn as nn

from mmseg.ops import resize
from ..builder import NECKS


@NECKS.register_module()
class SegFormerAdapter(nn.Module):

    def __init__(self, out_layers=[3], scales=[4]):
        super(SegFormerAdapter, self).__init__()
        self.out_layers = out_layers
        self.scales = scales

    def forward(self, x):
        _c = {}
        for i, s in zip(self.out_layers, self.scales):
            if s == 1:
                _c[i] = x[i]
            else:
                _c[i] = resize(
                    x[i], scale_factor=s, mode='bilinear', align_corners=False)
            # mmcv.print_log(f'{i}: {x[i].shape}, {_c[i].shape}', 'mmseg')

        x[-1] = torch.cat(list(_c.values()), dim=1)
        return x
