# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# UDA with Thing-Class ImageNet Feature Distance
_base_ = ['dacs.py']
uda = dict(
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_method='ratio',
    imnet_feature_dist_scale_min_ratio=0.75,
)
