# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# UDA with ImageNet Feature Distance
_base_ = ['dacs.py']
uda = dict(imnet_feature_dist_lambda=0.005, )
