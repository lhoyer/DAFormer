# UDA with ImageNet Feature Distance
_base_ = ['dacs.py']
uda = dict(imnet_feature_dist_lambda=0.005, )
