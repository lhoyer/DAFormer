# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

from .base_pixel_sampler import BasePixelSampler
from .ohem_pixel_sampler import OHEMPixelSampler

__all__ = ['BasePixelSampler', 'OHEMPixelSampler']
