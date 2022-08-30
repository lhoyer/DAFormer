# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

from .builder import build_pixel_sampler
from .sampler import BasePixelSampler, OHEMPixelSampler

__all__ = ['build_pixel_sampler', 'BasePixelSampler', 'OHEMPixelSampler']
