from ..builder import HEADS
from .aspp_head import ASPPModule
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class DLV2Head(BaseDecodeHead):

    def __init__(self, dilations=(6, 12, 18, 24), **kwargs):
        assert 'channels' not in kwargs
        assert 'dropout_ratio' not in kwargs
        assert 'norm_cfg' not in kwargs
        kwargs['channels'] = 1
        kwargs['dropout_ratio'] = 0
        kwargs['norm_cfg'] = None
        super(DLV2Head, self).__init__(**kwargs)
        del self.conv_seg
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.num_classes,
            conv_cfg=self.conv_cfg,
            norm_cfg=None,
            act_cfg=None)

    def forward(self, inputs):
        """Forward function."""
        # for f in inputs:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')
        x = self._transform_inputs(inputs)
        aspp_outs = self.aspp_modules(x)
        out = aspp_outs[0]
        for i in range(len(aspp_outs) - 1):
            out += aspp_outs[i + 1]
        return out
