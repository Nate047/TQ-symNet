"""GFocalV2 DGQP adapted to the project's MMDet 2.x GFL logits API.

See third_party/gfocalv2_reference/SOURCE.json for the preserved source,
mirror revision and hash. This is an adaptation, not upstream COCO replay.
Assignment, DFL, Integral, regression weighting, reduction and batched decode
are inherited from the bundled OpenMMLab GFLHead.
"""
from torch import nn
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.gfl_head import GFLHead
from TQ_SymNet.supplement_math import dgqp_joint_logit


@HEADS.register_module()
class TQGFLV2Head(GFLHead):
    def __init__(self, *args, reg_topk=4, reg_channels=64, add_mean=True, **kwargs):
        self.reg_topk = int(reg_topk)
        self.reg_channels = int(reg_channels)
        self.add_mean = bool(add_mean)
        reg_max = int(kwargs.get('reg_max', 16))
        if not 1 <= self.reg_topk <= reg_max+1 or self.reg_channels < 1:
            raise ValueError('Invalid DGQP dimensions')
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        super()._init_layers()
        channels = 4*(self.reg_topk + int(self.add_mean))
        self.reg_conf = nn.Sequential(
            nn.Conv2d(channels, self.reg_channels, 1), nn.ReLU(inplace=True),
            nn.Conv2d(self.reg_channels, 1, 1), nn.Sigmoid())

    def forward_single(self, x, scale):
        cls_logit, distribution = super().forward_single(x, scale)
        joint = dgqp_joint_logit(cls_logit, distribution, self.reg_conf,
                                self.reg_max, self.reg_topk, self.add_mean)
        return joint, distribution
