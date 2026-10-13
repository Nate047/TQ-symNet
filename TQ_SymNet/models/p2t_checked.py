"""Official P2T-Small computation with explicit, fully audited pretraining load."""
from pathlib import Path

from mmdet.models.backbones.p2t import p2t_small
from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger
from TQ_SymNet.pretrained import load_checked_pretrained


@BACKBONES.register_module()
class TQP2TSmall(p2t_small):
    def __init__(self, pretrained=None, init_cfg=None, **kwargs):
        if init_cfg is not None:
            if not isinstance(init_cfg,dict) or init_cfg.get('type') != 'Pretrained':
                raise ValueError('P2T adapter accepts only a Pretrained init_cfg')
            configured = init_cfg.get('checkpoint')
            if pretrained and configured and Path(pretrained).resolve()!=Path(configured).resolve():
                raise ValueError('P2T pretrained and init_cfg paths disagree')
            pretrained = pretrained or configured
        # Preserve the official architecture, initial RNG draws and forward path.
        super().__init__(pretrained=None,init_cfg=None,**kwargs)
        self.pretrained_audit = None
        if pretrained is not None:
            self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            self.pretrained_audit = load_checked_pretrained(self,pretrained,get_root_logger())
        # Parent detector initialization must not overwrite constructor-loaded weights.
        self._is_init = True
