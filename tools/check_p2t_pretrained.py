"""CPU regression of strict P2T loading using the real extracted backbone code.

MMCV registry/BaseModule are minimal adapters here; full detector initialization
must also pass tools/preflight.py --server-files --build on the server.
"""
from functools import partial
import logging
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from check_math import definitions
from TQ_SymNet.pretrained import load_checked_pretrained


def main():
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(2)

    class BaseModule(nn.Module):
        def __init__(self,init_cfg=None):
            super().__init__()
            self.init_cfg=init_cfg
            self._is_init=False

    class DropPath(nn.Module):
        def __init__(self,prob=0.):
            super().__init__()
        def forward(self,x):
            return x  # Regression compares eval mode only.

    scope=dict(torch=torch,nn=nn,F=F,np=np,partial=partial,BaseModule=BaseModule,
               DropPath=DropPath,to_2tuple=lambda x:(x,x),trunc_normal_=nn.init.trunc_normal_,
               get_root_logger=lambda:logging.getLogger('p2t-check'))
    definitions(ROOT/'mmdet/models/backbones/p2t.py',scope)
    scope['load_checked_pretrained']=load_checked_pretrained
    scope['Path']=Path
    definitions(ROOT/'TQ_SymNet/models/p2t_checked.py',scope)
    torch.manual_seed(0)
    reference=scope['p2t_small'](pretrained=None).eval()
    expected={k:v.detach().clone() for k,v in reference.state_dict().items()}
    assert scope['TQP2TSmall'].forward is scope['p2t_small'].forward
    folder=ROOT/'work_dirs/_p2t_loader_regression'
    folder.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='case-',dir=str(folder)) as temporary:
        path=Path(temporary)/'p2t_fixture.pth'
        checkpoint={'module.'+k:v for k,v in expected.items()}
        checkpoint['module.head.weight']=torch.zeros(1000,512)
        checkpoint['module.head.bias']=torch.zeros(1000)
        torch.save({'model':checkpoint},path)
        model=scope['TQP2TSmall'](pretrained=str(path)).eval()
        assert model.pretrained_audit['loaded_fraction']==1.
        assert model.pretrained_audit['layout']=='model'
        model.init_weights()
        for k,v in model.state_dict().items():
            assert torch.equal(v,expected[k]),k
        with torch.no_grad():
            x=torch.randn(1,3,128,128)
            actual,ref=model(x),reference(x)
        assert [list(v.shape) for v in actual]==[[1,64,32,32],[1,128,16,16],[1,320,8,8],[1,512,4,4]]
        for a,b in zip(actual,ref):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        for layout in ('state_dict','plain'):
            torch.save({layout:expected} if layout!='plain' else expected,path)
            load_checked_pretrained(model,path)
        broken=dict(expected)
        broken.pop('patch_embed1.proj.weight')
        torch.save({'model':broken},path)
        try:
            load_checked_pretrained(model,path)
        except RuntimeError as error:
            assert 'missing=' in str(error)
        else:
            raise AssertionError('Missing backbone weights accepted')
    print('PASS: official P2T eval features exactly preserved; model/state_dict/plain/DDP layouts; '
          '100% weight coverage; no reset on init_weights(); missing-key rejection')
    print('NOT TESTED: actual server checkpoint or full MMCV detector. Run preflight with --server-files --build.')


if __name__=='__main__':
    main()
