"""Exercise disabled routes, checkpoint backward, empty boxes and checkpoint isolation."""
import copy
import math
import sys
from pathlib import Path
from check_math import definitions

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint
    from TQ_SymNet.checkpoints import load_detector_state
    folder = ROOT / 'TQ_SymNet/models'
    scope = dict(torch=torch, nn=nn, F=F, math=math, checkpoint=checkpoint)
    definitions(folder / 'attention.py', scope)
    definitions(folder / 'edges.py', scope, {'edge_bias', 'EdgeQualityAttention'})
    definitions(folder / 'detail.py', scope, {'DetailResidualBlock', 'NativeDetailEncoder',
                                             'edge_anchor_points', 'DetailQualityAttention'})
    definitions(folder / 'light_quality.py', scope, {'PooledQualityScorer'})
    def overlaps(a, b):
        inter = (torch.minimum(a[:,None,2:],b[None,:,2:])-
                 torch.maximum(a[:,None,:2],b[None,:,:2])).clamp_min(0).prod(-1)
        area_a=(a[:,2:]-a[:,:2]).clamp_min(0).prod(-1)
        area_b=(b[:,2:]-b[:,:2]).clamp_min(0).prod(-1)
        return inter/(area_a[:,None]+area_b[None]-inter).clamp_min(1e-6)
    scope['bbox_overlaps'] = overlaps
    definitions(folder / 'selection.py', scope,
                {'all_iou_target', 'hard_select', 'gaussian_soft_select'})
    cases = dict(full={}, no_texture=dict(texture=False), no_lbp=dict(lbp_enabled=False),
                 no_gabor=dict(gabor_enabled=False), no_detail=dict(detail=False),
                 no_side=dict(edges=False, detail=False), no_gaussian=dict(gaussian=False))
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    for device in devices:
        for name, change in cases.items():
            torch.manual_seed(19)
            options=dict(hidden=16,heads=4,chunk=2,texture=True,gaussian=True,
                         edges=True,detail=True,checkpoint_candidates=True)
            options.update(change)
            model=scope['DetailQualityAttention'](**options).to(device).train()
            with torch.no_grad(): model.high_head.weight.normal_(std=.03)
            reference=copy.deepcopy(model)
            maps=[torch.rand(1,256,h,h,device=device) for h in (8,4,2,1)]
            gray=torch.rand(1,1,32,32,device=device)
            boxes=torch.tensor([[.1,.1,.9,.9],[.2,.2,.6,.7],[.45,.55,.8,.95]],device=device)
            sym=torch.tensor([[.8,.2,0],[.2,.7,0],[.4,.3,0]],device=device)
            memories=model.prepare(maps,gray,(28,30),(32,32))
            expected_memory=reference.prepare(maps,gray,(28,30),(32,32))
            actual=model(boxes,memories,sym,training_checkpoint=True)
            expected=reference(boxes,expected_memory,sym,training_checkpoint=False)
            torch.testing.assert_close(actual[0],expected[0])
            target=boxes.new_tensor([.9,.3,.65])
            F.binary_cross_entropy_with_logits(actual[0][:,0],target).backward()
            F.binary_cross_entropy_with_logits(expected[0][:,0],target).backward()
            for (an,ap),(bn,bp) in zip(model.named_parameters(),reference.named_parameters()):
                assert an == bn
                assert (ap.grad is None)==(bp.grad is None), (name,an)
                if ap.grad is not None:
                    assert torch.isfinite(ap.grad).all(), (name,an)
                    torch.testing.assert_close(ap.grad,bp.grad,rtol=1e-4,atol=1e-6)
            if not model.lbp_enabled:
                assert all(p.grad is None for n,p in model.named_parameters() if n.startswith(('lbp.','lbp_encoder.')))
                assert torch.equal(actual[1][:,1], torch.zeros_like(actual[1][:,1]))
            if not model.gabor_enabled:
                assert all(p.grad is None for n,p in model.named_parameters() if n.startswith(('gabor.','gabor_encoder.')))
                assert torch.equal(actual[1][:,2], torch.zeros_like(actual[1][:,2]))
            empty=model(boxes[:0],memories,sym[:0],training_checkpoint=True)
            assert empty[0].shape==(0,2)
            print('PASS',device,name,'forward/backward/checkpoint/empty')
        light=scope['PooledQualityScorer'](hidden=16,chunk=2).to(device).train()
        with torch.no_grad(): light.high_head.weight.normal_(std=.03)
        memory=light.prepare(maps,None,(28,30),(32,32))
        prediction=light(boxes,memory,sym)[0]
        prediction[:,0].square().mean().backward()
        assert light.visual_projection[0].weight.grad is not None
        assert light(boxes[:0],memory,sym[:0])[0].shape==(0,2)
        data=boxes.new_tensor([[0,0,1,1],[0,0,1,1],[2,2,3,3]])
        scores=boxes.new_tensor([.7,.9,.6])
        hard=scope['hard_select'](data,scores)[0]
        soft=scope['gaussian_soft_select'](data,scores)[0]
        assert hard.tolist()==[1,2] and soft.tolist()==[1,2,0]
        assert scope['hard_select'](data[:0],scores[:0])[0].numel()==0
        target=scope['all_iou_target'](torch.linspace(0,1,101,device=device))
        assert (target[1:]>target[:-1]).all()
        print('PASS',device,'light scorer, hard/soft selection and target monotonicity')
    toy=nn.Module()
    for name in ('backbone','neck','bbox_head','candidate_quality'):
        setattr(toy,name,nn.Linear(2,2))
    before={k:v.clone() for k,v in toy.candidate_quality.state_dict().items()}
    state={k:torch.full_like(v,9.) for k,v in toy.state_dict().items()}
    audit=load_detector_state(toy,state)
    for key,value in toy.candidate_quality.state_dict().items():
        assert torch.equal(value,before[key]), 'Quality pretraining leaked into matched ablation'
    assert (toy.backbone.weight==9).all()
    broken=dict(state); broken.pop('neck.weight')
    try: load_detector_state(toy,broken)
    except RuntimeError: pass
    else: raise AssertionError('Missing active detector weights were silently accepted')
    print('PASS checkpoint transfer isolates detector and rejects missing detector keys:', audit)


if __name__ == '__main__':
    main()
