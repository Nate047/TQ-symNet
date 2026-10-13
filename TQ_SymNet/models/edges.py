"""V58 explicit Sym conditioning and controlled single-edge supervision."""
import math
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from mmdet.models.builder import DETECTORS
from mmdet.core import bbox_overlaps
from .attention import CandidateQualityAttention, gaussian_bias
from .selection import (RetinaNAMHighQualitySelector,
                                      bounded_score, balanced_mean)


def pair_generator(seed, epoch, identity, device):
    digest=hashlib.sha256(('%s|%s|%s'%(seed,epoch,identity)).encode()).digest()
    generator=torch.Generator(device=device)
    generator.manual_seed(int.from_bytes(digest[:8],'little') % (2**63-1))
    return generator


def edge_bias(boxes, positions):
    """24 tokens: four edges, inner/outer strips, three along-edge slots."""
    center = (boxes[:, :2] + boxes[:, 2:]) * .5
    size = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-4)
    relative = (positions[None] - center[:, None]) / size[:, None]
    biases = []
    for edge in range(4):  # left, top, right, bottom
        axis = edge % 2
        sign = -1. if edge < 2 else 1.
        for offset in (-.06, .06):  # inner and outer, relative to edge sign
            normal_center = sign * (.5 + offset)
            for along in (-.30, 0., .30):
                normal = (relative[..., axis] - normal_center) / .08
                tangent = (relative[..., 1-axis] - along) / .20
                biases.append(-.5 * (normal.square() + tangent.square()))
    return torch.stack(biases, 1).clamp_min(-40.)


class EdgeQualityAttention(CandidateQualityAttention):
    def __init__(self, hidden=96, heads=6, edges=False, **kwargs):
        super().__init__(hidden=hidden, heads=heads, **kwargs)
        self.edges = edges
        self.attention_stats = []
        roles = 28 if edges else 4
        with torch.random.fork_rng(devices=[]):
            if edges:
                original_tokens=self.query_tokens.detach().clone()
                self.query_tokens = nn.Parameter(torch.cat((original_tokens,
                    torch.randn(24,hidden)*.02)))
            self.quality = nn.Sequential(nn.LayerNorm(hidden*(roles+1)),
                nn.Linear(hidden*(roles+1), hidden*2), nn.SiLU(),
                nn.Linear(hidden*2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            self.high_head = nn.Linear(hidden, 1)
            self.residual_head = nn.Sequential(nn.Linear(hidden+2, hidden),
                nn.SiLU(), nn.Linear(hidden, 1))
        for output in (self.high_head, self.residual_head[-1]):
            nn.init.zeros_(output.weight); nn.init.zeros_(output.bias)

    def _attention(self, query, boxes, key, value, positions, projection, gaussian_branch):
        if not self.edges:
            return super()._attention(query, boxes, key, value, positions,
                                      projection, gaussian_branch)
        n, roles, hidden = query.shape
        dim = hidden // self.heads
        q = self.q_projection[projection](self.query_norm[projection](query)).reshape(n,roles,self.heads,dim)
        k, v = key.reshape(-1,self.heads,dim), value.reshape(-1,self.heads,dim)
        similarity = torch.einsum('nqhd,thd->nhqt',q,k)/math.sqrt(dim)
        strength = (2*self.gaussian_raw[gaussian_branch].sigmoid()
                    if self.gaussian else query.new_tensor(0.))
        bias = torch.cat((gaussian_bias(boxes,positions,strength),
                          edge_bias(boxes,positions)*strength),1)
        valid = ((positions >= 0)&(positions <= 1)).all(1)
        bias = bias.masked_fill(~valid[None,None],-1e4)
        attention = (similarity+bias[:,None]).softmax(-1)
        if not self.training:
            with torch.no_grad():
                region=edge_bias(boxes,positions)>-2.
                mass=(attention[:,:,4:]*region[:,None]).sum(-1).mean((0,1))
                self.attention_stats.append((mass.cpu(),float(strength.detach())))
        selected = torch.einsum('nhqt,thd->nqhd',attention,v).reshape(n,roles,hidden)
        entropy = -(attention*attention.clamp_min(1e-8).log()).sum(-1).mean((1,2))
        return query+self.output_projection[projection](selected),entropy

    def forward(self, boxes, memories, sym, training_checkpoint=False):
        if not len(boxes):
            return boxes.new_empty((0,2)),boxes.new_empty((0,7)),boxes.new_empty((0,3))
        descriptor,gate,entropy = super().forward(boxes,memories,training_checkpoint)
        # Only the residual route consumes Sym: quality remains independent.
        sym_logits = torch.logit(sym[:,:2].clamp(1e-6,1.-1e-6)).detach()
        residual = self.residual_head(torch.cat((descriptor,sym_logits),1))
        return torch.cat((self.high_head(descriptor),residual),1),gate,entropy


class RetinaNAMEdgeSelector(RetinaNAMHighQualitySelector):
    def __init__(self,*args,selector_edges=False,selector_edge_weight=0.,**kwargs):
        super().__init__(*args,**kwargs)
        previous = self.candidate_quality
        with torch.random.fork_rng(devices=[]):
            self.candidate_quality = EdgeQualityAttention(hidden=previous.hidden,
                heads=previous.heads, texture=previous.texture, gaussian=previous.gaussian,
                chunk=previous.chunk,checkpoint_candidates=previous.checkpoint_candidates,
                edges=bool(selector_edges), lbp_enabled=previous.lbp_enabled,
                gabor_enabled=previous.gabor_enabled)
        self.selector_edge_weight = float(selector_edge_weight)
        self.selector_list_rank = True
        self.selector_pair_seed = torch.initial_seed()
        self.integrated_build_id = 'v58.1-isolated-rng-continuous-residual-fullgt-debug'
        self.edge_stats = []
        self.residual_stats = []

    def _residual_loss(self,base_z,residual,target,actual,matched):
        # Optimize the SAME final score used by ranking and inference. Use
        # continuous IoU, not a clipped common +.8 target for all good boxes.
        final_z=base_z+self.selector_beta*residual.tanh()
        desired_z=torch.logit(actual.clamp(.001,.999))
        blocked=(desired_z-base_z).abs()>self.selector_beta
        self.residual_stats.append([int(blocked.sum()),len(actual),
            int(((actual>=.85)&blocked).sum()),int((actual>=.85).sum())])
        return balanced_mean(F.binary_cross_entropy_with_logits(final_z,
            actual.detach(),reduction='none'),actual,matched)

    def _score_candidates(self,boxes,memory,sym,training=False):
        return self.candidate_quality(boxes,memory,sym,training_checkpoint=training)

    def v50_begin_oracle(self,epoch):
        self.candidate_quality.attention_stats=[]
        return super().v50_begin_oracle(epoch)

    def v50_finish_oracle(self,dataset,runner):
        stats=self.candidate_quality.attention_stats
        if stats:
            mass=torch.stack([x[0] for x in stats]).mean(0).reshape(4,2,3)
            runner.logger.info('[SELECTOR-EDGE-ATTENTION] chunkMeanMassWithin2Sigma '
                'left/top/right/bottom_inner_outer_slots=%s meanStrength=%.5f',
                mass.tolist(),sum(x[1] for x in stats)/len(stats))
        self.candidate_quality.attention_stats=[]
        return super().v50_finish_oracle(dataset,runner)

    def _extra_training_loss(self,boxes,sym,gt,matched,actual,memory,meta):
        zero = self.candidate_quality.high_head.weight.sum()*0.
        if self.selector_edge_weight <= 0 or not len(gt):
            return zero
        eligible = (actual >= .5).nonzero(as_tuple=False).squeeze(1)
        budget = min(len(eligible),max(1,len(boxes)//4))
        if not budget:
            return zero
        identity=meta.get('ori_filename',meta.get('filename','unknown'))
        identity += '|'+str(meta.get('flip',False))
        generator=pair_generator(self.selector_pair_seed,getattr(self,'_nam_epoch',0),
                                 identity,boxes.device)
        seeds=[]
        for g in matched[eligible].unique():
            candidates=eligible[matched[eligible]==g]
            seeds.append(candidates[actual[candidates].argmax()])
        seeds=torch.stack(seeds) if seeds else eligible[:0]
        # Every represented GT gets its best box when budget permits.
        if len(seeds)>budget:
            seeds=seeds[torch.randperm(len(seeds),device=boxes.device,generator=generator)[:budget]]
        available=torch.ones(len(boxes),device=boxes.device,dtype=torch.bool)
        available[seeds]=False
        rest=eligible[available[eligible]]
        high=rest[actual[rest]>=.85]
        high=high[torch.randperm(len(high),device=boxes.device,generator=generator)[:(budget-len(seeds)+1)//2]]
        available[high]=False
        rest=eligible[available[eligible]]
        random=rest[torch.randperm(len(rest),device=boxes.device,generator=generator)[:budget-len(seeds)-len(high)]]
        parents=torch.cat((seeds,high,random))
        budget=len(parents)
        changed = boxes[parents].clone()
        edge = torch.randint(4,(budget,),device=boxes.device,generator=generator)
        magnitudes = boxes.new_tensor([-.10,-.05,-.02,.02,.05,.10])
        shift = magnitudes[torch.randint(6,(budget,),device=boxes.device,generator=generator)]
        size = boxes[parents,2:]-boxes[parents,:2]
        changed[torch.arange(budget,device=boxes.device),edge] += shift*size[
            torch.arange(budget,device=boxes.device),edge%2]
        h,w = meta['img_shape'][:2]
        changed[:,0::2].clamp_(0,w);changed[:,1::2].clamp_(0,h)
        valid = ((changed[:,2:]-changed[:,:2])>1).all(1)
        overlap = bbox_overlaps(changed,gt)
        new_iou,new_gt = overlap.max(1)
        valid &= new_gt == matched[parents]
        valid &= (new_iou-actual[parents]).abs() > .02
        attempted=budget
        attempted85=int((actual[parents]>=.85).sum())
        if not valid.any():
            self.edge_stats.append(dict(count=0,correct=0,directions=[0]*4,
                correct_directions=[0]*4,attempted=attempted,attempted85=attempted85,accepted85=0))
            return zero
        parents,changed,new_iou,edge = parents[valid],changed[valid],new_iou[valid],edge[valid]
        # Same Sym for both sides of a controlled pair: no invented detector
        # score for a synthetic box, and no artificial Sym difference shortcut.
        scores = self._score_candidates(self._normalized(torch.cat((boxes[parents],changed)),meta),
            memory,torch.cat((sym[parents],sym[parents])),True)[0][:,1]
        base = torch.logit(sym[parents,:-1].max(1)[0].clamp(1e-6,1.-1e-6))
        count = len(parents)
        if getattr(self,'selector_quality_rank_only',False):
            first,second=scores[:count],scores[count:]
        else:
            first = base+self.selector_beta*scores[:count].tanh()
            second = base+self.selector_beta*scores[count:].tanh()
        difference = new_iou-actual[parents]
        margin = .5*difference.abs()
        signed = difference.sign()*(second-first)
        losses = F.softplus(margin-signed)
        terms = [losses[matched[parents]==g].mean() for g in matched[parents].unique()]
        self.edge_stats.append(dict(count=count,correct=int((signed.detach()>0).sum()),
            attempted=attempted,attempted85=attempted85,
            accepted85=int((actual[parents]>=.85).sum()),
            directions=[int((edge==k).sum()) for k in range(4)],
            correct_directions=[int(((edge==k)&(signed.detach()>0)).sum()) for k in range(4)]))
        return self.selector_edge_weight*torch.stack(terms).mean()
