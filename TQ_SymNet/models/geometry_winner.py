"""F2 + signed side supervision + good-set choice; outputs stay unmodified."""
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from mmcv.runner import HOOKS, Hook
from mmdet.models.builder import DETECTORS
from mmdet.core import bbox_overlaps

from ..geometry_winner_math import (winner_sets, winner_set_loss,
    signed_geometry_targets, signed_geometry_loss)
from .roi_query import ROIContentQualityAttention, TQSymNetROIQuery


class SignedGeometryQualityAttention(ROIContentQualityAttention):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with torch.random.fork_rng(devices=[]):
            # Per side: six ordered roles from visual/LBP/Gabor, D2 and D4.
            # Never reduce all sides to the final96 descriptor before supervision.
            self.signed_aux=nn.Sequential(nn.LayerNorm(18*self.hidden+12),
                nn.Linear(18*self.hidden+12,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.signed_aux[-1].weight)
        nn.init.zeros_(self.signed_aux[-1].bias)

    def _score(self,boxes,*inputs):
        body=inputs[:-2] if self.detail else inputs
        roles,geometry,gate,entropy=self._read_roles(boxes,*body)
        descriptor=self.quality(torch.cat((roles.flatten(1),geometry),1))
        if self.detail:
            edge_d2=self._local_attention(descriptor,boxes,inputs[-2],0)
            edge_d4=self._local_attention(descriptor,boxes,inputs[-1],1)
            descriptor=descriptor+self.detail_fusion(torch.cat(
                (descriptor,edge_d2.flatten(1),edge_d4.flatten(1)),1))
        else:
            edge_d2=edge_d4=torch.zeros_like(roles[:,4:])
        sides=torch.cat((roles[:,4:].reshape(len(boxes),4,-1),
                         edge_d2.reshape(len(boxes),4,-1),
                         edge_d4.reshape(len(boxes),4,-1)),2)
        center=(boxes[:,:2]+boxes[:,2:])*.5
        size=(boxes[:,2:]-boxes[:,:2]).clamp_min(1e-5)
        raw_geometry=torch.cat((boxes,center,size),1)[:,None].expand(-1,4,-1)
        side_id=torch.eye(4,device=boxes.device,dtype=boxes.dtype)[None].expand(len(boxes),-1,-1)
        offset=self.signed_aux(torch.cat((sides,raw_geometry,side_id),2)).squeeze(-1)
        return descriptor,gate,entropy,offset

    def forward(self,boxes,memories,sym,training_checkpoint=False,return_geometry=False):
        if not len(boxes):
            result=(boxes.new_empty((0,2)),boxes.new_empty((0,7)),boxes.new_empty((0,3)))
            return result+(boxes.new_empty((0,4)),) if return_geometry else result
        inputs=tuple(value for memory in memories for value in memory)
        outputs=[]
        for part in boxes.split(self.chunk):
            outputs.append(checkpoint(self._score,part,*inputs)
                if training_checkpoint and self.checkpoint_candidates else self._score(part,*inputs))
        descriptor,gate,entropy,offset=(torch.cat([x[k] for x in outputs]) for k in range(4))
        z=self.high_head(descriptor).squeeze(1)
        result=(torch.stack((z,z),1),gate,entropy)
        return result+(offset,) if return_geometry else result


def geometry_summary(prediction,target,valid,ambiguous,actual):
    with torch.no_grad():
        diff=(prediction.detach()-target).abs()
        meaningful=valid[:,None] & (target.abs()>=.02)
        sign_correct=((prediction.detach()*target)>0) & meaningful
        return torch.cat((torch.stack((actual.new_tensor(float(len(actual))),
            valid.sum().float(),ambiguous.sum().float(),(valid & (actual>=.85)).sum().float())),
            diff[valid].sum(0),sign_correct.sum(0).float(),meaningful.sum(0).float()))


@DETECTORS.register_module()
class TQSymNetGeometryWinner(TQSymNetROIQuery):
    def __init__(self,*args,geometry_weight=.05,geometry_beta=.05,
                 geometry_ambiguity_gap=.1,winner_temperature=.5,winner_max=4,**kwargs):
        super().__init__(*args,**kwargs)
        if not self.selector_quality_trainable:
            raise ValueError('Geometry winner requires trainable quality parameters')
        if (any(not math.isfinite(float(v)) or v<=0 for v in
                (geometry_weight,geometry_beta,winner_temperature)) or winner_max!=4):
            raise ValueError('Expected positive geometry/winner settings and up-to4 winners')
        previous=self.candidate_quality
        if not previous.paired_boundary or not previous.detail:
            raise ValueError('Geometry winner requires unchanged F2 paired/native-detail reading')
        with torch.random.fork_rng(devices=[]):
            q=SignedGeometryQualityAttention(hidden=previous.hidden,heads=previous.heads,
                texture=previous.texture,gaussian=previous.gaussian,chunk=previous.chunk,
                checkpoint_candidates=previous.checkpoint_candidates,edges=previous.edges,
                detail=previous.detail,lbp_enabled=previous.lbp_enabled,gabor_enabled=previous.gabor_enabled,
                paired_boundary=True,roi_pool_size=previous.roi_pool_size)
        q.fusion_gate_logit=nn.Parameter(previous.fusion_gate_logit.detach().clone())
        q.tb_lse_temperature_raw=nn.Parameter(previous.tb_lse_temperature_raw.detach().clone(),requires_grad=False)
        result=q.load_state_dict(previous.state_dict(),strict=False)
        if result.unexpected_keys or any(not k.startswith('signed_aux.') for k in result.missing_keys):
            raise RuntimeError('Geometry common-state transfer failed: %s'%(result,))
        self.candidate_quality=q
        self.geometry_weight=float(geometry_weight);self.geometry_beta=float(geometry_beta)
        self.geometry_ambiguity_gap=float(geometry_ambiguity_gap)
        if not 0<=self.geometry_ambiguity_gap<=1:
            raise ValueError('Invalid geometry ambiguity gap')
        self.winner_temperature=float(winner_temperature);self.winner_max=int(winner_max)
        self.geometry_train=[];self.winner_train=[];self.winner_sample=[];self.geometry_val=[]
        self.integrated_build_id='TQ-F2-signed-geometry-goodset-v1'
        self.active_rank_name='loss_winner_set'

    def _score_candidates(self,boxes,memory,sym,training=False):
        out,gate,entropy,delta=self.candidate_quality(boxes,memory,sym,
            training_checkpoint=training,return_geometry=True)
        if not self.candidate_quality.training and self.quality_collect:
            self.geometry_val.append(delta.detach().cpu())
        return out,gate,entropy

    def _training_quality_predictions(self,boxes,memory,sym):
        return self.candidate_quality(boxes,memory,sym,training_checkpoint=True,return_geometry=True)

    def _selector_sample(self,actual,matched,target,prediction,corrected,boxes):
        base=super()._selector_sample(actual,matched,target,prediction,corrected,boxes)
        groups=winner_sets(actual,matched,boxes,self.winner_max)
        sets=[base]+[ids for g in groups for ids in (g['high'],g['rivals'])]
        selected=torch.cat(sets).unique() if sets else base
        self.winner_sample.append([len(actual),len(base),len(selected),len(selected)-len(base)])
        return selected

    def _additional_supervised_loss(self,boxes,sym,gt,matched,actual,memory,meta,supervision):
        target,valid,ambiguous=signed_geometry_targets(boxes,gt,matched,actual,self.geometry_ambiguity_gap)
        self.geometry_train.append(geometry_summary(supervision,target,valid,ambiguous,actual))
        return self.geometry_weight*signed_geometry_loss(supervision,target,valid,matched,self.geometry_beta)

    def _replacement_rank_loss(self,logits,actual,matched,boxes):
        groups=winner_sets(actual,matched,boxes,self.winner_max)
        loss,count,stats=winner_set_loss(logits,groups,self.winner_temperature)
        self.winner_train.extend(stats)
        return loss,count

    def forward_train(self,*args,**kwargs):
        losses=super().forward_train(*args,**kwargs)
        losses['loss_signed_geometry']=losses.pop('loss_selector_edge_pair')
        losses['loss_winner_set']=losses.pop('loss_selector_competition')
        return losses

    def v50_begin_oracle(self,epoch):
        self.geometry_val=[]
        return super().v50_begin_oracle(epoch)

    def v50_finish_oracle(self,dataset,runner):
        if self.quality_collect:
            if len(self.geometry_val)!=len(self.quality_cache):
                raise RuntimeError('Geometry/validation box cache mismatch')
            summaries=[];ranking=[]
            high85=dict(groups=0,quality_hits=0,fused_hits=0)
            for index,(item,prediction) in enumerate(zip(self.quality_cache,self.geometry_val)):
                boxes=item['boxes']
                if len(prediction)!=len(boxes): raise RuntimeError('Geometry/candidate alignment mismatch')
                gt=torch.as_tensor(dataset.get_ann_info(index)['bboxes'],dtype=torch.float32).reshape(-1,4)
                if len(gt) and len(boxes): actual,matched=bbox_overlaps(boxes,gt).max(1)
                else:
                    actual=boxes.new_zeros(len(boxes));matched=torch.full((len(boxes),),-1,dtype=torch.long)
                target,valid,ambiguous=signed_geometry_targets(boxes,gt,matched,actual,self.geometry_ambiguity_gap)
                summaries.append(geometry_summary(prediction,target,valid,ambiguous,actual))
                groups=winner_sets(actual,matched,boxes,self.winner_max)
                _,_,stats=winner_set_loss(item['logit'],groups,self.winner_temperature)
                ranking.extend(stats)
                for g in matched[actual>=.5].unique():
                    ids=((matched==g)&(actual>=.5)).nonzero(as_tuple=False).squeeze(1)
                    if actual[ids].max()>=.85:
                        high85['groups']+=1
                        high85['quality_hits']+=int(actual[ids[item['logit'][ids].argmax()]]>=.85)
                        high85['fused_hits']+=int(actual[ids[item['corrected'][ids].argmax()]]>=.85)
            record=dict(epoch=self.quality_epoch,domain='full_pre_softnms_validation_pool',
                geometry=geometry_record(summaries),winner=winner_record(ranking),
                high85_positive_pool_top1=high85)
            (Path(runner.work_dir)/('geometry_winner_val_epoch_%02d.json'%self.quality_epoch)).write_text(
                json.dumps(record,indent=2),encoding='utf-8')
            runner.logger.info('[GEOMETRY-WINNER-VAL] %s',json.dumps(record))
        self.geometry_val=[]
        return super().v50_finish_oracle(dataset,runner)


def geometry_record(rows):
    if not rows: return dict(observations=0,eligible=0,ambiguous=0)
    v=torch.stack(rows).sum(0).cpu().tolist();n=max(v[1],1.)
    return dict(observations=int(v[0]),eligible=int(v[1]),ambiguous=int(v[2]),eligible85=int(v[3]),
        mae_left_top_right_bottom=[x/n for x in v[4:8]],
        sign_accuracy=[a/max(b,1) for a,b in zip(v[8:12],v[12:16])],
        sign_observations=v[12:16])


def winner_record(rows):
    if not rows: return dict(groups=0)
    values=torch.stack(rows).sum(0).cpu().tolist()
    return dict(groups=len(rows),winner_observations=int(values[0]),rival_observations=int(values[1]),
        neutral_equivalent=int(values[2]),good_set_mass_mean=values[3]/len(rows),
        good_set_top1_rate=values[4]/len(rows))


@HOOKS.register_module()
class TQGeometryWinnerAuditHook(Hook):
    def before_run(self,runner):
        self.model=runner.model.module if hasattr(runner.model,'module') else runner.model
        self.enabled=isinstance(self.model,TQSymNetGeometryWinner)
        if not self.enabled:return
        m=self.model
        row=dict(geometry_weight=m.geometry_weight,geometry_beta=m.geometry_beta,
            ambiguity_gap=m.geometry_ambiguity_gap,winners_max=m.winner_max,
            winner_temperature=m.winner_temperature,rank='goodset_replaces_pair_rank',
            count_normalized=True,rival_cap=None,unchosen_equivalent='neutral',
            geometry_only_training=True,output_boxes_unchanged=True)
        if runner.meta.get('geometry_winner_contract',row)!=row:
            raise RuntimeError('Resume geometry/winner contract changed')
        runner.meta['geometry_winner_contract']=row
        (Path(runner.work_dir)/'geometry_winner_contract.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        runner.logger.info('[GEOMETRY-WINNER-BUILD] %s',json.dumps(row))

    def before_train_epoch(self,runner):
        if self.enabled:
            self.model.geometry_train=[];self.model.winner_train=[];self.model.winner_sample=[]
            self.grad_rows=[]

    def after_train_iter(self,runner):
        if not self.enabled:return
        q=self.model.candidate_quality
        a,b=q.signed_aux[-1].weight.grad,q.high_head.weight.grad
        if a is not None and b is not None:
            with torch.no_grad():
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise RuntimeError('Nonfinite geometry/quality output-head gradient')
                self.grad_rows.append(torch.stack((a.norm(),b.norm(),(a!=0).any().to(a.dtype))))

    def after_train_epoch(self,runner):
        if not self.enabled:return
        m=self.model
        sampling=torch.tensor(m.winner_sample,dtype=torch.float64).sum(0).tolist() if m.winner_sample else []
        record=dict(epoch=runner.epoch+1,domain='supervised_BCE_sample_union_all_winner_competitors',
            geometry=geometry_record(m.geometry_train),winner=winner_record(m.winner_train),
            sampling_raw_base_final_added=sampling)
        if self.grad_rows:
            grad=torch.stack(self.grad_rows).sum(0).cpu().tolist()
            record['output_head_gradients']=dict(steps=len(self.grad_rows),aux_nonzero_steps=int(grad[2]),
                aux_norm_mean=grad[0]/len(self.grad_rows),quality_norm_mean=grad[1]/len(self.grad_rows),
                scope='different_output_heads_not_shared_gradient_conflict_measurement')
        runner.logger.info('[GEOMETRY-WINNER-TRAIN] %s',json.dumps(record))
        with (Path(runner.work_dir)/'geometry_winner_train.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        m.geometry_train=[];m.winner_train=[];m.winner_sample=[]
