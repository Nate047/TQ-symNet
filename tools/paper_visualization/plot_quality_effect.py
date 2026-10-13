"""Matched B1/A0 selection-effect figures, no images or CUDA needed.

The scorer does not refine boxes. 'Improvement' means better SELECTION of fixed
candidate coordinates. Compare identical Soft-NMS and output budgets.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import read_json, write_json, validate_comparison, iou_matrix, match_predictions


def per_gt_first_match(image):
    """First score-ordered one-to-one match at IoU .50 for each GT.

    Zero is a MISSING-MATCH sentinel, not a measured IoU. Assigned-GT IoU is
    computed explicitly: max-over-all-GT IoU is not used for this paired plot.
    """
    gt=[g['bbox_xyxy'] for g in image['gt']]
    predictions=image['predictions']
    result=match_predictions(predictions,gt,.50)
    overlap=iou_matrix([p['bbox_xyxy'] for p in predictions],gt)
    values=np.zeros(len(gt))
    indices=np.full(len(gt),-1,dtype=int)
    for index,target in enumerate(result['matched_gt']):
        if target>=0:
            values[target]=overlap[index,target]
            indices[target]=index
    coverage=[int((overlap>=t).any(0).sum()) for t in (.50,.75,.85)]
    return values,indices,coverage,result


def compare(baseline,quality):
    validate_comparison([baseline,quality])
    if not baseline['run'].get('complete') or not quality['run'].get('complete'):
        raise ValueError('Use complete validation exports, not --max-images smoke tests')
    keys=('selection_nms','selection_sigma','selection_topk')
    for key in keys:
        if key in baseline['run'] and key in quality['run']:
            if baseline['run'][key] != quality['run'][key]:
                raise ValueError('Unmatched suppression setting: '+key)
    if baseline['run'].get('selection_nms') not in (None,'gaussian_soft'):
        raise ValueError('Use B1 Soft-NMS to isolate quality, not B0 Hard-NMS')
    totals={name:dict(coverage=[0,0,0],error_counts={status:0 for status in
                ('TP','duplicate','localization','background')},prediction_count=0)
            for name in ('baseline','quality')}
    rows=[]
    for a,b in zip(baseline['images'],quality['images']):
        if a.get('ignored_gt') or b.get('ignored_gt'):
            raise ValueError('This paired diagnostic does not implement crowd matching')
        va,ia,ca,ma=per_gt_first_match(a)
        vb,ib,cb,mb=per_gt_first_match(b)
        for name,im,c,m in [('baseline',a,ca,ma),('quality',b,cb,mb)]:
            totals[name]['coverage']=[x+y for x,y in zip(totals[name]['coverage'],c)]
            totals[name]['prediction_count']+=len(im['predictions'])
            for status in totals[name]['error_counts']:
                totals[name]['error_counts'][status]+=m['status'].count(status)
        for g,(x,y) in enumerate(zip(va,vb)):
            rows.append(dict(image_id=a['image_id'],gt_index=g,
                subtype=a['gt'][g]['label'],gt_bbox=a['gt'][g]['bbox_xyxy'],
                baseline_iou=float(x),quality_iou=float(y),
                baseline_matched=bool(ia[g]>=0),quality_matched=bool(ib[g]>=0),
                baseline_prediction_index=int(ia[g]),quality_prediction_index=int(ib[g]),
                baseline_prediction=(a['predictions'][ia[g]] if ia[g]>=0 else None),
                quality_prediction=(b['predictions'][ib[g]] if ib[g]>=0 else None)))
    if not rows:raise ValueError('No eligible GT boxes')
    before=np.array([r['baseline_iou'] for r in rows])
    after=np.array([r['quality_iou'] for r in rows])
    delta=after-before
    summary=dict(gt_count=len(rows),selection_improved=int((delta>1e-6).sum()),
        selection_worsened=int((delta< -1e-6).sum()),
        unchanged=int((np.abs(delta)<=1e-6).sum()),
        gained_match50=int(((before==0)&(after>0)).sum()),
        lost_match50=int(((before>0)&(after==0)).sum()),
        both_unmatched50=int(((before==0)&(after==0)).sum()),
        stages=totals,zero_note='0 means unmatched at IoU .50; not actual IoU 0',
        interpretation='Selection quality only; candidate coordinates are unchanged.',
        suppression_verified=all(k in baseline['run'] and k in quality['run'] for k in keys))
    return before,after,rows,summary


def draw(before,after,summary,labels,out):
    colors=['#d18232','#1b5e85']
    fig,axes=plt.subplots(1,3,figsize=(10.7,3.1),constrained_layout=True)
    def distribution(value):
        return [int((value==0).sum()),int(((value>=.5)&(value<.75)).sum()),
                int(((value>=.75)&(value<.85)).sum()),int((value>=.85).sum())]
    for index,(value,label) in enumerate(zip((before,after),labels)):
        bars=axes[0].bar(np.arange(4)+(index-.5)*.32,distribution(value),
                         width=.30,color=colors[index],label=label)
        axes[0].bar_label(bars,fontsize=7,padding=2)
        coverage=summary['stages'][('baseline','quality')[index]]['coverage']
        bars=axes[1].bar(np.arange(3)+(index-.5)*.32,coverage,
                         width=.30,color=colors[index],label=label)
        axes[1].bar_label(bars,fontsize=7,padding=2)
    axes[0].set(xticks=range(4),xticklabels=['Missed .50','.50-.75','.75-.85','>= .85'],
                ylabel='GT count',title='(a) First score-ordered match')
    axes[0].tick_params(axis='x',labelrotation=20)
    axes[0].legend(frameon=False,fontsize=8)
    axes[1].set(xticks=range(3),xticklabels=['IoU .50','IoU .75','IoU .85'],
                ylabel='Covered GT count',title='(b) At least one final box covers GT')
    for ax in axes[:2]:ax.set_ylim(0,summary['gt_count']*1.15);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    difference=after-before
    for mask,color,label in [(difference>1e-6,'#368579','Improved'),
            (difference< -1e-6,'#ad635e','Worsened'),(np.abs(difference)<=1e-6,'#8e99a2','Unchanged')]:
        axes[2].scatter(before[mask],after[mask],s=12,c=color,alpha=.6,
                        label=label+' (%d)'%int(mask.sum()))
    axes[2].plot([0,1],[0,1],'--',color='#777777',lw=.8)
    axes[2].set(xlim=(-.03,1.03),ylim=(-.03,1.03),xlabel=labels[0]+' matched IoU',
                ylabel=labels[1]+' matched IoU',title='(c) Paired GT outcome; 0 = missed .50')
    axes[2].legend(fontsize=7,frameon=False,loc='upper left');axes[2].grid(alpha=.15)
    for suffix in ('png','svg'):fig.savefig(out/('quality_selection_effect.'+suffix),dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(4.7,2.8),constrained_layout=True)
    statuses=['TP','duplicate','localization','background']
    for index,label in enumerate(labels):
        counts=summary['stages'][('baseline','quality')[index]]['error_counts']
        bars=ax.bar(np.arange(4)+(index-.5)*.32,[counts[s] for s in statuses],width=.30,
                    color=colors[index],label=label)
        ax.bar_label(bars,fontsize=7,padding=2)
    ax.set(xticks=range(4),xticklabels=['TP','Duplicate FP','Localization FP','IoU < .10 FP'],
           ylabel='Final predictions',title='Diagnostic one-to-one matching at IoU .50')
    ax.tick_params(axis='x',labelrotation=15);ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for suffix in ('png','svg'):fig.savefig(out/('final_error_composition.'+suffix),dpi=300,bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',required=True,help='B1 visualization.json.gz')
    parser.add_argument('--quality',required=True,help='A0 visualization.json.gz')
    parser.add_argument('--out-dir',required=True)
    args=parser.parse_args()
    baseline,quality=read_json(args.baseline),read_json(args.quality)
    if baseline.get('schema')!='tq-paper-v1' or quality.get('schema')!='tq-paper-v1':
        raise ValueError('Use export_predictions.py files')
    before,after,rows,summary=compare(baseline,quality)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    draw(before,after,summary,[baseline['run']['tag'],quality['run']['tag']],out)
    with open(out/'paired_gt_selection.csv','w',newline='',encoding='utf-8') as stream:
        fields=['image_id','gt_index','subtype','baseline_iou','quality_iou',
                'baseline_matched','quality_matched','baseline_prediction_index','quality_prediction_index']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    write_json(out/'quality_effect_audit.json',dict(baseline=baseline['run'],
        quality=quality['run'],summary=summary,paired_gt=rows,
        note='Diagnostic matching, not interpolated COCO AP. No display threshold used.'))
    print('Saved quality-selection mechanism figures and paired GT audit:',out.resolve())


if __name__=='__main__':main()
