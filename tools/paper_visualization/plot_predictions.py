"""Reproducible paper figures from export_predictions.py JSON files.

No oracle-selected boxes, coordinate edits or alternative NMS. Images are drawn
at native resolution. GT is used for offline diagnostics and case selection.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from common import (BIN_LABELS, read_json, write_json, box_stats,
                    iou_matrix, match_predictions, validate_comparison)

COLORS = ['#1b5e85','#d18232','#587c4d','#98546d']


def ranks(values):
    """Average ranks for ties; NumPy-only Spearman helper."""
    values = np.asarray(values)
    if not len(values):
        return np.empty(0,dtype=float)
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    change = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1], True])
    output = np.empty(len(values), dtype=float)
    for lo, hi in zip(change[:-1], change[1:]):
        output[order[lo:hi]] = (lo+hi-1)/2.
    return output


def correlation(x,y):
    if len(x)<2 or np.std(x)==0 or np.std(y)==0:
        return None
    return float(np.corrcoef(ranks(x),ranks(y))[0,1])


def save(fig,out,name):
    # PNG for LaTeX, SVG for editing. No extra PDF files/large scatter-PDF.
    fig.savefig(out/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(out/(name+'.svg'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def summarize(run):
    total = {key:dict(count=0,iou_bins=[0]*5,covered_gt=[0]*3)
             for key in ('eligible','final')}
    matches = {str(t):dict(tp=0,fp=0,fn=0) for t in (.5,.75,.85)}
    per_image=[]
    for image in run['images']:
        if image['ignored_gt']:
            raise ValueError('This diagnostic matcher does not handle crowd/ignore GT; use COCOeval.')
        gt=[item['bbox_xyxy'] for item in image['gt']]
        for stage,boxes in [('eligible',image.get('candidates',{}).get('boxes_xyxy',[])),
                            ('final',[p['bbox_xyxy'] for p in image['predictions']])]:
            stats,_=box_stats(boxes,gt)
            total[stage]['count']+=stats['count']
            for key in ('iou_bins','covered_gt'):
                total[stage][key]=[a+b for a,b in zip(total[stage][key],stats[key])]
        row=dict(image_id=image['image_id'],gt=len(gt))
        for t in (.5,.75,.85):
            matched=match_predictions(image['predictions'],gt,t)
            for k in ('tp','fp','fn'):
                matches[str(t)][k]+=matched[k]
            row['tp'+str(t)]=matched['tp']
            row['fn'+str(t)]=matched['fn']
        per_image.append(row)
    return dict(stages=total,one_to_one=matches,images=per_image,
                has_candidates=all('candidates' in row for row in run['images']))


def stages_figure(runs, summaries, out):
    fig,axes=plt.subplots(1,3,figsize=(11,3.1),constrained_layout=True)
    for index,(run,summary) in enumerate(zip(runs,summaries)):
        color=COLORS[index%len(COLORS)]
        for stage,linestyle in [('eligible','--'),('final','-')]:
            if stage=='eligible' and not summary['has_candidates']:
                continue
            stats=summary['stages'][stage]
            percent=100*np.asarray(stats['iou_bins'])/max(stats['count'],1)
            axes[0].plot(range(5),percent,marker='o',color=color,linestyle=linestyle,
                         label=run['run']['tag']+' '+stage)
        final=summary['stages']['final']
        shift=(index-(len(runs)-1)/2)*.24
        bars=axes[1].bar(np.arange(3)+shift,final['covered_gt'],width=.23,
                        color=color,label=run['run']['tag'])
        axes[1].bar_label(bars,fontsize=7,padding=2)
        matched=summary['one_to_one']['0.5']
        axes[2].bar(index,matched['tp'],color=color)
        axes[2].bar(index,matched['fp'],bottom=matched['tp'],color=color,alpha=.30)
        axes[2].text(index,matched['tp']+matched['fp'],
                     'TP %d\nFP %d'%(matched['tp'],matched['fp']),ha='center',va='bottom',fontsize=8)
    axes[0].set(xticks=range(5),xticklabels=BIN_LABELS,xlabel='Max GT IoU',
                ylabel='Candidate share (%)',title='(a) Box quality distribution')
    axes[0].tick_params(axis='x',labelrotation=30)
    axes[0].legend(fontsize=6.4,frameon=False)
    axes[1].set(xticks=range(3),xticklabels=['IoU >= .50','IoU >= .75','IoU >= .85'],
                ylabel='Covered GT boxes',title='(b) Unique GT coverage')
    axes[1].set_ylim(0,max(1,max(s['stages']['final']['covered_gt'][0] for s in summaries))*1.18)
    axes[1].legend(fontsize=7,frameon=False)
    axes[2].set(xticks=range(len(runs)),xticklabels=[r['run']['tag'] for r in runs],
                ylabel='Final predictions',title='(c) One-to-one IoU .50 matching')
    axes[2].set_ylim(0,max(s['stages']['final']['count'] for s in summaries)*1.2+1)
    for ax in axes: ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    save(fig,out,'candidate_quality_distribution')


def score_figures(run,out,max_points):
    arrays={name:[] for name in ('iou','sym','quality','fused','sym_z','quality_z')}
    csv_path=out/'candidate_scores.csv'
    with open(csv_path,'w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['image_id','candidate_index','x1','y1','x2','y2','matched_gt',
            'true_iou','p_active','p_obsolete','sym_max','quality_logit',
            'quality_surrogate','fused_probability','selected_final','final_decayed_score'])
        for image in run['images']:
            candidates=image.get('candidates',{})
            if 'quality_logit' not in candidates: continue
            boxes=np.asarray(candidates['boxes_xyxy']).reshape(-1,4)
            sym=np.asarray(candidates['sym_probability']).reshape(-1,2)
            qz=np.asarray(candidates['quality_logit'])
            q=np.asarray(candidates['quality_probability'])
            fused=np.asarray(candidates['fused_probability'])
            if not all(len(v)==len(boxes) for v in (sym,qz,q,fused)):
                raise ValueError('Candidate fields have inconsistent lengths')
            overlap=iou_matrix(boxes,[g['bbox_xyxy'] for g in image['gt']])
            u=overlap.max(1) if overlap.shape[1] else np.zeros(len(boxes))
            gt=overlap.argmax(1) if overlap.shape[1] else np.full(len(boxes),-1)
            selected=dict(zip(candidates['selected_indices'],candidates['selected_scores']))
            maximum=sym.max(1)
            p=np.clip(maximum,1e-6,1.-1e-6)
            z=np.log(p/(1.-p))
            for name,values in zip(arrays,(u,maximum,q,fused,z,qz)):
                arrays[name].extend(values.tolist())
            for i,b in enumerate(boxes):
                writer.writerow([image['image_id'],i,*b,gt[i],u[i],*sym[i],maximum[i],
                    qz[i],q[i],fused[i],int(i in selected),selected.get(i,'')])
    if not arrays['iou']:
        return dict(note='No quality cache: export the quality run with --candidates')
    a={k:np.asarray(v) for k,v in arrays.items()}
    ids=np.arange(len(a['iou']))
    if len(ids)>max_points:
        ids=np.sort(np.random.default_rng(0).choice(ids,max_points,replace=False))
    levels=np.digitize(a['iou'][ids],[.50,.75,.85])
    palette=np.array(['#9ba5ac','#d7a84a','#36969d','#8a3b6a'])
    fig,axes=plt.subplots(1,3,figsize=(10.8,3.2),constrained_layout=True)
    final_p=np.clip(a['fused'],1e-6,1.-1e-6)
    for ax,name,x,title in zip(axes,['Sym logit','Quality logit zQ','Fused logit'],
         [a['sym_z'],a['quality_z'],np.log(final_p/(1.-final_p))],
         ['(a) Sym','(b) Quality','(c) Fusion before Soft-NMS']):
        ax.scatter(x[ids],a['iou'][ids],s=2,c=palette[levels],alpha=.25,rasterized=True)
        ax.set(xlabel=name,ylabel='True max GT IoU',ylim=(0,1.02),title=title)
        for t in (.5,.75,.85): ax.axhline(t,ls=':',color='#666666',lw=.6)
        ax.grid(alpha=.15)
    save(fig,out,'score_vs_true_iou')
    fig,ax=plt.subplots(figsize=(4.8,4.),constrained_layout=True)
    for level,label in enumerate(['IoU < .50','.50-.75','.75-.85','>= .85']):
        subset=ids[levels==level]
        ax.scatter(a['sym_z'][subset],a['quality_z'][subset],s=3,color=palette[level],
                   alpha=.3,label=label,rasterized=True)
    ax.set(xlabel='Sym max subtype logit',ylabel='Quality logit zQ',
           title='Candidate-score complementarity')
    ax.legend(markerscale=3,fontsize=8,frameon=False); ax.grid(alpha=.15)
    save(fig,out,'sym_quality_joint_distribution')
    stats=dict(candidates=len(ids) if len(a['iou'])<=max_points else len(a['iou']),
               displayed_points=len(ids),score_note=run['run']['quality_score_note'],
               spearman_all={k:correlation(a[k],a['iou']) for k in ('sym','quality','fused')},
               spearman_iou_ge75={k:correlation(a[k][a['iou']>=.75],a['iou'][a['iou']>=.75])
                                  for k in ('sym','quality','fused')})
    # Candidate correlations describe a duplicated candidate population, not
    # patient-independent statistics. Also export per-GT best-score selection.
    lesion=[]
    for image in run['images']:
        c=image.get('candidates',{})
        if 'quality_logit' not in c: continue
        overlap=iou_matrix(c['boxes_xyxy'],[g['bbox_xyxy'] for g in image['gt']])
        if not len(overlap): continue
        scores=[np.asarray(c['sym_probability']).max(1),c['quality_probability'],c['fused_probability']]
        for g in range(overlap.shape[1]):
            u=overlap[:,g]
            related=np.flatnonzero(u>=.5)
            if not len(related): continue
            item=dict(image_id=image['image_id'],gt_index=g,best_available_iou=float(u.max()))
            for label,score in zip(('sym','quality','fused'),scores):
                chosen=related[np.argmax(np.asarray(score)[related])]
                item[label+'_chosen_iou']=float(u[chosen])
            lesion.append(item)
    write_json(out/'per_gt_ranking.json',lesion)
    return stats


def choose_cases(runs,count,requested):
    images=runs[0]['images']
    if requested:
        ids=[int(i) for i in requested.split(',')]
        index={image['image_id']:i for i,image in enumerate(images)}
        if any(i not in index for i in ids): raise ValueError('Requested image ID not exported')
        return [(index[i],'author-selected') for i in ids]
    gt=[[g['bbox_xyxy'] for g in im['gt']] for im in images]
    covers=[[box_stats([p['bbox_xyxy'] for p in im['predictions']],g)[0]['covered_gt']
             for im,g in zip(run['images'],gt)] for run in runs]
    selected=[]
    def take(candidates,reason):
        candidates=[i for i in candidates if i not in [v[0] for v in selected]]
        if candidates and len(selected)<count: selected.append((candidates[0],reason))
    if len(runs)>1:
        take(sorted(range(len(images)),key=lambda i:(-(covers[-1][i][2]-covers[0][i][2]),images[i]['image_id'])),
             'largest IoU .85 coverage gain (post-hoc)')
    take([i for i,im in enumerate(images) if any(g['label']==1 for g in im['gt'])],
         'Obsolete-containing image')
    take(sorted(range(len(images)),key=lambda i:(-(len(gt[i])-covers[-1][i][1]),images[i]['image_id'])),
         'IoU .75 coverage failure (post-hoc)')
    for i in range(len(images)): take([i],'deterministic remaining case')
    return selected


def overlay(ax,image,gt,predictions,classes,topk,min_score,title,paper_style=False):
    ax.imshow(image,cmap='gray',vmin=0,vmax=255)
    for g in gt:
        x1,y1,x2,y2=g['bbox_xyxy']
        ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec='#00cbea',lw=1.0,
                             ls='--' if paper_style else '-'))
        if not paper_style:
            ax.text(x1,y1,'GT '+('O' if g['label']==1 else 'A'),color='#00cbea',fontsize=6,
                    bbox=dict(fc='black',alpha=.45,pad=.5,ec='none'))
    matched=match_predictions(predictions,[g['bbox_xyxy'] for g in gt],.5)
    shown=0
    displayed=[]
    for i,p in enumerate(predictions):
        if shown>=topk or p['score']<min_score: continue
        x1,y1,x2,y2=p['bbox_xyxy']; status=matched['status'][i]
        color='#65e572' if status=='TP' else ('#ffcc46' if status=='duplicate' else '#ff7171')
        ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec=color,lw=1.1))
        label=(str(i+1) if paper_style
               else '%s s=%.2f u=%.2f'%('O' if p['label']==1 else 'A',p['score'],matched['max_iou'][i]))
        label_x=x1 if not paper_style else max(0,x2-12)
        label_y=min(image.shape[0]-5,y2) if not paper_style else max(0,y1)
        ax.text(label_x,label_y,label,
                color=color,fontsize=7 if paper_style else 5.8,
                bbox=dict(fc='black',alpha=.55,pad=.5,ec='none'))
        displayed.append((i+1,'O' if p['label']==1 else 'A',matched['max_iou'][i]))
        shown+=1
    ax.set_title(title if paper_style else title+'\n%d displayed / %d outputs'%(shown,len(predictions)),
                 fontsize=8 if paper_style else 9)
    ax.set_axis_off()
    if paper_style and displayed:
        text=' | '.join('%d:%s'%(rank,subtype) for rank,subtype,u in displayed)
        text+='\nu: '+' / '.join('%.2f'%u for rank,subtype,u in displayed)
        ax.text(.02,-.035,text,transform=ax.transAxes,va='top',ha='left',fontsize=7)


def qualitative(runs,cases,out,img_root,topk,min_score,paper_style=False):
    columns=len(runs)+1
    figure_size=(7.1,len(cases)*2.45+.40) if paper_style else (columns*3.,len(cases)*3.2)
    fig,axes=plt.subplots(len(cases),columns,figsize=figure_size,squeeze=False)
    manifest=[]
    for row,(index,reason) in enumerate(cases):
        reference=runs[0]['images'][index]
        path=Path(img_root)/reference['file_name'] if img_root else Path(reference['filename'])
        if not path.is_file(): raise FileNotFoundError('Use --img-root: '+str(path))
        image=np.asarray(Image.open(path).convert('L'))
        if image.shape!=(reference['height'],reference['width']):
            raise ValueError('Image size differs from annotation; no silent resizing: '+str(path))
        axes[row,0].imshow(image,cmap='gray',vmin=0,vmax=255)
        for gt in reference['gt']:
            x1,y1,x2,y2=gt['bbox_xyxy']
            axes[row,0].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec='#00cbea',lw=1.6))
            axes[row,0].text(x1,y1,'Obsolete' if gt['label']==1 else 'Active',color='#00cbea',fontsize=7)
        axes[row,0].set_title('GT | image ID %s'%reference['image_id'],fontsize=8 if paper_style else 9)
        axes[row,0].set_axis_off()
        for col,run in enumerate(runs,1):
            im=run['images'][index]
            tag=run['run']['tag']
            title=({'B1':'Sym + Soft-NMS','A0':'TQ-SymNet'}.get(tag,tag) if paper_style else tag)
            overlay(axes[row,col],image,im['gt'],im['predictions'],run['run']['classes'],
                    topk,min_score,title,paper_style)
        manifest.append(dict(image_id=reference['image_id'],reason=reason,path=str(path)))
    fig.suptitle('Cyan: GT | Green: matched TP (.50) | Yellow: duplicate FP | Red: other FP\n'
                 'Display only: top %d, score >= %.3f; metrics use ALL original outputs.'%(topk,min_score),
                 fontsize=7 if paper_style else 9)
    fig.subplots_adjust(top=.90 if paper_style else .92,bottom=.04 if paper_style else .01,
                         left=.01,right=.99,hspace=.36 if paper_style else .20,wspace=.03)
    save(fig,out,'qualitative_detections')
    write_json(out/'case_manifest.json',manifest)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',nargs='+',required=True,help='B0/B1/A0 visualization.json.gz, in column order')
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--img-root',help='Local override; joins original relative file_name')
    parser.add_argument('--image-ids',help='Comma-separated COCO image IDs; otherwise gain/minority/failure cases')
    parser.add_argument('--cases',type=int,default=4)
    parser.add_argument('--display-topk',type=int,default=5)
    parser.add_argument('--display-min-score',type=float,default=.05)
    parser.add_argument('--max-points',type=int,default=20000)
    parser.add_argument('--no-images',action='store_true')
    parser.add_argument('--paper-style',action='store_true',help='Compact IEEE-width panels and box labels')
    args=parser.parse_args()
    if min(args.cases,args.display_topk,args.max_points)<1: parser.error('Budgets must be positive')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    runs=[read_json(path) for path in args.runs]
    if any(r.get('schema')!='tq-paper-v1' for r in runs):
        raise ValueError('Use export_predictions.py exports, not historical incompatible score dumps')
    if any(not r['images'] for r in runs):
        raise ValueError('Empty image export')
    if any(not any(im['gt'] for im in r['images']) for r in runs):
        raise ValueError('No GT boxes: cannot label TP/FP or true IoU. Use plot_prediction_only.py for unlabelled test images.')
    validate_comparison(runs)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    summaries=[summarize(r) for r in runs]
    stages_figure(runs,summaries,out)
    score_run=next((r for r in reversed(runs) if any('quality_logit' in im.get('candidates',{})
                                                  for im in r['images'])),runs[-1])
    score_stats=score_figures(score_run,out,args.max_points)
    if not args.no_images:
        cases=choose_cases(runs,min(args.cases,len(runs[0]['images'])),args.image_ids)
        qualitative(runs,cases,out,args.img_root,args.display_topk,args.display_min_score,args.paper_style)
    write_json(out/'figure_audit.json',dict(sources=args.runs,
        runs=[r['run'] for r in runs],summaries=summaries,score_diagnostics=score_stats,
        note='Matching counts are diagnostic only; AP comes from original COCOeval outputs.'))
    print('Saved real-data figures + candidate_scores.csv + audit to',out.resolve())


if __name__=='__main__':
    main()
