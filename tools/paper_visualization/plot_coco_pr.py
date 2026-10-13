"""Official-style category-agnostic COCO precision-recall and AP-by-IoU curves.

Consumes predictions.bbox.json from the actual exporter. No score filtering.
The annotation file must be the SAME one used for inference/evaluation.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import read_json, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ann-file',required=True)
    parser.add_argument('--predictions',nargs='+',required=True)
    parser.add_argument('--labels',nargs='+',required=True)
    parser.add_argument('--out-dir',required=True)
    args=parser.parse_args()
    if len(args.labels)!=len(args.predictions): parser.error('One label per prediction file')
    if len(set(args.labels))!=len(args.labels):parser.error('Labels must be unique')
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt=COCO(args.ann_file)
    if not set(gt.getCatIds()).issubset({1,2,3}):
        raise ValueError('Category IDs do not match the official [1,2,3] protocol')
    thresholds=np.linspace(.5,.95,10)
    fig,axes=plt.subplots(1,3,figsize=(10.8,3.),constrained_layout=True)
    audit={}
    for path,label in zip(args.predictions,args.labels):
        predictions=read_json(path)
        if not predictions:
            precision=np.zeros((10,101))
        else:
            if any(p['image_id'] not in gt.imgs for p in predictions):
                raise ValueError('Prediction contains image IDs absent from GT')
            dt=gt.loadRes(predictions)
            evaluator=COCOeval(gt,dt,'bbox')
            evaluator.params.imgIds=sorted(gt.imgs)
            evaluator.params.catIds=[1,2,3]
            evaluator.params.useCats=0
            evaluator.params.maxDets=[1,10,100]
            evaluator.params.iouThrs=thresholds
            evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
            precision=evaluator.eval['precision'][:,:,0,0,2]
            precision=np.where(precision>=0,precision,np.nan)
        recall=np.linspace(0,1,101)
        axes[0].plot(recall,precision[0],label=label,lw=1.4)
        axes[1].plot(recall,precision[5],label=label,lw=1.4)
        ap=np.nanmean(precision,axis=1)
        axes[2].plot(thresholds,100*ap,marker='o',ms=3,label=label)
        audit[label]=dict(predictions=str(path),AP=float(np.nanmean(precision)),
                          AP50=float(ap[0]),AP75=float(ap[5]),AP_by_IoU=ap.tolist())
    for ax,title in zip(axes,['(a) IoU .50 PR','(b) IoU .75 PR','(c) Localization thresholds']):
        ax.set_title(title);ax.grid(alpha=.15);ax.legend(fontsize=7,frameon=False)
    for ax in axes[:2]:ax.set(xlabel='Recall',ylabel='Interpolated precision',xlim=(0,1),ylim=(0,1.02))
    axes[2].set(xlabel='IoU threshold',ylabel='AP (%)',ylim=(0,100))
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg'):fig.savefig(out/('precision_recall.'+suffix),dpi=300,bbox_inches='tight')
    write_json(out/'coco_curve_audit.json',dict(annotation=args.ann_file,useCats=0,
        catIds=[1,2,3],maxDets=[1,10,100],all_original_predictions=True,results=audit))
    plt.close(fig)


if __name__=='__main__':main()
