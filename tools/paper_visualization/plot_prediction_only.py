"""Actual prediction overlays for an UNLABELLED test split.

Reads COCO image metadata + actual predictions.bbox.json. Does not read GT
annotations, compute IoU, label TP/FP or submit anything to the online server.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import read_json, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--images-json',required=True)
    parser.add_argument('--predictions',required=True,help='Actual COCO bbox predictions: XYWH')
    parser.add_argument('--img-root',required=True)
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--image-ids',help='Comma-separated COCO image IDs')
    parser.add_argument('--count',type=int,default=6)
    parser.add_argument('--display-topk',type=int,default=5)
    parser.add_argument('--display-min-score',type=float,default=.05)
    args=parser.parse_args()
    if min(args.count,args.display_topk)<1:parser.error('Budgets must be positive')
    metadata=read_json(args.images_json)
    infos={int(im['id']):im for im in metadata['images']}
    if len(infos)!=len(metadata['images']):raise ValueError('Duplicate image IDs')
    categories={int(c['id']):c['name'] for c in metadata.get('categories',[])}
    records=read_json(args.predictions)
    grouped=defaultdict(list)
    for p in records:
        if int(p['image_id']) not in infos:raise ValueError('Prediction image ID absent from metadata')
        if len(p['bbox'])!=4 or p['bbox'][2]<0 or p['bbox'][3]<0:
            raise ValueError('Expected native-image COCO XYWH boxes')
        if not np.isfinite([*p['bbox'],p['score']]).all():raise ValueError('Non-finite prediction')
        grouped[int(p['image_id'])].append(p)
    ids=([int(value) for value in args.image_ids.split(',')] if args.image_ids
         else list(infos)[:args.count])
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for image_id in ids:
        if image_id not in infos:raise ValueError('Unknown requested image ID')
        info=infos[image_id];path=Path(args.img_root)/info['file_name']
        image=np.asarray(Image.open(path).convert('L'))
        if image.shape!=(info['height'],info['width']):
            raise ValueError('Metadata/image size mismatch; no silent coordinate rescaling')
        predictions=sorted(grouped[image_id],key=lambda p:-p['score'])
        displayed=[p for p in predictions if p['score']>=args.display_min_score][:args.display_topk]
        fig,ax=plt.subplots(figsize=(4.2,4.5))
        ax.imshow(image,cmap='gray',vmin=0,vmax=255)
        for rank,p in enumerate(displayed,1):
            x,y,w,h=p['bbox']
            ax.add_patch(Rectangle((x,y),w,h,fill=False,ec='#ffcc46',lw=1.4))
            label=categories.get(int(p['category_id']),'category '+str(p['category_id']))
            ax.text(x,y,'%d %s %.3f'%(rank,label,p['score']),color='#ffcc46',fontsize=7,
                    bbox=dict(fc='black',alpha=.5,pad=.5,ec='none'))
        ax.set_axis_off()
        ax.set_title('Image %s | prediction only\nNo GT matching, TP/FP or true IoU'%image_id,fontsize=9)
        fig.tight_layout()
        fig.savefig(out/('prediction_%s.png'%image_id),dpi=300,bbox_inches='tight')
        plt.close(fig)
        manifest.append(dict(image_id=image_id,file_name=info['file_name'],
            total_predictions=len(predictions),displayed=len(displayed),predictions=displayed))
    write_json(out/'prediction_only_audit.json',dict(images_json=args.images_json,
        predictions=args.predictions,coordinate_format='COCO native XYWH',
        gt_read=False,score_threshold_display_only=args.display_min_score,images=manifest))
    print('Saved prediction-only images:',out.resolve())


if __name__=='__main__':main()
