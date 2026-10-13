"""Replay classwise Sym/quality Soft-NMS from an actual complete candidate export.

No GT is used in scoring, thresholding or selection. GT is read only by COCOeval.
Existing project functions are loaded from their AST to avoid an MMCV/CUDA build
dependency on the visualization workstation. Requires torch and pycocotools.
"""
import argparse, ast, contextlib, gzip, hashlib, io, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DEPS = ROOT.parent / 'tmp/paper_subtype_audit/deps'
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selection_functions():
    ns = {'torch': torch}
    paths = [(ROOT/'mmdet/core/bbox/iou_calculators/iou2d_calculator.py',
              {'fp16_clamp', 'bbox_overlaps'}),
             (ROOT/'TQ_SymNet/models/selection.py', {'gaussian_soft_select'})]
    hashes = {}
    for path, names in paths:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in functions} != names:
            raise ValueError('Required inference functions not found: ' + str(path))
        module = ast.Module(body=functions, type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ns)
        hashes[str(path.relative_to(ROOT))] = digest(path)
    return ns['gaussian_soft_select'], hashes


def coco_metrics(gt, predictions, categories, use_categories=True):
    from pycocotools.cocoeval import COCOeval
    if not predictions:
        raise ValueError('Empty replay predictions')
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(predictions)
        evaluator = COCOeval(gt, dt, 'bbox')
        evaluator.params.imgIds = sorted(gt.getImgIds())
        evaluator.params.catIds = categories
        evaluator.params.useCats = int(use_categories)
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.params.iouThrs = np.linspace(.5, .95, 10)
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    result = {key: float(evaluator.stats[i]) for i, key in enumerate(('AP','AP50','AP75'))}
    if use_categories:
        for index, catid in enumerate(categories):
            precision = evaluator.eval['precision'][:, :, index, 0, -1]
            name = gt.loadCats(catid)[0]['name']
            result[name] = dict(AP=float(precision[precision > -1].mean()),
                               AP50=float(precision[0][precision[0] > -1].mean()),
                               AP75=float(precision[5][precision[5] > -1].mean()))
    return result


def record(image_id, box, score, label):
    x1,y1,x2,y2 = map(float,box)
    return dict(image_id=int(image_id), bbox=[x1,y1,x2-x1,y2-y1],
                score=float(score), category_id=int(label)+1)


def main():
    from pycocotools.coco import COCO
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', required=True)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    with gzip.open(args.export, 'rt', encoding='utf-8') as stream:
        exported = json.load(stream)
    meta = exported['run']
    if not meta['complete'] or meta['gt_used_in_inference'] or meta['mode'] != 'tb':
        raise ValueError('Expected a complete, GT-free TB candidate export')
    if meta['ann_sha256'] != digest(args.ann_file):
        raise ValueError('Annotation source differs from the original evaluation')
    if meta['classes'] != ['ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis']:
        raise ValueError('Unexpected subtype order')
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(args.ann_file)
    if sorted(row['image_id'] for row in exported['images']) != sorted(gt.getImgIds()):
        raise ValueError('Incomplete or mismatched image IDs')
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    select, code_hashes = selection_functions()
    torch.set_num_threads(2)
    sigma = float(meta['selection_sigma']); topk = int(meta['selection_topk'])
    cap = int(meta['max_per_img']); threshold = float(meta['eligibility_score_thr'])
    alpha = float(meta['fusion_alpha'])
    predictions = {name: [] for name in ('original_tb','replay_tb','sym_classwise','quality_classwise')}
    mismatch_images = 0; score_errors = []
    for image_number, row in enumerate(exported['images'], 1):
        candidates = row['candidates']
        boxes = torch.tensor(candidates['boxes_xyxy'], dtype=torch.float32).reshape(-1,4)
        sym = torch.tensor(candidates['sym_probability'], dtype=torch.float32).reshape(-1,2)
        q = torch.tensor(candidates['quality_logit'], dtype=torch.float32)
        if len(boxes) != len(sym) or len(boxes) != len(q):
            raise ValueError('Candidate field alignment error')
        if len(boxes) and not (sym.max(1).values > threshold).all():
            raise ValueError('Unexpected candidate eligibility domain')
        for item in row['predictions']:
            predictions['original_tb'].append(record(row['image_id'], item['bbox_xyxy'], item['score'], item['label']))
        ids, scores = select(boxes, torch.tensor(candidates['fused_probability'], dtype=torch.float32), topk, sigma)
        ids, scores = ids[:cap], scores[:cap]
        expected = candidates['selected_indices']
        if ids.tolist() != expected:
            mismatch_images += 1
        if len(scores) == len(candidates['selected_scores']):
            score_errors.append(float((scores-torch.tensor(candidates['selected_scores'])).abs().max()))
        for index, score in zip(ids.tolist(), scores.tolist()):
            predictions['replay_tb'].append(record(row['image_id'], boxes[index].tolist(), score, sym[index].argmax()))
        fused = ((1-alpha)*torch.logit(sym.clamp(1e-6,1-1e-6)) + alpha*q[:,None]).sigmoid()
        for name, probability in (('sym_classwise', sym), ('quality_classwise', fused)):
            kept_ids, kept_scores, labels = [], [], []
            for label in range(2):
                eligible = (probability[:,label] > threshold).nonzero(as_tuple=False).squeeze(1)
                if not len(eligible):continue
                chosen, decayed = select(boxes[eligible], probability[eligible,label], topk, sigma)
                kept_ids.append(eligible[chosen]);kept_scores.append(decayed)
                labels.append(torch.full_like(chosen,label))
            if kept_ids:
                all_ids=torch.cat(kept_ids);all_scores=torch.cat(kept_scores);all_labels=torch.cat(labels)
                order=all_scores.argsort(descending=True)[:cap]
                for k in order.tolist():
                    predictions[name].append(record(row['image_id'], boxes[all_ids[k]].tolist(),
                                                    all_scores[k], all_labels[k]))
        if image_number % 50 == 0:print('Replayed %d/%d images' % (image_number,len(exported['images'])),flush=True)
    reports = {name: coco_metrics(gt,items,[1,2]) for name,items in predictions.items()}
    original_pooled = coco_metrics(gt,predictions['original_tb'],[1,2,3],False)
    replay_pooled = coco_metrics(gt,predictions['replay_tb'],[1,2,3],False)
    expected = meta['metrics']
    for metric, key in (('AP','official_tb_bbox_mAP'),('AP50','official_tb_bbox_mAP_50'),('AP75','official_tb_bbox_mAP_75')):
        if round(original_pooled[metric],3) != expected[key] or round(replay_pooled[metric],3) != expected[key]:
            raise ValueError('TB replay does not reproduce the original published metrics')
    for category in meta['classes']:
        for metric, suffix in (('AP','mAP'),('AP50','mAP_50')):
            key='bbox_'+category+'_'+suffix
            if round(reports['original_tb'][category][metric],3) != expected[key]:
                raise ValueError('Category evaluation does not reproduce original metrics')
    report=dict(source_export=str(args.export),export_sha256=digest(args.export),
                checkpoint=meta['checkpoint'],checkpoint_sha256=meta['checkpoint_sha256'],
                checkpoint_epoch=meta['checkpoint_epoch'],annotation_sha256=digest(args.ann_file),
                candidates=sum(len(row['candidates']['boxes_xyxy']) for row in exported['images']),
                images=len(exported['images']),fusion_alpha=alpha,sigma=sigma,threshold=threshold,
                per_class_topk=topk,global_cap=cap,selection_code_sha256=code_hashes,
                original_pooled=original_pooled,replay_pooled=replay_pooled,
                tb_replay_source_mismatch_images=mismatch_images,
                max_score_absolute_difference=max(score_errors,default=0.),
                metrics=reports,gt_used_for_selection=False,
                note='CPU replay of actual epoch-specific candidates using current inference functions; GT used only by COCOeval.')
    (out/'subtype_replay_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for name in ('sym_classwise','quality_classwise'):
        (out/(name+'.bbox.json')).write_text(json.dumps(predictions[name]),encoding='utf-8')
    print(json.dumps({k:reports[k] for k in ('sym_classwise','quality_classwise')},indent=2))
    print('Validated TB replay:',original_pooled,replay_pooled,'ID-mismatch images:',mismatch_images)


if __name__ == '__main__':main()
