"""Time actual inference, not a reimplemented scoring pipeline. CUDA FP32 only.

Sequential model loading; batch 1; same images; no GT tensors or Oracle.
Reports uninstrumented wall latency plus a separate event-instrumented pass.
Stage event intervals may include CPU launch gaps; they are not pure kernel time.
"""
import argparse
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def stats(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(count=0)
    return dict(count=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.percentile(a, 90)), min=float(a.min()), max=float(a.max()))


class Events:
    def __init__(self):
        self.undo = []
        self.rows = defaultdict(list)
        self.raw = None

    def wrap(self, obj, attr, name, capture=False):
        import torch
        original = getattr(obj, attr)
        def call(*args, **kwargs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            end.record()
            self.rows[name].append((start, end))
            if capture:
                self.raw = result
            return result
        self.undo.append((obj, attr, original))
        setattr(obj, attr, call)

    def clear(self):
        self.rows.clear()
        self.raw = None

    def restore(self):
        for obj, attr, original in reversed(self.undo):
            setattr(obj, attr, original)
        self.undo.clear()

    def values(self):
        return {name: sum(a.elapsed_time(b) for a,b in pairs) for name,pairs in self.rows.items()}


def prediction_digest(result):
    import numpy as np
    h = hashlib.sha256()
    for image in result:
        for cls in image:
            a = np.ascontiguousarray(cls)
            h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def params(model):
    quality = getattr(model, 'candidate_quality', None)
    return dict(total=sum(p.numel() for p in model.parameters()),
        quality=sum(p.numel() for p in quality.parameters()) if quality is not None else 0,
        declared_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        parameter_mib=sum(p.numel()*p.element_size() for p in model.parameters())/2**20,
        note='Allocated architecture parameters, including inactive slots; not FLOPs or active-only counts.')


def benchmark_detector(name, config, checkpoint, args):
    import torch
    from mmcv import Config
    from mmcv.parallel import scatter
    from mmcv.utils import import_modules_from_strings
    from mmdet.datasets import build_dataset, build_dataloader
    from mmdet.models import build_detector
    from TQ_SymNet.checkpoints import load_retained_checkpoint, load_detector_checkpoint
    from TQ_SymNet.models.candidates import main_hash
    from TQ_SymNet.models import selection, detail
    cfg = Config.fromfile(config)
    import_modules_from_strings(**cfg.custom_imports)
    cfg.model.backbone.pretrained = None
    if 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None
    quality = cfg.model.type == 'TQSymNet'
    if quality:
        cfg.model.diagnostics = False
        cfg.model.inference_mode = 'tb'
    elif cfg.model.type not in ('TQDetector', 'TQDenseHeadBaseline') or cfg.model.get('inference_nms') != 'tb_soft':
        raise ValueError('Baseline must be TQDetector/TQDenseHeadBaseline with tb_soft')
    if cfg.model.type == 'TQDenseHeadBaseline':
        cfg.model.diagnostics = False
    data_cfg = cfg.data.val.copy()
    if args.annotation: data_cfg.ann_file = args.annotation
    if args.image_root: data_cfg.img_prefix = args.image_root
    data_cfg.test_mode = True
    dataset = build_dataset(data_cfg)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=args.workers,
                             dist=False, shuffle=False)
    core = build_detector(cfg.model)
    load = load_retained_checkpoint if quality else load_detector_checkpoint
    _, audit = load(core, checkpoint)
    digest = main_hash(core)
    core.cuda().eval()
    core.quality_collect = False
    core.capture_candidates = False
    setup = params(core)
    n = min(args.images, len(dataset))
    if n < 1: raise ValueError('Empty evaluation dataset')
    first = next(iter(loader))
    if any(k.startswith('gt_') for k in first): raise ValueError('GT in inference pipeline')
    first = scatter(first, [0])[0]
    with torch.no_grad():
        for _ in range(args.warmup): core(return_loss=False, rescale=True, **first)
    torch.cuda.synchronize()
    del first
    times, h2d, peaks, increments, outputs = [], [], [], [], []
    # Authoritative throughput uses normal inference, no hooks/events within it.
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= n: break
            torch.cuda.synchronize(); start = time.perf_counter()
            batch = scatter(batch, [0])[0]
            torch.cuda.synchronize(); h2d.append((time.perf_counter()-start)*1000)
            for repeat in range(args.repeats):
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                start = time.perf_counter()
                result = core(return_loss=False, rescale=True, **batch)
                torch.cuda.synchronize()
                times.append((time.perf_counter()-start)*1000)
                peak = torch.cuda.max_memory_allocated()
                peaks.append(peak/2**20); increments.append((peak-base)/2**20)
                if repeat == 0: outputs.append(prediction_digest(result))
                del result
            del batch
    events = Events()
    events.wrap(core.backbone, 'forward', 'backbone')
    events.wrap(core.neck, 'forward', 'neck')
    events.wrap(core.bbox_head, 'forward', 'head')
    events.wrap(core.bbox_head, 'get_bboxes', 'decode', capture=True)
    if quality:
        events.wrap(core, '_image_memory', 'quality_memory')
        events.wrap(core, '_score_candidates', 'quality_queries')
        events.wrap(core, '_final_score', 'fusion')
    # The real call sites: local import in TQDetector, module global in selector.
    events.wrap(selection, 'gaussian_soft_select', 'soft_nms')
    events.wrap(detail, 'gaussian_soft_select', 'soft_nms')
    if cfg.model.type == 'TQDenseHeadBaseline':
        from TQ_SymNet import supplement
        events.wrap(supplement, 'gaussian_soft_select', 'soft_nms')
    stages, images = defaultdict(list), []
    try:
        with torch.no_grad():
            for index,batch in enumerate(loader):
                if index >= n: break
                batch = scatter(batch,[0])[0]; events.clear()
                result = core(return_loss=False,rescale=True,**batch)
                torch.cuda.synchronize()
                if prediction_digest(result) != outputs[index]:
                    raise RuntimeError('Instrumented output differs from normal inference at image %d' % index)
                measurements = events.values()
                for key,value in measurements.items(): stages[key].append(value)
                boxes,sym = events.raw[0]
                keep = (sym[:,:-1] > core.bbox_head.test_cfg.score_thr).any(1)
                h = hashlib.sha256()
                for tensor in (boxes[keep],sym[keep]):
                    array = tensor.detach().cpu().contiguous().numpy()
                    h.update(str(array.shape).encode());h.update(array.tobytes())
                images.append(dict(image_id=int(dataset.img_ids[index]),raw=len(boxes),
                    eligible=int(keep.sum()),final=sum(len(c) for c in result[0]),
                    candidate_sha256=h.hexdigest(),stage_ms=measurements))
                del result,batch,boxes,sym,keep
                events.clear()
    finally:
        events.restore()
    report = dict(name=name, config=str(config), config_sha256=sha(config),
        resolved_config_sha256=hashlib.sha256(cfg.pretty_text.encode('utf-8')).hexdigest(),
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), checkpoint_audit=audit,
        detector_state_sha256=digest, annotation=str(data_cfg.ann_file),
        annotation_sha256=sha(data_cfg.ann_file), images=n, repeats=args.repeats,
        warmup=args.warmup, batch_size=1, fp32=True, amp=False, parameters=setup,
        inference_wall_ms=stats(times), scatter_wall_ms=stats(h2d),
        fps=1000./(sum(times)/len(times)),peak_allocated_mib=max(peaks),
        peak_increment_mib=max(increments), stage_event_ms={k:stats(v) for k,v in stages.items()},
        selection=dict(sigma=core.selector_sigma if quality else core.baseline_sigma,
                       topk=128 if quality else core.baseline_topk,
                       score_thr=core.bbox_head.test_cfg.score_thr,
                       nms_pre=core.bbox_head.test_cfg.nms_pre,
                       max_per_img=core.bbox_head.test_cfg.max_per_img),
        candidate_statistics=images,
        note='Wall excludes dataloader and initial scatter; includes native-image transfer inside memory preparation and CPU output conversion. Stages are separate instrumented CUDA event intervals, not pure kernel cost. Diagnostics disabled. No classifier included.')
    del events,core,loader,dataset
    gc.collect(); torch.cuda.empty_cache()
    return report


def benchmark_classifier(checkpoint,args):
    import torch
    from mmcv import Config
    from mmcv.runner.checkpoint import _load_checkpoint
    from torch.utils.data import DataLoader
    from TQ_SymNet.classification import (build_frozen_classifier,restore_classification_snapshot,
        TBXImageDataset,state_sha256)
    snapshot=_load_checkpoint(checkpoint,map_location='cpu')
    cfg=Config(snapshot['config']);source=args.classifier_source or snapshot['backbone_checkpoint']
    if sha(source)!=snapshot['backbone_checkpoint_sha256']:
        raise ValueError('Wrong classifier initialization source')
    model,_=build_frozen_classifier(cfg,source)
    restore_classification_snapshot(model,snapshot)
    if state_sha256(model.backbone)!=snapshot['backbone_state_sha256']:
        raise ValueError('Classifier backbone restore mismatch')
    if (getattr(model,'neck',None) is not None
            and state_sha256(model.neck)!=snapshot.get('neck_state_sha256')):
        raise ValueError('Classifier neck restore mismatch')
    model.cuda().eval()
    annotation=args.classifier_annotation or cfg.data.val_annotation
    dataset=TBXImageDataset(annotation,args.image_root or cfg.data.image_root,training=False,
        labeled=False,image_size=cfg.data.image_size,mean=cfg.data.mean,std=cfg.data.std,
        clahe=cfg.data.get('clahe'))
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=args.workers)
    times=[];peaks=[]
    with torch.no_grad():
        x=next(iter(loader))[0].cuda()
        for _ in range(args.warmup):model(x)
        torch.cuda.synchronize(); del x
        for index,(x,_,ids) in enumerate(loader):
            if index>=args.images:break
            x=x.cuda();torch.cuda.synchronize()
            for _ in range(args.repeats):
                torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                y=model(x).softmax(1);y.cpu();torch.cuda.synchronize()
                times.append((time.perf_counter()-start)*1000)
                peaks.append(torch.cuda.max_memory_allocated()/2**20)
            del x,y
    result=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        annotation=str(annotation),annotation_sha256=sha(annotation),parameters=params(model),
        inference_wall_ms=stats(times),peak_allocated_mib=max(peaks),batch_size=1,
        images=len(times)//args.repeats,repeats=args.repeats,warmup=args.warmup,
        fps=1000./(sum(times)/len(times)),
        note='Classifier only, preprocessing/loading and input transfer excluded; CPU output included. Not full-screening throughput.')
    del model,loader,dataset,snapshot;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',nargs=3,action='append',metavar=('NAME','CONFIG','CHECKPOINT'))
    parser.add_argument('--classifier-checkpoint')
    parser.add_argument('--classifier-source')
    parser.add_argument('--classifier-annotation')
    parser.add_argument('--annotation')
    parser.add_argument('--image-root')
    parser.add_argument('--images',type=int,default=200)
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--out-dir',required=True)
    args=parser.parse_args()
    if not args.run and not args.classifier_checkpoint:parser.error('Supply --run or --classifier-checkpoint')
    if args.images<1 or args.warmup<1 or args.repeats<1:parser.error('Positive budgets required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required, no CPU fallback')
    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    cfgs=args.run or []
    if len({r[0] for r in cfgs})!=len(cfgs):raise ValueError('Duplicate run names')
    result=dict(environment=dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),tf32=False,
        cudnn_deterministic=True),runs=[], complete=False)
    for name,config,checkpoint in cfgs:
        print('Benchmarking',name,flush=True)
        result['runs'].append(benchmark_detector(name,config,checkpoint,args))
        (out/'costs.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if len(result['runs'])>1:
        base=result['runs'][0]
        for other in result['runs'][1:]:
            for key in ('annotation_sha256','detector_state_sha256','selection'):
                if base[key]!=other[key]:raise RuntimeError('Not matched controls: '+key)
            a=[(r['image_id'],r['candidate_sha256']) for r in base['candidate_statistics']]
            b=[(r['image_id'],r['candidate_sha256']) for r in other['candidate_statistics']]
            if a!=b:raise RuntimeError('Eligible candidate boxes/scores differ between controls')
        result['paired_candidates_verified']=True
    if args.classifier_checkpoint:
        result['classifier']=benchmark_classifier(args.classifier_checkpoint,args)
    result['complete']=True
    (out/'costs.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# Cost benchmark','', '| Model | Parameters (M) | Quality (M) | Mean ms | P90 ms | Peak MiB |',
           '|---|---:|---:|---:|---:|---:|']
    for row in result['runs']:
        lines.append('| %s | %.3f | %.3f | %.2f | %.2f | %.1f |' % (row['name'],
            row['parameters']['total']/1e6,row['parameters']['quality']/1e6,
            row['inference_wall_ms']['mean'],row['inference_wall_ms']['p90'],row['peak_allocated_mib']))
    if 'classifier' in result:
        row=result['classifier']
        lines+=['','## Separate classifier cost','',
            'Parameters: %.3f M; mean: %.2f ms; P90: %.2f ms; peak: %.1f MiB; images: %d.' % (
                row['parameters']['total']/1e6,row['inference_wall_ms']['mean'],
                row['inference_wall_ms']['p90'],row['peak_allocated_mib'],row['images'])]
    lines+=['','Timing excludes loader and initial scatter. Same GPU, FP32, batch 1. See JSON for all timing boundaries and per-stage distributions.']
    (out/'costs.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved',out/'costs.json')


if __name__=='__main__':main()
