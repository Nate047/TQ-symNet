"""Compare the retained legacy MAX/all-IoU network to the split package on CUDA.

Run BEFORE replacing the optional legacy import shim. By default disable the
verified inactive V44 mask switches and use the new mask-free validation batch.
No results are fabricated when a dependency, dataset or checkpoint is missing.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from TQ_SymNet.checkpoints import load_retained_checkpoint
    from mmcv.utils import import_modules_from_strings
    from mmdet.apis import set_random_seed
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='The SAME selected MAX/all-IoU checkpoint for both models')
    parser.add_argument('--legacy-config', required=True, help='Explicit archived legacy config; not needed for standalone training')
    parser.add_argument('--new-config', default='configs/test_val.py')
    parser.add_argument('--images', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--loss-check', action='store_true')
    parser.add_argument('--legacy-masks', action='store_true',
                        help='Use original mask files when available; default comparison is mask-free')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Run in the original server CUDA/MMCV environment')
    set_random_seed(args.seed, deterministic=True)
    torch.cuda.set_device(args.device)
    legacy_cfg, new_cfg = Config.fromfile(args.legacy_config), Config.fromfile(args.new_config)
    for cfg in (legacy_cfg, new_cfg):
        import_modules_from_strings(**cfg.custom_imports)
    if not args.legacy_masks:
        head = legacy_cfg.model.bbox_head
        if (head.type != 'RetinaNAMGuideAttHeadV44'
                or head.positional_encoding.get('use_upper_prior', False)
                or head.get('lung_anatomy_bias', False)
                or head.get('lung_negative_weighting', False)):
            raise ValueError('Mask-free legacy comparison is valid only for the audited inactive-mask V44 route')
        head.lung_anatomy_injection = 'off'
    dataset_cfg = copy.deepcopy(legacy_cfg.data.test if args.legacy_masks else new_cfg.data.test)
    dataset_cfg.test_mode = True
    dataset = build_dataset(dataset_cfg)
    models = []
    for cfg in (legacy_cfg, new_cfg):
        model = build_detector(cfg.model)
        model.init_weights()
        checkpoint, audit = load_retained_checkpoint(model, args.checkpoint)
        print('Checkpoint compatibility:', audit)
        model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
        model.cuda(args.device).eval()
        model.quality_collect = False
        models.append(model)
    old, new = models
    assert set(old.state_dict()) == set(new.state_dict()), 'Checkpoint parameter/buffer keys changed'
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), 'Loaded tensor differs: ' + name
    maximum = dict(feature=0., box=0., subtype=0., quality=0., final_score=0.)
    def same(left, right, kind):
        assert left.shape == right.shape, (kind, left.shape, right.shape)
        if left.numel():
            maximum[kind] = max(maximum[kind], float((left-right).abs().max()))
        torch.testing.assert_close(left, right, rtol=args.rtol, atol=args.atol)
    for index in range(min(args.images, len(dataset))):
        data = scatter(collate([dataset[index]], samples_per_gpu=1), [args.device])[0]
        img, metas = data['img'][0], data['img_metas'][0]
        originals = old._v330_unwrap_original_images(data['original_img'], len(metas))
        with torch.no_grad():
            a_features, a_raw, _ = old._main(img, metas)
            b_features, b_raw, _ = new._main(img, metas)
            for a, b in zip(a_features, b_features):
                same(a, b, 'feature')
            for i, ((a_box, a_sym), (b_box, b_sym), meta, original) in enumerate(
                    zip(a_raw, b_raw, metas, originals)):
                same(a_box, b_box, 'box'); same(a_sym, b_sym, 'subtype')
                keep = (a_sym[:, :-1] > old.bbox_head.test_cfg.score_thr).any(1)
                a_memory = old._image_memory(a_features, img, meta, original, i)
                b_memory = new._image_memory(b_features, img, meta, original, i)
                normalized = old._normalized(a_box[keep], meta)
                a_out = old._score_candidates(normalized, a_memory, a_sym[keep])[0]
                b_out = new._score_candidates(normalized, b_memory, b_sym[keep])[0]
                same(a_out, b_out, 'quality')
                same(old._final_score(a_sym[keep], *a_out.unbind(1)),
                     new._final_score(b_sym[keep], *b_out.unbind(1)), 'final_score')
            for mode in ('tb', 'subtype'):
                old.inference_mode = new.inference_mode = mode
                a_result = old.simple_test(img, metas, original_img=data['original_img'], rescale=True)
                b_result = new.simple_test(img, metas, original_img=data['original_img'], rescale=True)
                for a_image, b_image in zip(a_result, b_result):
                    for a_class, b_class in zip(a_image, b_image):
                        np.testing.assert_allclose(a_class, b_class, rtol=args.rtol, atol=args.atol)
            old.inference_mode = new.inference_mode = 'tb'
        for model in models:
            model.detail_debug = []; model.candidate_quality.attention_stats = []
        if index == 0 and args.loss_check:
            gt = [torch.as_tensor(dataset.get_ann_info(index)['bboxes'], device=img.device)]
            labels = [torch.as_tensor(dataset.get_ann_info(index)['labels'], device=img.device)]
            # The validation transform is unflipped; restore GT to resized coordinates.
            gt[0] = gt[0] * gt[0].new_tensor(metas[0]['scale_factor'])
            cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            values, gradients = [], []
            for model in models:
                torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(gpu_rng)
                model.train(); model.zero_grad()
                loss = model.forward_train(img, metas, gt, labels, original_img=data['original_img'])
                total = sum(v.mean() for k, v in loss.items() if k.startswith('loss'))
                total.backward()
                values.append({k: v.detach().cpu() for k, v in loss.items()})
                gradients.append({k: p.grad.detach().cpu() for k, p in model.named_parameters()
                                  if p.grad is not None})
                assert not any(p.grad is not None for k, p in model.named_parameters()
                               if not k.startswith('candidate_quality.')), 'Frozen detector got gradients'
                model.eval()
            assert values[0].keys() == values[1].keys()
            for key in values[0]:
                torch.testing.assert_close(values[0][key], values[1][key], rtol=args.rtol, atol=args.atol)
            assert gradients[0].keys() == gradients[1].keys()
            for key in gradients[0]:
                torch.testing.assert_close(gradients[0][key], gradients[1][key], rtol=args.rtol, atol=args.atol)
        print('PASS image', index, metas[0].get('ori_filename', metas[0].get('filename')))
    print(json.dumps({'passed': True, 'max_absolute_differences': maximum,
                      'images': min(args.images, len(dataset)), 'loss_gradient_checked': args.loss_check}, indent=2))


if __name__ == '__main__':
    main()
