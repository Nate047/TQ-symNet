"""Run the real selectors without MMCV: empty-after-suppression is not empty input."""
from pathlib import Path
import torch
from check_math import definitions

ROOT = Path(__file__).resolve().parents[1]


def overlaps(a, b):
    intersection = (torch.minimum(a[:, None, 2:], b[None, :, 2:]) -
                    torch.maximum(a[:, None, :2], b[None, :, :2])).clamp_min(0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection).clamp_min(1e-6)


def main():
    from torchvision.ops import nms as reference_nms
    scope = dict(torch=torch, bbox_overlaps=overlaps)
    definitions(ROOT / 'TQ_SymNet/models/selection.py', scope,
                {'hard_select', 'gaussian_soft_select'})
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    for device in devices:
        duplicate = torch.tensor([[0., 0., 10., 10.]] * 3, device=device)
        scores = torch.tensor([.7, .9, .8], device=device)
        ids, values = scope['hard_select'](duplicate, scores)
        assert ids.tolist() == [1], ('all-overlap', device, ids)
        torch.testing.assert_close(values, scores[ids])
        boxes = torch.cat((duplicate, duplicate[:1] + 20), dim=0)
        mixed_scores = torch.cat((scores, scores.new_tensor([.6])))
        ids, values = scope['hard_select'](boxes, mixed_scores)
        assert ids.tolist() == [1, 3], ('two-clusters', device, ids)
        torch.testing.assert_close(values, mixed_scores[ids])
        for name in ('hard_select', 'gaussian_soft_select'):
            for data, score, budget in ((duplicate[:0], scores[:0], 128),
                                        (duplicate[:1], scores[:1], 128),
                                        (duplicate, scores, 0),
                                        (duplicate, scores, 1)):
                selected, value = scope[name](data, score, topk=budget)
                assert len(selected) == len(value)
                assert len(selected) <= min(len(data), budget)
                assert selected.dtype == torch.long and selected.device == data.device
                assert value.device == score.device and value.dtype == score.dtype
                if not len(data) or budget == 0:
                    assert selected.shape == value.shape == (0,)
            soft_ids, soft_scores = scope['gaussian_soft_select'](boxes, mixed_scores)
            assert len(soft_ids) == len(boxes)
            assert len(set(soft_ids.tolist())) == len(boxes)
            assert torch.isfinite(soft_scores).all()
            for bad_boxes, bad_scores in ((duplicate, scores[:2]), (duplicate, scores[:, None])):
                try:
                    scope[name](bad_boxes, bad_scores)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Invalid candidate/score shapes accepted: ' + name)
        generator = torch.Generator().manual_seed(20261004)
        for count in range(1, 65):
            starts = torch.rand((count, 2), generator=generator) * 10
            ends = starts + torch.rand((count, 2), generator=generator) * 8 + .1
            random_boxes = torch.cat((starts, ends), dim=1).to(device)
            # Unique scores avoid implementation-dependent tie handling.
            random_scores = torch.randperm(count, generator=generator).float().to(device) / count + .01
            selected, value = scope['hard_select'](random_boxes, random_scores)
            expected = reference_nms(random_boxes, random_scores, .5)
            assert selected.tolist() == expected.tolist(), (device, count, selected, expected)
            torch.testing.assert_close(value, random_scores[expected])
        print('PASS', device, 'hard/soft empty-input, suppression-empty, clusters, topk, score alignment')
        print('PASS', device, '64 random hard-NMS cases equal torchvision.ops.nms')


if __name__ == '__main__':
    main()
