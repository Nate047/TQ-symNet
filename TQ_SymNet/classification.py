"""Official Sym image classifier on a strictly frozen detection backbone."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

CLASS_NAMES = ('Healthy', 'Sick non-TB', 'TB')


def image_class_from_filename(filename):
    """Match Sym's published image labels, without scanning parent directories.

    External negative cases are Healthy (0), not Sick non-TB (1).
    MC/Shenzhen use the final _0/_1; DA/DB use an n/p filename prefix.
    """
    name = filename.replace('\\', '/')
    directory = name.split('/')[0].lower()
    if directory in ('health', 'healthy'):
        return 0
    if directory == 'sick':
        return 1
    if directory == 'tb':
        return 2
    parts = name.split('/')
    if len(parts) == 4 and parts[0].lower() == 'extra' and parts[2] in ('train', 'val', 'test'):
        if parts[1] == 'mc+shenzhen':
            match = re.fullmatch(r'(?:CHNCXR|MCUCXR)_\d+_([01])\.png', parts[3])
            if match:
                return 2 * int(match.group(1))
        elif parts[1] == 'da+db':
            match = re.fullmatch(r'([np])x?\d+\.(?:png|jpg)', parts[3])
            if match:
                return 2 if match.group(1) == 'p' else 0
    raise ValueError('Unknown image label for %s; expected official TBX11K or extra filename' % filename)


def official_sym_classifier(channels=576):
    # Retain the released first 1x1 convolution's padding=1 intentionally.
    return nn.Sequential(
        nn.Conv2d(channels, 512, 1, padding=1, bias=True), nn.ReLU(inplace=False),
        nn.Conv2d(512, 512, 3, padding=1, bias=True), nn.ReLU(inplace=False),
        nn.Conv2d(512, 512, 3, padding=1, bias=True), nn.ReLU(inplace=False),
        nn.Conv2d(512, 512, 3, padding=1, bias=True), nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2, 0),
        nn.Conv2d(512, 512, 3, padding=1, bias=True), nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3, padding=1, bias=True), nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(512, 3, 1, bias=True))


def file_sha256(filename):
    digest = hashlib.sha256()
    with open(filename, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_frozen_backbone(backbone_config, checkpoint_path):
    backbone, _, audit = load_frozen_features(backbone_config, None, checkpoint_path)
    return backbone, audit


def load_frozen_features(backbone_config, neck_config, checkpoint_path):
    from mmcv.runner.checkpoint import _load_checkpoint
    from mmdet.models import build_backbone, build_neck
    from TQ_SymNet.checkpoints import load_retained_state
    config = dict(backbone_config)
    config['pretrained'] = None
    config.pop('init_cfg', None)
    holder = nn.Module()
    holder.backbone = build_backbone(config)
    if neck_config is not None:
        # Register the standalone project's measured CrossScale implementation.
        import TQ_SymNet.models  # noqa: F401
        holder.neck = build_neck(dict(neck_config))
    checkpoint = _load_checkpoint(str(checkpoint_path), map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    selected = {}
    for key, value in state.items():
        name = key[7:] if key.startswith('module.') else key
        if name.startswith('backbone.') or (neck_config is not None and name.startswith('neck.')):
            if name in selected:
                raise ValueError('Duplicate frozen-feature tensor: ' + name)
            selected[name] = value
    if not selected:
        raise ValueError('Expected a detector/TQ checkpoint with backbone.* tensors')
    # Uses the existing audited loading contract, including regenerated caches.
    audit = load_retained_state(holder, selected)
    holder.backbone.requires_grad_(False).eval()
    if neck_config is not None:
        holder.neck.requires_grad_(False).eval()
    return holder.backbone, getattr(holder, 'neck', None), audit


class FrozenBackboneSymClassifier(nn.Module):
    def __init__(self, backbone, channels=576):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.classifier = official_sym_classifier(channels)
        self.channels = channels

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            features = self.backbone(images)
            last = features[-1]
        if last.ndim != 4 or last.shape[1] != self.channels:
            raise ValueError('Unexpected final backbone feature shape: %s' % (tuple(last.shape),))
        # flatten(1) retains the batch axis even for a one-image last batch.
        return self.classifier(last).flatten(1)


class FrozenPyramidClassifier(nn.Module):
    """Separate image classifier with a frozen detection backbone and neck."""

    def __init__(self, backbone, neck, classifier):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.neck = neck.requires_grad_(False).eval()
        self.classifier = classifier

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        self.neck.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            pyramid = self.neck(self.backbone(images))
        return self.classifier(pyramid)


def build_frozen_classifier(config, checkpoint_path):
    """Reconstruct the architecture recorded in a classification config."""
    mode = config.get('classification_head', 'sym')
    if mode == 'sym':
        if config.get('neck') is not None:
            raise ValueError('Sym backbone-only control must not specify a neck')
        backbone, audit = load_frozen_backbone(config.backbone, checkpoint_path)
        return FrozenBackboneSymClassifier(backbone, config.input_channels), audit
    if mode not in ('hybrid_pool', 'class_query'):
        raise ValueError('Unsupported classification_head: %s' % mode)
    if config.get('neck') is None:
        raise ValueError('CrossScale classification requires the exact detector neck config')
    backbone, neck, audit = load_frozen_features(config.backbone, config.neck, checkpoint_path)
    from TQ_SymNet.classification_heads import CrossScaleClassQueryHead, CrossScaleHybridPoolHead
    options = dict(config.classification_options)
    classifier = (CrossScaleHybridPoolHead(**options) if mode == 'hybrid_pool'
                  else CrossScaleClassQueryHead(**options))
    return FrozenPyramidClassifier(backbone, neck, classifier), audit


class TBXImageDataset(Dataset):
    def __init__(self, annotation, image_root, training=False, labeled=True,
                 image_size=512, mean=(123.675, 116.28, 103.53),
                 std=(58.395, 57.12, 57.375), flip_probability=.5):
        self.annotation = str(annotation)
        self.image_root = Path(image_root)
        self.training = training
        self.image_size = image_size
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self.flip_probability = flip_probability
        index = json.loads(Path(annotation).read_text(encoding='utf-8'))
        self.images = index['images']  # Preserve submission index order exactly.
        self.image_ids = [item['id'] for item in self.images]
        self.filenames = [item['file_name'].replace('\\', '/') for item in self.images]
        if len(set(self.image_ids)) != len(self.images) or len(set(self.filenames)) != len(self.images):
            raise ValueError('Duplicate image IDs or filenames in ' + str(annotation))
        if not self.images:
            raise ValueError('Empty image index: ' + str(annotation))
        self.labels = [image_class_from_filename(name) if labeled else -1 for name in self.filenames]

    def source_counts(self):
        counts = {}
        for name, label in zip(self.filenames, self.labels):
            parts = name.split('/')
            source = parts[1] if parts[0] == 'extra' else 'TBX11K'
            entry = counts.setdefault(source, dict(images=0, class_counts=[0, 0, 0]))
            entry['images'] += 1
            if label >= 0:
                entry['class_counts'][label] += 1
        return counts

    def validate_image_files(self):
        """Fail before training on missing/empty files; never silently skip data."""
        invalid = []
        for name in self.filenames:
            path = self.image_root / name
            if not path.is_file() or path.stat().st_size == 0:
                invalid.append(name)
        if invalid:
            raise ValueError('%s: %d missing/empty images, examples=%s' %
                             (self.annotation, len(invalid), invalid[:10]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        import mmcv
        filename = self.image_root / self.filenames[index]
        image = mmcv.imread(str(filename), flag='color')
        if image is None:
            raise FileNotFoundError(str(filename))
        if image.shape[:2] != (self.image_size, self.image_size):
            raise ValueError('Expected official %dx%d image: %s has %s' %
                             (self.image_size, self.image_size, filename, image.shape))
        if self.training and torch.rand(()).item() < self.flip_probability:
            image = image[:, ::-1].copy()
        image = mmcv.imnormalize(image, self.mean, self.std, to_rgb=True)
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        return tensor, self.labels[index], index


def classification_metrics(labels, probabilities):
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError('Nonfinite class probabilities')
    if set(np.unique(labels)) != {0, 1, 2}:
        raise ValueError('Validation requires all three classes')
    prediction = probabilities.argmax(1)
    confusion = np.zeros((3, 3), np.int64)
    np.add.at(confusion, (labels, prediction), 1)
    recall = np.diag(confusion) / confusion.sum(1)
    is_tb = labels == 2
    onehot = np.eye(3)[labels]
    return dict(accuracy=float((prediction == labels).mean()),
                tb_auc=float(roc_auc_score(is_tb, probabilities[:, 2])),
                tb_sensitivity=float((prediction[is_tb] == 2).mean()),
                tb_specificity=float((prediction[~is_tb] != 2).mean()),
                macro_ap=float(average_precision_score(onehot, probabilities, average='macro')),
                macro_recall=float(recall.mean()), per_class_recall=recall.tolist(),
                confusion_matrix=confusion.tolist(), class_order=list(CLASS_NAMES))
