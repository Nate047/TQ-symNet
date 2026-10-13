"""Conservative transforms and raw-image preservation for CXR models."""

import cv2
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class PreserveOriginalImage:
    """Keep an untouched copy of the decoded image for local refinement.

    This transform must run immediately after ``LoadImageFromFile`` and before
    every resize, flip, photometric transform, normalization, or padding step.
    The copy is intentionally not added to ``img_fields``: the detector maps
    coarse boxes back through ``scale_factor``/``flip`` and crops in the true
    original-image coordinate system.
    """

    def __init__(self, key='original_img'):
        self.key = str(key)

    def __call__(self, results):
        image = results.get('img')
        if image is None:
            raise KeyError(
                'PreserveOriginalImage requires LoadImageFromFile first')
        results[self.key] = np.ascontiguousarray(image.copy())
        results[self.key + '_shape'] = tuple(image.shape)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(key={self.key!r})'


@PIPELINES.register_module()
class PreserveAlignedImage:
    """Copy the current image after geometry, before normalization/padding.

    Unlike :class:`PreserveOriginalImage`, this copy deliberately lives in the
    detector's current coordinate system.  It is intended for a small local
    detail branch: Resize and RandomFlip must therefore precede this transform
    and Normalize/Pad must follow it.  The copied image is not put in
    ``img_fields`` so later image-only photometric transforms cannot silently
    desynchronise it from bbox coordinates.
    """

    def __init__(self, key='detail_img'):
        self.key = str(key)

    def __call__(self, results):
        image = results.get('img')
        if image is None:
            raise KeyError('PreserveAlignedImage requires an image')
        results[self.key] = np.ascontiguousarray(image.copy())
        results[self.key + '_shape'] = tuple(image.shape)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(key={self.key!r})'


@PIPELINES.register_module()
class CXRIntensityJitter:
    """Apply shared-channel gamma/contrast/noise without CLAHE or colour shift.

    A single 2-D noise field is shared by all three decoded channels, keeping
    a replicated grayscale CXR grayscale.  Bounding boxes and lung masks are
    unchanged because this transform is photometric only.
    """

    def __init__(self,
                 prob=0.8,
                 gamma_range=(0.92, 1.08),
                 contrast_range=(0.92, 1.08),
                 noise_std_range=(0.0, 2.0)):
        self.prob = float(prob)
        self.gamma_range = tuple(gamma_range)
        self.contrast_range = tuple(contrast_range)
        self.noise_std_range = tuple(noise_std_range)

    def __call__(self, results):
        if np.random.rand() >= self.prob:
            return results
        image = results['img']
        original_dtype = image.dtype
        work = image.astype(np.float32)
        gamma = np.random.uniform(*self.gamma_range)
        contrast = np.random.uniform(*self.contrast_range)
        noise_std = np.random.uniform(*self.noise_std_range)

        work = 255.0 * np.power(np.clip(work / 255.0, 0.0, 1.0), gamma)
        mean = work.mean(axis=(0, 1), keepdims=True)
        work = mean + contrast * (work - mean)
        if noise_std > 0:
            noise = np.random.normal(
                0.0, noise_std, size=work.shape[:2]).astype(np.float32)
            if work.ndim == 3:
                noise = noise[:, :, None]
            work = work + noise
        results['img'] = np.clip(work, 0.0, 255.0).astype(original_dtype)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(prob={self.prob}, '
                f'gamma_range={self.gamma_range}, '
                f'contrast_range={self.contrast_range}, '
                f'noise_std_range={self.noise_std_range})')


@PIPELINES.register_module()
class CXRRandomRotate:
    """Apply a small, image-centred CXR rotation consistently.

    This transform is purposely narrow: there is no translation, shear or
    scale change.  Besides the detector image and boxes, it transforms the
    original-pixel image used by the V330/V331 local refiner and the lung
    mask.  Thus the refiner's box-to-original mapping remains valid: its
    ``original_img`` is the image in the same pre-resize coordinate system as
    the augmented detector input.

    It must run after ``PreserveOriginalImage``, ``LoadLungMask`` and
    ``LoadAnnotations``, but before ``Resize``.  The transform records its
    angle for auditability; it is training-only and therefore cannot affect
    validation/test coordinates.
    """

    def __init__(self, prob=0.3, max_degree=3.0, border_value=None,
                 min_bbox_size=2.0):
        if not 0.0 <= float(prob) <= 1.0:
            raise ValueError('prob must lie in [0, 1]')
        if float(max_degree) < 0.0:
            raise ValueError('max_degree must be non-negative')
        self.prob = float(prob)
        self.max_degree = float(max_degree)
        self.border_value = border_value
        self.min_bbox_size = float(min_bbox_size)

    @staticmethod
    def _warp(image, matrix, size, interpolation, border_value):
        return cv2.warpAffine(
            image, matrix, size, flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

    def __call__(self, results):
        if np.random.rand() >= self.prob or self.max_degree == 0.0:
            results['cxr_rotation_degree'] = 0.0
            return results

        image = results['img']
        height, width = image.shape[:2]
        angle = float(np.random.uniform(-self.max_degree, self.max_degree))
        matrix = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), angle, 1.0)
        if self.border_value is None:
            border_value = tuple(float(v) for v in image.reshape(-1, image.shape[-1]).mean(0)) \
                if image.ndim == 3 else float(image.mean())
        else:
            border_value = self.border_value

        results['img'] = self._warp(
            image, matrix, (width, height), cv2.INTER_LINEAR, border_value)
        # PreserveOriginalImage deliberately runs first.  Keeping this copy in
        # the same rotated coordinate frame is essential for local refinement.
        if 'original_img' in results:
            results['original_img'] = self._warp(
                results['original_img'], matrix, (width, height),
                cv2.INTER_LINEAR, border_value)
        if results.get('lung_mask') is not None:
            results['lung_mask'] = self._warp(
                results['lung_mask'], matrix, (width, height),
                cv2.INTER_NEAREST, 0).astype(results['lung_mask'].dtype)

        for key in results.get('bbox_fields', []):
            boxes = results[key]
            if boxes.size == 0:
                continue
            corners = np.stack([
                boxes[:, [0, 2, 2, 0]], boxes[:, [1, 1, 3, 3]],
                np.ones((len(boxes), 4), dtype=boxes.dtype),
            ], axis=-1)
            warped = corners @ matrix.T
            rotated = np.stack([
                warped[:, :, 0].min(axis=1), warped[:, :, 1].min(axis=1),
                warped[:, :, 0].max(axis=1), warped[:, :, 1].max(axis=1),
            ], axis=1).astype(boxes.dtype, copy=False)
            rotated[:, [0, 2]] = np.clip(rotated[:, [0, 2]], 0, width)
            rotated[:, [1, 3]] = np.clip(rotated[:, [1, 3]], 0, height)
            valid = ((rotated[:, 2] - rotated[:, 0]) >= self.min_bbox_size
                     ) & ((rotated[:, 3] - rotated[:, 1]) >= self.min_bbox_size)
            results[key] = rotated[valid]
            if key == 'gt_bboxes' and 'gt_labels' in results:
                results['gt_labels'] = results['gt_labels'][valid]

        results['img_shape'] = results['img'].shape
        results['cxr_rotation_degree'] = angle
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(prob={self.prob}, '
                f'max_degree={self.max_degree}, '
                f'min_bbox_size={self.min_bbox_size})')
