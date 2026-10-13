"""TBX11K-specific dataset views that preserve the official annotations."""

from mmcv.utils import print_log

from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class CocoCategorySubsetDataset(CocoDataset):
    """Training-only image subset selected by one or more COCO categories.

    All annotations on each selected image are retained. ``exclude_classes``
    can remove mixed-class images entirely; annotations are never selectively
    deleted because that would turn valid lesions into false background.
    """

    def __init__(self,
                 subset_classes,
                 exclude_classes=(),
                 log_stats=False,
                 **kwargs):
        self.subset_classes = tuple(subset_classes)
        self.exclude_classes = tuple(exclude_classes)
        self.log_stats = bool(log_stats)
        super().__init__(**kwargs)

    def _filter_imgs(self, min_size=32):
        valid_indices = super()._filter_imgs(min_size=min_size)
        subset_cat_ids = self.coco.get_cat_ids(cat_names=self.subset_classes)
        if len(subset_cat_ids) != len(self.subset_classes):
            raise ValueError(
                'Some subset_classes are absent from the annotation file: '
                '{}'.format(self.subset_classes))
        subset_image_ids = set(
            self.coco.get_img_ids(cat_ids=subset_cat_ids))
        if self.exclude_classes:
            exclude_cat_ids = self.coco.get_cat_ids(
                cat_names=self.exclude_classes)
            if len(exclude_cat_ids) != len(self.exclude_classes):
                raise ValueError(
                    'Some exclude_classes are absent from the annotation '
                    'file: {}'.format(self.exclude_classes))
            excluded_image_ids = set()
            for category_id in exclude_cat_ids:
                excluded_image_ids.update(
                    self.coco.get_img_ids(cat_ids=[category_id]))
            subset_image_ids.difference_update(excluded_image_ids)
        valid_indices = [
            index for index in valid_indices
            if self.data_infos[index]['id'] in subset_image_ids
        ]
        self.img_ids = [self.data_infos[index]['id'] for index in valid_indices]
        if self.log_stats:
            annotation_count = sum(
                len(self.coco.get_ann_ids(img_ids=[image_id]))
                for image_id in self.img_ids)
            print_log(
                '[TBX-SUBSET] include={} exclude={} images={} annotations={}'
                .format(self.subset_classes, self.exclude_classes,
                        len(self.img_ids), annotation_count),
                logger='root')
        return valid_indices
