"""Remove the spatial-detail group: four-side queries AND native side-detail.

Native detail itself contains 24 side queries; leaving it on is not a complete
side-query ablation. A4 separately isolates the native-detail path.
"""
_base_ = ['./_matched_quality.py']
model = dict(selector_edges=False, selector_detail=False)
