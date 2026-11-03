"""F2 = F1 + explicit inner/outer paired semantic differences, not edge GT."""
_base_ = ['./f1_roi_content_query.py']
model = dict(experiment_tag='ROI_QUERY_F2_PAIRED', roi_paired_boundary=True)
