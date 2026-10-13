"""CPU-only regression checks. Artificial boxes are test fixtures, not figures."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from common import (box_stats, iou_matrix, match_predictions, read_json,
                    validate_comparison, write_json)
from plot_predictions import ranks, correlation, summarize, score_figures, qualitative
from plot_quality_effect import per_gt_first_match, compare, draw


class Checks(unittest.TestCase):
    def test_paired_iou_uses_assigned_gt_not_max_iou(self):
        image=dict(gt=[dict(bbox_xyxy=[0,0,10,10]),dict(bbox_xyxy=[2,0,12,10])],
            predictions=[dict(bbox_xyxy=[0,0,10,10],score=.9),
                         dict(bbox_xyxy=[0,0,10,10],score=.8)])
        values,indices,coverage,_=per_gt_first_match(image)
        np.testing.assert_allclose(values,[1.,2./3.])
        np.testing.assert_array_equal(indices,[0,1])

    def test_paired_effect_and_suppression_guard(self):
        baseline=dict(run=dict(tag='FIXTURE_BASE',complete=True,ann_sha256='a',
            detector_sha256='b',selection_nms='gaussian_soft',selection_sigma=.7,
            selection_topk=128),images=[dict(image_id=1,width=32,height=32,ignored_gt=[],
                gt=[dict(bbox_xyxy=[0,0,10,10],label=0)],
                predictions=[dict(bbox_xyxy=[0,0,20,10],score=.9),
                             dict(bbox_xyxy=[0,0,10,10],score=.8)])])
        quality=copy.deepcopy(baseline)
        quality['run']['tag']='FIXTURE_QUALITY'
        quality['images'][0]['predictions']=[dict(bbox_xyxy=[0,0,10,10],score=.9),
                                           dict(bbox_xyxy=[0,0,20,10],score=.8)]
        before,after,rows,summary=compare(baseline,quality)
        self.assertEqual(summary['selection_improved'],1)
        self.assertEqual(summary['stages']['baseline']['coverage'],[1,1,1])
        self.assertEqual(summary['stages']['quality']['coverage'],[1,1,1])
        with tempfile.TemporaryDirectory() as temp:
            draw(before,after,summary,['FIXTURE_BASE','FIXTURE_QUALITY'],Path(temp))
            self.assertTrue((Path(temp)/'quality_selection_effect.png').is_file())
        quality['run']['selection_sigma']=.5
        with self.assertRaises(ValueError):compare(baseline,quality)

    def test_continuous_iou_and_empty(self):
        np.testing.assert_allclose(iou_matrix([[0,0,10,10]],[[0,0,10,10]]),[[1.]])
        np.testing.assert_allclose(iou_matrix([[0,0,20,10]],[[0,0,10,10]]),[[.5]])
        self.assertEqual(iou_matrix([],[[0,0,10,10]]).shape,(0,1))
        stats,best=box_stats([],[[0,0,10,10]])
        self.assertEqual(stats['covered_gt'],[0,0,0])
        self.assertEqual(len(best),0)

    def test_duplicate_is_not_second_tp(self):
        predictions=[dict(bbox_xyxy=[0,0,10,10],score=.9),
                     dict(bbox_xyxy=[0,0,10,10],score=.8),
                     dict(bbox_xyxy=[20,20,30,30],score=.7)]
        result=match_predictions(predictions,[[0,0,10,10]])
        self.assertEqual(result['status'],['TP','duplicate','background'])
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,2,0))

    def test_choose_unmatched_gt_instead_of_duplicate(self):
        p=[dict(bbox_xyxy=[0,0,10,10],score=.9),dict(bbox_xyxy=[0,0,10,10],score=.8)]
        result=match_predictions(p,[[0,0,10,10],[0,0,10,10]])
        self.assertEqual(result['tp'],2)

    def test_tie_rank_and_empty(self):
        np.testing.assert_allclose(ranks([2,1,1,3]),[2,.5,.5,3])
        self.assertEqual(len(ranks([])),0)
        self.assertIsNone(correlation([1,1],[0,1]))
        self.assertAlmostEqual(correlation([0,1,2],[2,1,0]),-1.)

    def test_gzip_and_failed_cohort(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'fixture.json.gz';write_json(p,dict(text='测试'))
            self.assertEqual(read_json(p),dict(text='测试'))
        a=dict(run=dict(ann_sha256='same',detector_sha256='same'),images=[
            dict(image_id=1,width=10,height=10,gt=[],ignored_gt=[])])
        b=copy.deepcopy(a);validate_comparison([a,b])
        b['run']['detector_sha256']='different'
        with self.assertRaises(ValueError):validate_comparison([a,b])

    def test_render_pipeline_only_to_temporary_directory(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);Image.fromarray(np.zeros((32,32),np.uint8)).save(out/'test.png')
            im=dict(image_id=1,width=32,height=32,filename=str(out/'test.png'),file_name='test.png',
                gt=[dict(bbox_xyxy=[0,0,10,10],label=0,category_id=1)],ignored_gt=[],
                predictions=[dict(bbox_xyxy=[0,0,10,10],label=0,score=.8)],
                candidates=dict(boxes_xyxy=[[0,0,10,10],[16,16,20,20]],sym_probability=[[.8,.1],[.5,.1]],
                    quality_logit=[2.,-2.],quality_probability=[.880797,.119203],
                    fused_probability=[.85,.25],selected_indices=[0],selected_scores=[.8]))
            run=dict(run=dict(tag='TEST_FIXTURE_NOT_A_RESULT',classes=['A','O'],
                quality_score_note='Fixture only'),images=[im])
            self.assertEqual(summarize(run)['one_to_one']['0.5']['tp'],1)
            self.assertEqual(score_figures(run,out,10)['candidates'],2)
            qualitative([run],[(0,'unit test only')],out,None,3,0.)
            self.assertTrue((out/'qualitative_detections.png').is_file())


if __name__=='__main__':unittest.main()
