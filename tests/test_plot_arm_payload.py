import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE = Path(__file__).resolve().parents[1]/'scripts'/'plot_arm_payload.py'
if not MODULE.exists(): MODULE=Path(__file__).with_name('plot_arm_payload.py')
spec=importlib.util.spec_from_file_location('plots',MODULE)
plots=importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


class PlotTests(unittest.TestCase):
    def test_command_filter_preserves_gaps(self):
        data={'time_s':np.arange(6,dtype=float),'command_id':np.array([0,1,1,2,3,3])}
        segments=plots.select_samples(data,[1,3])
        self.assertEqual([x.tolist() for x in segments],[[1,2],[4,5]])

    def test_short_joint_names(self):
        names=['right_elbow_pitch_joint','right_wrist_pitch_joint']
        self.assertEqual(plots.select_joints(names,['elbow_pitch']),names[:1])
        with self.assertRaises(ValueError): plots.select_joints(names,['unknown'])

    def test_incomplete_csv_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'samples.csv'
            path.write_text('time_s,x\n0\n')
            with self.assertRaisesRegex(ValueError,'Incomplete'): plots.read_log(path)

    def test_rating_validation(self):
        import json
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'ratings.json'
            def config(rated,peak):
                return {'units':'joint_output_Nm','joints':{'j':{'rated_torque_Nm':rated,'peak_torque_Nm':peak}}}
            p.write_text(json.dumps(config(20,40)))
            self.assertEqual(plots.load_ratings(p,['j'])['j']['rated_torque_Nm'],20)
            p.write_text(json.dumps(config(None,None)))
            self.assertIsNone(plots.load_ratings(p,['j'])['j']['peak_torque_Nm'])
            p.write_text(json.dumps(config(50,40)))
            with self.assertRaises(ValueError):plots.load_ratings(p,['j'])

    def test_yaml_and_curve_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'ratings.yaml'
            p.write_text('reference: joint_output\njoints:\n  j:\n    rated_torque_nm: 10\n    peak_torque_nm: 40\n    rated_velocity_rad_s: 6\n    no_load_velocity_rad_s: 8\n')
            e=plots.load_ratings(p,['j'])['j']
            peak,rated=plots.motor_envelope(np.array([0,6,8,9]),e)
            np.testing.assert_allclose(peak,[40,10,0,0])
            np.testing.assert_allclose(rated,[10,10,0,0])
            r,p=plots.outside_envelopes(np.array([-1,1,8,9]),np.array([-12,50,0,0]),e)
            np.testing.assert_array_equal(r,[True,False,False,False])
            np.testing.assert_array_equal(p,[False,True,False,True])

    def test_time_weighted_summary_excludes_gaps(self):
        data={'time_s':np.array([0.,1.,10.,13.]),
              'j_torque_Nm':np.array([-2.,-2.,4.,4.]),
              'j_velocity_rad_s':np.array([1.,1.,-3.,-3.])}
        result=plots.measured_summary(data,'j',[np.array([0,1]),np.array([2,3])])
        np.testing.assert_allclose(result,[np.sqrt(13),4,np.sqrt(7),3])

    def test_saved_values_and_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'samples.csv'
            fields=['time_s','right_elbow_pitch_joint_position_rad','right_elbow_pitch_joint_velocity_rad_s',
                    'right_elbow_pitch_joint_torque_Nm','right_elbow_pitch_joint_saturated']
            with path.open('w') as stream:
                writer=csv.writer(stream);writer.writerow(fields)
                writer.writerows([[0,0,0,3,0],[1,0,2,3,1],[2,0,-1,3,0]])
            data,joints,meta,commands=plots.read_log(path)
            output=Path(temp)/'plots'
            plots.draw(data,joints,meta,commands,plots.select_samples(data),output,dpi=60,
                       ratings={joints[0]:{'rated_torque_Nm':2.0,'peak_torque_Nm':5.0}})
            with (output/'summary.csv').open() as stream:
                summary=list(csv.DictReader(stream))[0]
            self.assertAlmostEqual(float(summary['rated_velocity_rad_s']),1.5)
            self.assertEqual(float(summary['peak_velocity_rad_s']),2)
            self.assertEqual(len(summary),9)
            self.assertEqual(float(summary['motor_rated_torque_Nm']),2)
            self.assertEqual(float(summary['motor_peak_torque_Nm']),5)
            self.assertEqual(summary['motor_no_load_velocity_rad_s'],'')
            self.assertEqual(float(summary['rated_torque_Nm']),3)
            self.assertEqual(float(summary['peak_torque_Nm']),3)
            self.assertTrue((output/'torque_table.png').exists())
            self.assertTrue((output/'v_t.png').exists())
            for filename in ['torque_time.png','velocity_time.png','torque_velocity.png']:
                self.assertGreater((output/filename).stat().st_size,1000)


if __name__=='__main__': unittest.main()
