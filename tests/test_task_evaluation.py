import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path('/home/jiho/robot_keyframe_kit')
sys.path.insert(0,str(ROOT/'scripts'))
from evaluation.trajectory import TimedPath
from evaluation.tasks import load_tasks,compile_task
from evaluation.runtime import TaskRunner
from evaluation.analysis import read_run,summaries
from manipulation_platform import load_catalog,build_experiment

class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog,cls.base=load_catalog(ROOT/'configs/manipulation_platform.yaml')
        cls.config=load_tasks(ROOT/'configs/evaluation_tasks.yaml')
    def bench(self):
        return build_experiment(self.catalog,self.base,('prototype_v211','waypoint_motion','base','right'))
    def test_all_18_paths_same_geometry_with_duration_scaling(self):
        b=self.bench();self.assertEqual(len(self.config['tasks']),18)
        for name in self.config['tasks']:
            with self.subTest(task=name):
                path=compile_task(b,self.config,name)
                for phase in (.0,.13,.29,.63,.91):
                    q1,v1,a1=path.at(phase,1)
                    q5,v5,a5=path.at(phase*5,5)
                    np.testing.assert_allclose(q1,q5,atol=1e-12)
                    np.testing.assert_allclose(v1,5*v5,atol=1e-12)
                    np.testing.assert_allclose(a1,25*a5,atol=1e-10)
                for phase in path.phases:
                    _,v,a=path.at(phase*2,2,False)
                    np.testing.assert_allclose(v,0,atol=1e-10)
                    np.testing.assert_allclose(a,0,atol=1e-10)
    def test_logging_switch_pause_and_cycle_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=self.bench();r=TaskRunner(b,self.config,'pick_and_place',tmp,.1)
            self.assertFalse(r.logger.active)
            r.start(.5)
            for _ in range(320):r.tick()
            self.assertEqual(r.state,'RUNNING')
            self.assertAlmostEqual(r.task_time,.54,places=8)
            r.start_logging();path=r.logger.path
            for _ in range(20):r.tick()
            r.start_logging() # idempotent: do not reset the recording clock
            for _ in range(20):r.tick()
            r.pause();t=r.task_time
            for _ in range(10):r.tick()
            self.assertEqual(r.task_time,t)
            r.start();self.assertEqual(r.state,'PREPARING')
            r.select('pouring');self.assertFalse(r.logger.active)
            self.assertEqual(r.state,'PREPARING');self.assertEqual(r.task_time,0)
            meta=json.loads((path/'metadata.json').read_text())
            self.assertEqual(meta['motion_duration'],.5);self.assertEqual(meta['cycle_time'],.5)
            self.assertEqual(meta['trajectory_frequency_hz'],2)
            self.assertEqual(meta['trajectory_speed_scale'],self.config['tasks']['pick_and_place']['reference_duration']/.5)
            self.assertEqual(meta['status'],'task_changed')
            _,meta,data,segments=read_run(path)
            result=summaries(meta,data,segments)
            self.assertEqual(len(result),16)
            for j in b.names:np.testing.assert_allclose(data[j+'_power_W'],data[j+'_torque_Nm']*data[j+'_velocity_rad_s'])
            for value in (0,-1,float('nan'),float('inf')):
                with self.assertRaises(ValueError):r.set_duration(value)
    def test_18_task_physics_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=self.bench()
            for name in self.config['tasks']:
                with self.subTest(task=name):
                    mujoco.mj_resetData(b.model,b.data);mujoco.mj_forward(b.model,b.data)
                    r=TaskRunner(b,self.config,name,tmp,.1)
                    # Begin at path origin to test one full cycle without initial-pose transient.
                    b.data.qpos[b.qadr]=r.path.points[0];mujoco.mj_forward(b.model,b.data)
                    r.state='RUNNING';r.duration=1.
                    for _ in range(510):r.tick()
                    self.assertGreater(r.task_time,1)
                    self.assertEqual(r.state,'RUNNING')
                    self.assertTrue(np.isfinite(b.data.qpos).all())
                    self.assertIn('left',r.ee_sites)
    def test_duration_change_closes_log_without_changing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=TaskRunner(self.bench(),self.config,'wiping',tmp)
            original=r.path.fingerprint;r.start_logging();folder=r.logger.path;r.tick()
            r.set_duration(5)
            self.assertFalse(r.logger.active);self.assertEqual(r.path.fingerprint,original)
            self.assertEqual(json.loads((folder/'metadata.json').read_text())['status'],'duration_changed')
            r.reset();self.assertEqual(r.state,'PREPARING')

if __name__=='__main__':unittest.main()
