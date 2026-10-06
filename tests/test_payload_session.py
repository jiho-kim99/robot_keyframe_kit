import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE=Path(__file__).resolve().parent
SCRIPTS=HERE.parent/'scripts'
if not SCRIPTS.exists():SCRIPTS=HERE
sys.path.insert(0,str(SCRIPTS))
import arm_payload as arm
from payload_session import ExperimentSession
from test_arm_payload import fixture


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        xml=root/'robot.xml';xml.write_text(fixture())
        self.b=arm.ResidualBench(*arm.build_model(xml,'right',payload=1),kp=200,kd=40)
        self.s=ExperimentSession(self.b,root/'logs',{'payload_kg':1})

    def populate(self):
        for name,q in zip(('A','B','C'),[.05,.1,.05]):
            self.s.set_pose(np.full(7,q));self.s.save_pose(name)
        self.s.switch_mode('measure')

    def test_no_recording_before_start(self):
        for _ in range(10):self.s.tick()
        self.s.switch_mode('measure')
        for _ in range(10):self.s.tick()
        self.assertFalse(self.s.output.exists())
        self.assertEqual(self.s.count,0)

    def test_start_holds_final_stop_and_repeat(self):
        self.populate()
        for _ in range(200):self.s.tick()
        plan=[dict(name='A',seconds=.3,hold=.4),dict(name='B',seconds=.5,hold=99)]
        self.s.start(plan)
        first=self.s.path
        for _ in range(5000):
            self.s.tick()
            if not self.s.active:break
        self.assertEqual(self.s.state,'COMPLETED')
        with (first/'samples.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(float(rows[0]['time_s']),0)
        self.assertEqual(rows[-1]['phase'],'COMPLETED')
        self.assertTrue(any(r['phase']=='HOLDING' for r in rows))
        self.assertLess(float(rows[-1]['time_s']),5)  # Final dwell 99 is explicitly ignored.
        commands=[json.loads(x) for x in (first/'commands.jsonl').read_text().splitlines()]
        self.assertGreaterEqual(commands[1]['start_time_s']-commands[0]['start_time_s'],.3+.4)
        self.assertEqual(commands[-1]['hold_seconds'],0)
        size=(first/'samples.csv').stat().st_size
        for _ in range(100):self.s.tick()
        self.assertEqual((first/'samples.csv').stat().st_size,size)
        self.s.start([dict(name='C',seconds=.3,hold=0)])
        second=self.s.path
        for _ in range(4000):
            self.s.tick()
            if not self.s.active:break
        self.assertNotEqual(first,second)
        with (second/'samples.csv').open() as f:rows2=list(csv.DictReader(f))
        self.assertEqual(float(rows2[0]['time_s']),0)
        self.assertEqual(float(rows2[0]['command_id']),1)

    def test_missing_pose_does_not_create_experiment(self):
        self.s.switch_mode('measure')
        with self.assertRaises(ValueError):self.s.start([dict(name='A',seconds=1,hold=0)])
        self.assertFalse(self.s.output.exists())

    def test_saved_frames_roundtrip(self):
        self.s.set_pose(np.full(7,.05));self.s.save_pose('A')
        saved=self.s.poses['A']
        self.s.set_pose(np.full(7,.1))
        other=ExperimentSession(self.b,self.s.output,{})
        other.load_poses(self.s.saved_path)
        np.testing.assert_allclose(other.poses['A']['joints'],saved['joints'])
        np.testing.assert_allclose(other.poses['A']['position_m'],saved['position_m'])

    def test_gain_range_and_recorded_values(self):
        self.s.set_gains(0,200)
        self.assertEqual((self.b.kp,self.b.kd),(0,200))
        for kp,kd in [(-1,0),(1001,40),(0,201),(float('nan'),40)]:
            with self.assertRaises(ValueError):self.s.set_gains(kp,kd)
        self.s.set_gains(1000,0)
        self.populate();self.s.start([dict(name='A',seconds=1,hold=0)])
        with self.assertRaises(ValueError):self.s.set_gains(10,10)
        self.s.finish('user_stopped')
        meta=json.loads((self.s.path/'metadata.json').read_text())
        self.assertEqual((meta['kp'],meta['kd']),(1000,0))

    def test_reload_file_is_atomic_and_locked_during_measurement(self):
        self.s.save_pose('A')
        old=self.s.poses.copy()
        invalid=self.s.output/'invalid.json'
        invalid.write_text(json.dumps(dict(joint_names=self.b.names,poses={
            'A':dict(joints=[.1]*7),'B':dict(joints=[1e6]*7)})))
        with self.assertRaises(ValueError):self.s.load_poses(invalid)
        self.assertEqual(self.s.poses,old)
        self.s.switch_mode('measure')
        with self.assertRaises(ValueError):self.s.load_poses(self.s.saved_path)

    def test_abort_and_mode_lock(self):
        self.populate();self.s.start([dict(name='B',seconds=2,hold=0)])
        with self.assertRaises(ValueError):self.s.switch_mode('edit')
        for _ in range(20):self.s.tick()
        self.s.finish('user_stopped')
        meta=json.loads((self.s.path/'metadata.json').read_text())
        self.assertEqual(meta['run_status'],'user_stopped')
        self.assertEqual(meta['samples'],20)
        self.s.switch_mode('edit')
        self.assertEqual(self.s.state,'EDIT')


if __name__=='__main__':unittest.main()
