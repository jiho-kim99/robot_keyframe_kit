import sys,tempfile,unittest,json
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from manipulation_platform import load_catalog,build_experiment
from evaluation.tasks import load_tasks
from evaluation.runtime import TaskRunner
class ObjectMassTests(unittest.TestCase):
 def setUp(self):
  cat,base=load_catalog(ROOT/'configs/manipulation_platform.yaml');self.b=build_experiment(cat,base,('prototype_barrett','waypoint_motion','base','right'));self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.r=TaskRunner(self.b,load_tasks(ROOT/'configs/evaluation_tasks.yaml'),'pick_and_place',self.tmp.name)
 def test_carry_force_and_release(self):
  r,b=self.r,self.b;r.set_object_mass(5);r.state='RUNNING'
  for phase,expected in [(0,0),(.1,5),(.32,0),(.5,5),(.82,0)]:
   r.task_time=phase*r.duration;b.data.qpos[b.qadr]=r.path.at(r.task_time,r.duration)[0];mujoco.mj_forward(b.model,b.data)
   qload,w=r.load_wrenches(np.zeros(len(b.names)));self.assertEqual(r.active_object_mass(),expected)
   np.testing.assert_allclose(w['right'][0],[0,0,-9.81*expected])
   jp=np.zeros((3,b.model.nv));jr=np.zeros_like(jp);mujoco.mj_jacSite(b.model,b.data,jp,jr,b.sid)
   np.testing.assert_allclose(qload,jp.T@np.array([0,0,-9.81*expected]),atol=1e-10)
  r.task_time=.1*r.duration;r.pause();self.assertEqual(r.active_object_mass(),5)
  r.stop();self.assertEqual(r.active_object_mass(),0)
 def test_mass_changes_close_log_and_validate(self):
  r=self.r;r.start_logging();path=r.logger.path;r.tick();r.set_object_mass(2)
  self.assertFalse(r.logger.active);self.assertEqual(json.loads((path/'metadata.json').read_text())['status'],'object_mass_changed')
  self.assertEqual(r.metadata()['object_mass_kg'],2)
  for value in [-1,float('nan'),float('inf')]:
   with self.assertRaises(ValueError):r.set_object_mass(value)
  r.start()
  with self.assertRaises(ValueError):r.set_object_mass(3)
  r.set_object_mass(2)
 def test_mass_increases_required_motor_torque(self):
  r,b=self.r,self.b;r.set_object_mass(2);r.state='PAUSED';r.task_time=.1*r.duration
  b.data.qpos[b.qadr]=r.path.at(r.task_time,r.duration)[0];r.hold=b.data.qpos[b.qadr].copy();mujoco.mj_forward(b.model,b.data)
  load,_=r.load_wrenches(np.zeros(len(b.names)))
  r.tick();loaded=b.data.ctrl.copy()
  # Compare the same state with gravity load removed.
  b.data.qpos[b.qadr]=r.hold;b.data.qvel[:]=0;r.set_object_mass(0);mujoco.mj_forward(b.model,b.data);r.tick();empty=b.data.ctrl.copy()
  np.testing.assert_allclose(loaded-empty,-load[b.dadr],atol=1e-8)
  self.assertGreater(np.linalg.norm(loaded-empty),1.)
if __name__=='__main__':unittest.main()
