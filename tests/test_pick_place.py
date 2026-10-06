import sys, unittest, tempfile
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from manipulation_platform import load_catalog,build_experiment
from evaluation.tasks import load_tasks,compile_task
from evaluation.runtime import TaskRunner

class PickPlaceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cat,base=load_catalog(ROOT/'configs/manipulation_platform.yaml')
  cls.b=build_experiment(cat,base,('prototype_barrett','waypoint_motion','base','right'))
  cls.config=load_tasks(ROOT/'configs/evaluation_tasks.yaml');cls.path=compile_task(cls.b,cls.config,'pick_and_place')
 def fk(self,phase):
  b=self.b;d=mujoco.MjData(b.model);d.qpos[b.qadr]=self.path.at(phase,1,False)[0];mujoco.mj_forward(b.model,d);return d
 def test_layout_and_top_down_pick(self):
  b=self.b;p=self.path;layout=p.pick_place
  self.assertAlmostEqual(layout['table_height_m'],b.data.xanchor[b.model.joint('waist_pitch_joint').id,2])
  for phase in [.055,.075,.475,.495]:
   d=self.fk(phase);np.testing.assert_allclose(d.site_xmat[b.sid].reshape(3,3)[:,0],[0,0,-1],atol=.003)
   np.testing.assert_allclose(d.site_xpos[b.sid],np.array(layout['front_object_xyz'])+[0,0,layout['grasp_offset_m']],atol=.002)
  for phase,key in [(.275,'right_basket_xyz'),(.775,'rear_basket_xyz')]:
   np.testing.assert_allclose(self.fk(phase).site_xpos[b.sid,:2],layout[key][:2],atol=.002)
 def test_arm_only_side_and_waist_rear(self):
  b=self.b;p=self.path;y=b.names.index('waist_yaw_joint');pitch=b.names.index('waist_pitch_joint')
  for phase in np.linspace(0,.55,150):self.assertEqual(p.at(phase,1)[0][y],0)
  for phase in [.73,.775,.795,.84]:self.assertAlmostEqual(p.at(phase,1)[0][y],-np.pi)
  for phase in np.linspace(0,1,301):self.assertEqual(p.at(phase,1,False)[0][pitch],0)
  # Transfer stays outside torso rather than cutting across the center.
  for phase in np.linspace(.13,.23,80):
   pos=self.fk(phase).site_xpos[b.sid];self.assertAlmostEqual(np.linalg.norm(pos[:2]),p.pick_place['radius_m'],delta=.002)
  np.testing.assert_array_equal(p.points[0],p.points[-1])
 def test_duration_scaling_and_metadata(self):
  for phase in np.linspace(0,1,151):
   q,v,a=self.path.at(phase,1,False);qq,vv,aa=self.path.at(phase*5,5,False)
   np.testing.assert_allclose(q,qq,atol=1e-10);np.testing.assert_allclose(v,5*vv,atol=1e-8);np.testing.assert_allclose(a,25*aa,atol=1e-6)
  with tempfile.TemporaryDirectory() as tmp:
   r=TaskRunner(self.b,self.config,'pick_and_place',tmp)
   self.assertEqual(r.metadata()['task_layout']['sequence'],['front','right_90','front','rear_180'])
   self.assertEqual(r.metadata()['object_mass_kg'],0)
if __name__=='__main__':unittest.main()
