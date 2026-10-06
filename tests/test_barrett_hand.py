import sys, unittest, tempfile
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
import mujoco
ROOT=Path('/home/jiho/robot_keyframe_kit');sys.path.insert(0,str(ROOT/'scripts'))
from manipulation_platform import load_catalog,build_experiment
from evaluation.runtime import TaskRunner
from evaluation.tasks import load_tasks
class BarrettTests(unittest.TestCase):
 def test_hand_mass_and_topology(self):
  m=mujoco.MjModel.from_xml_path(str(ROOT/'prototype_v2.1.1/mjcf/scene_barrett.xml'))
  u=E.parse(ROOT/'prototype_v2.1.1/urdf/prototype_v2.1.1_barrett.urdf')
  self.assertEqual(len([j for j in u.findall('joint') if j.get('name').startswith('right_hand_') and j.get('type')=='revolute']),8)
  for link in u.findall('link'):
   name=link.get('name')
   if name.startswith('right_hand_') or name=='right_wrist_pitch_link':
    self.assertAlmostEqual(float(link.find('inertial/mass').get('value')),float(m.body(name).mass[0]))
  mass=sum(m.body_mass[i] for i in range(m.nbody) if m.body(i).name.startswith('right_hand_') or m.body(i).name=='right_wrist_pitch_link')
  self.assertAlmostEqual(mass,1.098463)
  np.testing.assert_allclose(m.site('right_hand_grasp').pos,[.12,-.0315,0],atol=1e-12)
  oldmesh=m.mesh('right_wrist_pitch_link').id
  self.assertFalse(any(m.geom_dataid[g]==oldmesh for g in range(m.ngeom) if m.geom_bodyid[g]==m.body('right_wrist_pitch_link').id))
 def test_all_tasks_with_no_payload(self):
  cat,base=load_catalog(ROOT/'configs/manipulation_platform.yaml');b=build_experiment(cat,base,('prototype_barrett','waypoint_motion','base','right'))
  cfg=load_tasks(ROOT/'configs/evaluation_tasks.yaml')
  self.assertEqual(len(b.names),16)
  with tempfile.TemporaryDirectory() as tmp:
   for name in cfg['tasks']:
    with self.subTest(task=name):
     mujoco.mj_resetData(b.model,b.data)
     r=TaskRunner(b,cfg,name,tmp,.1);b.data.qpos[b.qadr]=r.path.points[0];mujoco.mj_forward(b.model,b.data)
     r.state='RUNNING';r.duration=1.
     for _ in range(510):r.tick()
     self.assertTrue(np.isfinite(b.data.qpos).all());self.assertEqual(r.metadata()['payload_mass_kg'],0)
     self.assertEqual(r.metadata()['object_mass_kg'],0)
if __name__=='__main__':unittest.main()
