import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from manipulation_platform import load_catalog,build_experiment,PlatformSelector
from payload_session import ExperimentSession

class PlatformTests(unittest.TestCase):
    def test_prototype_arms_waist_and_variants(self):
        c,base=load_catalog(ROOT/'configs/manipulation_platform.yaml')
        for variant in ('base','10kg','15kg','20kg'):
            b=build_experiment(c,base,('prototype_v211','payload_lift',variant,'right'))
            self.assertEqual(len(b.names),16)
            self.assertEqual(b.model.nv,16)
            self.assertTrue(all('hip' not in n and 'ankle' not in n and 'knee' not in n for n in b.names))
            self.assertIn('waist_pitch_joint',b.names)
            self.assertTrue(b.hidden_bodies)
            if variant!='base':self.assertEqual(b.model.body('right_wrist_pitch_link').mass[0],float(variant[:-2]))
            self.assertEqual(len(b.curl_presets()[0]),16)
            row,*_=b.sample()
            self.assertTrue(np.isfinite(row).all())

    def test_second_robot_logging_and_pose_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'robot.xml').write_text('''<mujoco><worldbody><body><joint name="base_turn" type="hinge" range="-90 90" actuatorfrcrange="-20 20"/><geom type="sphere" size=".1" mass="1"/><body pos=".2 0 0"><joint name="tool_turn" type="hinge" range="-90 90" actuatorfrcrange="-10 10"/><geom type="sphere" size=".05" mass=".2"/><site name="tip" pos=".1 0 0"/></body></body></worldbody></mujoco>''')
            c={'robots':{'mini':{'models':{'base':'robot.xml'},'joint_groups':{'arm':['base_turn','tool_turn']},'default_groups':['arm'],'end_effectors':{'tool':{'site':'tip'}}}},'domains':{'motion':{'kind':'waypoint_motion'}}}
            b=build_experiment(c,root,('mini','motion','base','tool'))
            self.assertEqual(b.names,['base_turn','tool_turn'])
            session=ExperimentSession(b,root/'logs',b.platform_metadata)
            session.set_pose([.1,.2]);session.save_pose('A')
            other=build_experiment(c,root,('mini','motion','base','tool'))
            loaded=ExperimentSession(other,root/'other',{})
            loaded.load_poses(session.saved_path)
            session.switch_mode('measure');session.start([dict(name='A',seconds=.2,hold=0)])
            for _ in range(3000):
                session.tick()
                if not session.active:break
            self.assertEqual(session.state,'COMPLETED')
            other.pose_context['domain']='different'
            with self.assertRaises(ValueError):loaded.load_poses(session.saved_path)
            c['robots']['mini']['excluded_joints']=['tool_turn']
            with self.assertRaises(ValueError):build_experiment(c,root,('mini','motion','base','tool'))

    def test_domain_changes_and_scope_validation(self):
        c,base=load_catalog(ROOT/'configs/manipulation_platform.yaml')
        b=build_experiment(c,base,('prototype_v211','waypoint_motion','base','left'),['waist','left_arm'])
        self.assertEqual(len(b.names),9)
        self.assertFalse(b.presets)
        with self.assertRaises(ValueError):build_experiment(c,base,('prototype_v211','waypoint_motion','base','left'),['right_arm'])
        c['domains']['bad']={'kind':'pick_place'}
        with self.assertRaises(ValueError):build_experiment(c,base,('prototype_v211','bad','base','right'))

if __name__=='__main__':unittest.main()
