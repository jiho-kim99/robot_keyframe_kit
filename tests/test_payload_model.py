import sys
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from arm_payload import select_payload_model,build_model

class HeavyWristTests(unittest.TestCase):
    def test_select_and_no_double_payload(self):
        for kg in (10,15,20):
            with self.subTest(kg=kg):self.check_model(kg)

    def check_model(self,kg):
        base=ROOT/'prototype_v2.1.1/mjcf/scene.xml'
        xml,added,description=select_payload_model(base,'right',kg,(0,0,0))
        self.assertEqual(xml.name,f'scene_wrist{kg}kg.xml')
        self.assertEqual(added,0)
        m,d,*_=build_model(xml,'right',payload=added)
        original=mujoco.MjModel.from_xml_path(str(base))
        a=original.body('right_wrist_pitch_link');b=m.body('right_wrist_pitch_link')
        self.assertEqual(m.body_mass[b.id],kg)
        expected_com=original.body_ipos[a.id].copy();expected_com[1]=-0.0315
        np.testing.assert_allclose(m.body_ipos[b.id],expected_com)
        np.testing.assert_allclose(m.body_inertia[b.id],(kg/0.2)*original.body_inertia[a.id])
        np.testing.assert_allclose(m.body_iquat[b.id],original.body_iquat[a.id])
        self.assertEqual(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'bench_payload'),-1)
        self.assertAlmostEqual(m.body_mass.sum()-original.body_mass.sum(),kg-0.2)
        self.assertEqual(select_payload_model(xml,'right',kg,(0,0,0))[1],0)
        with self.assertRaises(ValueError):select_payload_model(xml,'right',15 if kg==10 else 10,(0,0,0))
        with self.assertRaises(ValueError):select_payload_model(base,'left',kg,(0,0,0))
        with self.assertRaises(ValueError):select_payload_model(base,'right',kg,(.1,0,0))

    def test_urdf_mass_and_full_tensor(self):
        for kg in (10,15,20):
            with self.subTest(kg=kg):self.check_urdf(kg)

    def check_urdf(self,kg):
        folder=ROOT/'prototype_v2.1.1/urdf'
        def inertial(name):
            return ET.parse(folder/name).find(".//link[@name='right_wrist_pitch_link']/inertial")
        old=inertial('prototype_v2.1.1.urdf');new=inertial(f'prototype_v2.1.1_wrist{kg}kg.urdf')
        self.assertEqual(float(new.find('mass').get('value')),kg)
        expected_com=old.find('origin').get('xyz').split();expected_com[1]='-0.0315'
        self.assertEqual(new.find('origin').get('xyz'),' '.join(expected_com))
        self.assertEqual(old.find('origin').get('rpy'),new.find('origin').get('rpy'))
        for key,value in old.find('inertia').attrib.items():
            self.assertAlmostEqual(float(new.find('inertia').get(key)),(kg/0.2)*float(value))

if __name__=='__main__':unittest.main()
