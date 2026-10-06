"""Regression tests for the independent-arm bench, using a small physical fixture."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

MODULE = Path(__file__).resolve().parents[1] / 'scripts' / 'arm_payload.py'
if not MODULE.exists():
    MODULE = Path(__file__).with_name('arm_payload.py')
spec = importlib.util.spec_from_file_location('arm_payload', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    bodies = ''
    for i, part in enumerate(module.PARTS):
        name = 'right_wrist_pitch_link' if i == 6 else f'link{i}'
        axis = '0 1 0' if i % 2 == 0 else '0 0 1'
        bodies += f'<body name="{name}" pos="0.1 0 0"><joint name="right_{part}_joint" axis="{axis}" range="-2 2" actuatorfrcrange="-30 30"/><geom type="sphere" size="0.025" mass="0.1"/>'
    bodies += '<site name="right_welding_point" pos="0.1 0 0"/>' + '</body>'*7
    return '<mujoco><compiler angle="radian"/><default><joint damping="0.2" armature="0.05"/></default><worldbody><body><freejoint/><geom type="sphere" size="0.05" mass="1"/>'+bodies+'</body></worldbody></mujoco>'


class ArmPayloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.xml = Path(self.tmp.name)/'fixture.xml'
        self.xml.write_text(fixture())

    def bench(self, payload=0):
        args = module.build_model(self.xml, 'right', payload=payload)
        return module.Bench(*args, kp=200, kd=20, gravity_comp=True)

    def test_only_arm_is_dynamic_and_payload_has_mass(self):
        b, loaded = self.bench(), self.bench(2)
        self.assertEqual(b.model.nv, 7)
        self.assertAlmostEqual(loaded.model.body_mass.sum()-b.model.body_mass.sum(), 2)
        self.assertEqual(loaded.model.nu, 7)

    def test_torque_clipping_and_velocity_logging(self):
        b = self.bench()
        b.target[:] = 2
        row, torque, _, _ = b.sample()
        self.assertTrue(np.all(np.abs(torque) <= 30))
        self.assertTrue(np.any(row[-7:] == 1))
        b.data.qvel[:] = 0.2
        self.assertTrue(np.allclose(b.sample()[3], 0.2))

    def test_ik_reaches_and_holds_reference(self):
        b = self.bench()
        scratch = mujoco.MjData(b.model)
        scratch.qpos[:] = 0.12
        mujoco.mj_forward(b.model, scratch)
        target = scratch.site_xpos[b.sid].copy()
        b.set_ik(target, 0.5)
        for _ in range(1250):
            b.sample()
            mujoco.mj_step(b.model, b.data)
        qref, vref = b.reference()
        self.assertTrue(np.allclose(qref, b.target))
        self.assertTrue(np.allclose(vref, 0))
        mujoco.mj_forward(b.model, b.data)
        self.assertLess(np.linalg.norm(b.data.site_xpos[b.sid]-target), 0.005)
        self.assertLess(np.max(np.abs(b.data.qvel)), 0.01)

    def test_slide_direct_pose_ignores_pd_and_gravity(self):
        b = self.bench(payload=2)
        b.target[:] = 0.25
        b.data.qvel[:] = 3
        b.data.ctrl[:] = 10
        for _ in range(20):
            row, torque, xyz, velocity = b.sample('slide')
            b.advance('slide')
        np.testing.assert_allclose(b.data.qpos[b.qadr], 0.25)
        np.testing.assert_array_equal(torque, 0)
        np.testing.assert_array_equal(velocity, 0)
        np.testing.assert_array_equal(b.data.ctrl, 0)
        self.assertGreater(b.data.time, 0)
        scratch = mujoco.MjData(b.model)
        scratch.qpos[b.qadr] = 0.25
        mujoco.mj_forward(b.model, scratch)
        np.testing.assert_allclose(xyz, scratch.site_xpos[b.sid])

    def test_slide_frame_solves_position_and_orientation(self):
        b = self.bench()
        scratch = mujoco.MjData(b.model)
        scratch.qpos[:] = np.linspace(0.05, 0.2, b.model.nq)
        mujoco.mj_forward(b.model, scratch)
        xyz = scratch.site_xpos[b.sid].copy()
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, scratch.site_xmat[b.sid])
        before = b.data.qpos.copy()
        b.target = b.solve_slide_pose(xyz, quat)
        np.testing.assert_array_equal(b.data.qpos, before)
        _, torque, actual, _ = b.sample('slide')
        np.testing.assert_allclose(actual, xyz, atol=0.005)
        np.testing.assert_allclose(b.data.site_xmat[b.sid], scratch.site_xmat[b.sid], atol=0.03)
        np.testing.assert_array_equal(torque, 0)
        with self.assertRaises(ValueError):
            b.solve_slide_pose([100, 100, 100], quat)
        np.testing.assert_allclose(b.data.site_xpos[b.sid], actual)

    def test_unreachable_ik_does_not_change_target(self):
        b = self.bench()
        initial = b.target.copy()
        with self.assertRaises(ValueError):
            b.set_ik([100, 100, 100], 1)
        np.testing.assert_array_equal(b.target, initial)


class ResidualTests(unittest.TestCase):
    setUp = ArmPayloadTests.setUp
    def bench(self, payload=0):
        return module.ResidualBench(*module.build_model(self.xml, 'right', payload=payload), kp=200, kd=20)

    def command_for(self, b, q, frame):
        scratch = mujoco.MjData(b.model)
        scratch.qpos[:] = b.data.qpos
        scratch.qpos[b.qadr] = q
        mujoco.mj_forward(b.model, scratch)
        xyz, quat = b.ee_pose()
        current = module.Rotation.from_quat(np.roll(quat,-1))
        target = module.Rotation.from_matrix(scratch.site_xmat[b.sid].reshape(3,3))
        delta = scratch.site_xpos[b.sid]-xyz
        if frame == 'World':
            rotation = target*current.inv()
        else:
            delta = current.inv().apply(delta)
            rotation = current.inv()*target
        return delta*1000, np.rad2deg(rotation.as_rotvec())

    def test_sequential_relative_world_and_local_moves(self):
        b = self.bench(payload=2)
        for frame in ('World','EE local'):
            trans, rot = self.command_for(b, b.data.qpos[b.qadr]+0.03, frame)
            start_xyz = b.ee_pose()[0]
            r = b.begin_move(trans, rot, .5, frame)
            np.testing.assert_allclose(r['start_ee_position_m'], start_xyz)
            with self.assertRaises(ValueError):
                b.begin_move([0,0,1],[0,0,0],1)
            for _ in range(1500):
                row, torque, xyz, velocity = b.sample()
                b.advance()
                if b.phase == 'READY': break
            self.assertEqual(b.phase,'READY')
            self.assertLess(b.pos_error,.003)
            self.assertLess(np.max(np.abs(velocity)),.02)
            self.assertTrue(np.all(np.isfinite(row)))
        self.assertEqual(b.command_id,2)
        self.assertEqual(len(b.records),2)

    def test_reject_bad_commands_without_changing_hold(self):
        b = self.bench()
        before = b.target.copy()
        for trans, seconds in (([float('nan'),0,0],1),([0,0,1],0),([100000,0,0],1)):
            with self.assertRaises(ValueError): b.begin_move(trans,[0,0,0],seconds)
        np.testing.assert_array_equal(b.target,before)
        self.assertEqual(b.command_id,0)

    def test_curl_sequence_preserves_joint_poses_and_segment_times(self):
        b = self.bench(payload=2)
        points = b.curl_presets()
        seconds = [.5,.7,.9]
        b.start_sequence([dict(joints=q,seconds=t,label=label) for q,t,label in zip(points,seconds,'ABC')])
        for _ in range(10000):
            b.advance_sequence()
            b.sample()
            b.advance()
            if len(b.records)==3 and b.phase=='READY' and not b.sequence: break
        self.assertEqual(b.phase,'READY')
        self.assertEqual([r['seconds'] for r in b.records],seconds)
        self.assertEqual([r['label'] for r in b.records],list('ABC'))
        for record,expected in zip(b.records,points):
            np.testing.assert_allclose(record['target_joint_rad'],expected)
        np.testing.assert_allclose(b.data.qpos[b.qadr],points[0],atol=.003)

    def test_sequence_validation_is_atomic(self):
        b = self.bench()
        with self.assertRaises(ValueError):
            b.start_sequence([dict(joints=b.target,seconds=1,label='A'),
                              dict(joints=b.target,seconds=-1,label='B')])
        self.assertEqual(b.command_id,0)
        self.assertFalse(b.sequence)

    def test_inverse_hold_and_saturation(self):
        b = self.bench(payload=2)
        row,torque,_,_=b.sample()
        np.testing.assert_allclose(torque,b.last_inverse)
        self.assertGreater(np.max(np.abs(torque)),0)
        b.limits[:]=[-.01,.01]
        row,torque,_,_=b.sample()
        self.assertTrue(np.all(np.abs(torque)<=.01+1e-10))
        self.assertTrue(np.any(row[-7:]))


if __name__ == '__main__':
    unittest.main()
