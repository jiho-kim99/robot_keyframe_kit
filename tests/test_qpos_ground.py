"""Qpos preview preserves interpolated root values without automatic lifting."""
import unittest
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np

from robot_keyframe_kit.editor import ViserKeyframeEditor


class QposGroundTest(unittest.TestCase):
    def test_valid_endpoints_penetrating_middle(self):
        model = mujoco.MjModel.from_xml_string('''
        <mujoco><worldbody>
          <geom type="plane" size="3 3 .1"/>
          <body name="robot"><freejoint/>
            <geom type="sphere" size=".05"/>
            <body name="leg"><joint axis="0 1 0"/>
              <geom type="capsule" fromto="0 0 0 0 0 -1" size=".1"/>
            </body>
          </body>
          <body name="environment" pos="4 0 -1"><geom type="box" size=".2 .2 .2"/></body>
        </worldbody></mujoco>''')
        editor = ViserKeyframeEditor.__new__(ViserKeyframeEditor)
        editor.model = model
        editor.data = mujoco.MjData(model)
        editor._com_root_body_id = model.body('robot').id
        initial = editor.data.qpos.copy()
        start = initial.copy()
        start[2], start[7] = .6, -np.pi / 3
        end = start.copy()
        end[7] = np.pi / 3
        frames = editor._interpolate_qpos_trajectory([0, 1], [start, end], np.linspace(0, 1, 51))
        self.assertAlmostEqual(frames[25][2], .6)
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = frames[25]
        mujoco.mj_forward(model, scratch)
        self.assertIn(model.body('leg').id, editor._ground_penetration_depths(scratch))
        # Playback must include penetrating frames unchanged, without physics.
        editor.worker_lock = threading.Lock()
        editor.worker = SimpleNamespace(is_testing=False, request_trajectory_test=Mock())
        editor.sequence_list = [('start', 0.0), ('end', 1.0)]
        editor.keyframes = [SimpleNamespace(name='start', qpos=start), SimpleNamespace(name='end', qpos=end)]
        editor.selected_sequence = 0
        editor.relative_frame_checked = None
        editor.dt = .02
        editor._test_qpos_trajectory()
        call = editor.worker.request_trajectory_test.call_args
        self.assertEqual(len(call.args[1]), 51)
        np.testing.assert_allclose(call.args[1], frames, atol=1e-8)
        self.assertFalse(call.kwargs['physics_enabled'])
        self.assertTrue(call.kwargs['is_qpos_traj'])
        np.testing.assert_allclose(frames[0], start, atol=1e-8)
        np.testing.assert_allclose(frames[-1], end, atol=1e-8)
        np.testing.assert_array_equal(editor.data.qpos, initial)
        scratch = mujoco.MjData(model)
        for q in frames:
            scratch.qpos[:] = q
            mujoco.mj_forward(model, scratch)
            self.assertAlmostEqual(np.linalg.norm(q[3:7]), 1)
        # Opposite quaternion signs represent the same orientation, not a zero quaternion midway.
        opposite = start.copy()
        opposite[3:7] *= -1
        q = editor._interpolate_qpos_trajectory([0, 1], [start, opposite], [.5])[0]
        self.assertAlmostEqual(np.linalg.norm(q[3:7]), 1)


if __name__ == '__main__':
    unittest.main()
