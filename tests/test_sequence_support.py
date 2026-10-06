import unittest
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np

from robot_keyframe_kit.sequence_support import interpolate_supported, support_frames
from robot_keyframe_kit.npz_export import build_holosoma_motion
from robot_keyframe_kit.editor import ViserKeyframeEditor


class SequenceSupportTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string('''
        <mujoco><worldbody>
          <body name="pelvis" pos="0 0 1"><freejoint/>
            <geom type="sphere" size=".1"/>
            <body name="leg"><joint name="hip" axis="0 1 0"/>
              <geom type="capsule" fromto="0 0 0 0 0 -1" size=".05"/>
              <site name="foot" pos="0 0 -1"/>
            </body>
          </body>
          <site name="world_site"/>
        </worldbody></mujoco>''')
        self.data = mujoco.MjData(self.model)
        self.start = self.data.qpos.copy()
        self.end = self.start.copy()
        self.end[7] = 1.0
        self.end[:3] += [1, 2, 3]

    def pose(self, q):
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[0].copy(), self.data.site_xmat[0].copy()

    def test_support_fixed_joints_unchanged_and_export_consistent(self):
        times = np.linspace(0, 1, 101)
        frames = interpolate_supported(self.model, [0, 1], [self.start, self.end], times, ['site:foot'])
        anchor = self.pose(self.start)
        for t, q in zip(times, frames):
            pos, rot = self.pose(q)
            np.testing.assert_allclose(pos, anchor[0], atol=1e-12)
            np.testing.assert_allclose(rot, anchor[1], atol=1e-12)
            self.assertAlmostEqual(q[7], t)
        exported = build_holosoma_motion(self.model, frames, .01, 'pelvis')
        np.testing.assert_allclose(exported['body_pos_w'][:, 0], np.array(frames)[:, :3], atol=1e-6)
        self.assertTrue(np.isfinite(exported['body_lin_vel_w']).all())

    def test_free_matches_saved_endpoints(self):
        frames = interpolate_supported(self.model, [0, 1], [self.start, self.end], [0, .5, 1], [None])
        np.testing.assert_allclose(frames[0], self.start)
        np.testing.assert_allclose(frames[-1], self.end)
        np.testing.assert_allclose(frames[1][:3], (self.start[:3] + self.end[:3]) / 2)

    def test_switching_support_and_free_is_continuous(self):
        final = self.start.copy()
        final[7] = -.5
        for next_support in ('body:pelvis', None):
            frames = interpolate_supported(self.model, [0, 1, 2], [self.start, self.end, final],
                                           [1 - 1e-8, 1, 1 + 1e-8, 2], ['site:foot', next_support])
            np.testing.assert_allclose(frames[0], frames[1], atol=1e-7)
            np.testing.assert_allclose(frames[1], frames[2], atol=1e-7)
            # Starting playback later must not re-anchor at the saved raw root.
            only_later = interpolate_supported(self.model, [0, 1, 2], [self.start, self.end, final],
                                               [1, 2], ['site:foot', next_support])
            np.testing.assert_allclose(only_later, [frames[1], frames[3]])

    def test_invalid_environment_frame_rejected(self):
        self.assertNotIn('site:world_site', support_frames(self.model))
        with self.assertRaises(ValueError):
            interpolate_supported(self.model, [0, 1], [self.start, self.end], [0], ['site:world_site'])

    def test_editor_passes_corrected_sequence_to_worker(self):
        editor = ViserKeyframeEditor.__new__(ViserKeyframeEditor)
        editor.model = self.model
        editor.worker_lock = threading.Lock()
        editor.worker = SimpleNamespace(is_testing=False, request_trajectory_test=Mock())
        editor.sequence_list = [('a', 1.), ('b', 2.)]
        editor.sequence_supports = ['site:foot', None]
        editor.keyframes = [SimpleNamespace(name='a', qpos=self.start),
                            SimpleNamespace(name='b', qpos=self.end)]
        editor.selected_sequence = None
        editor.relative_frame_checked = None
        editor.dt = .01
        editor._test_qpos_trajectory()
        call = editor.worker.request_trajectory_test.call_args
        self.assertFalse(call.kwargs['physics_enabled'])
        anchor = self.pose(self.start)
        for q in call.args[1]:
            np.testing.assert_allclose(self.pose(q)[0], anchor[0], atol=1e-12)

    def test_sequence_reordering_removal_and_legacy_defaults(self):
        editor = ViserKeyframeEditor.__new__(ViserKeyframeEditor)
        editor.sequence_list = [('a', 0.), ('b', 1.), ('a', 2.)]
        editor.selected_sequence = None
        editor._refresh_sequence_table = Mock()
        editor._normalize_sequence_supports()
        self.assertEqual(editor.sequence_supports, [None, None, None])
        editor.sequence_supports = ['site:foot', 'body:pelvis', None]
        editor._reorder_sequence(0, 1)
        self.assertEqual(editor.sequence_supports, ['body:pelvis', 'site:foot', None])
        editor.selected_sequence = 1
        editor._remove_from_sequence()
        self.assertEqual(editor.sequence_supports, ['body:pelvis', None])
        self.assertEqual(len(editor.sequence_supports), len(editor.sequence_list))


if __name__ == '__main__':
    unittest.main()
