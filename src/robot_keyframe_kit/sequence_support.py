"""Kinematic sequence interpolation with an optional world-fixed support frame."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def support_frames(model):
    """Only expose frames belonging to the one free-root robot."""
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        return {}
    root = int(model.jnt_bodyid[free[0]])
    frames = {}
    for kind, count, obj in (("site", model.nsite, mujoco.mjtObj.mjOBJ_SITE),
                             ("body", model.nbody, mujoco.mjtObj.mjOBJ_BODY)):
        for index in range(count):
            body = int(model.site_bodyid[index]) if kind == "site" else index
            while body and body != root:
                body = int(model.body_parentid[body])
            name = mujoco.mj_id2name(model, obj, index)
            if body == root and name:
                frames[f"{kind}:{name}"] = (kind, index)
    return frames


def interpolate_supported(model, times, keyframes, sample_times, supports):
    """Preserve each selected frame's pose at its corrected segment start.

    Joint angles are unchanged by support correction. Corrected boundary poses
    feed the next segment so switching anchors cannot teleport the root.
    Free segments interpolate from the corrected boundary to the saved end pose.
    """
    times = np.asarray(times, dtype=float)
    if len(times) != len(keyframes) or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Sequence requires matching keyframes and increasing times")
    available = support_frames(model)
    supports = list(supports)
    if len(supports) != len(times) - 1:
        raise ValueError("One support setting is required per sequence segment")
    for support in supports:
        if support and support not in available:
            raise ValueError(f"Unknown robot support frame: {support}")
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    root_addr = int(model.jnt_qposadr[free[0]]) if len(free) == 1 else None
    data = mujoco.MjData(model)

    def frame_pose(q, support):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        kind, index = available[support]
        pos = data.site_xpos[index] if kind == "site" else data.xpos[index]
        mat = data.site_xmat[index] if kind == "site" else data.xmat[index]
        return pos.copy(), mat.reshape(3, 3).copy()

    def blend(start, end, fraction, support, anchor):
        q = np.array(start, dtype=float, copy=True)
        end = np.array(end, dtype=float, copy=True)
        mujoco.mj_normalizeQuat(model, q)
        mujoco.mj_normalizeQuat(model, end)
        velocity = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, velocity, 1.0, q, end)
        mujoco.mj_integratePos(model, q, velocity, fraction)
        if support:
            pos, mat = frame_pose(q, support)
            rotation = anchor[1] @ mat.T
            a = root_addr
            q[a:a + 3] = anchor[0] + rotation @ (q[a:a + 3] - pos)
            root_rot = Rotation.from_quat(q[a + 3:a + 7], scalar_first=True)
            q[a + 3:a + 7] = (Rotation.from_matrix(rotation) * root_rot).as_quat(scalar_first=True)
        return q

    starts, anchors = [], []
    start = np.array(keyframes[0], dtype=float, copy=True)
    for i, support in enumerate(supports):
        anchor = frame_pose(start, support) if support else None
        starts.append(start)
        anchors.append(anchor)
        start = blend(start, keyframes[i + 1], 1.0, support, anchor)
    result = []
    for t in sample_times:
        i = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2))
        fraction = float(np.clip((t - times[i]) / (times[i + 1] - times[i]), 0, 1))
        result.append(blend(starts[i], keyframes[i + 1], fraction, supports[i], anchors[i]))
    return result
