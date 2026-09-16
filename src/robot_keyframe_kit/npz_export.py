"""Export recorded MuJoCo poses to the Holosoma motion array schema."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
from scipy.spatial.transform import Rotation
import numpy as np


def build_holosoma_motion(model, qpos, dt, root_body, *, xml_path=None, urdf_path=None):
    """Recompute world kinematics without modifying the editor's live MjData.

    Velocities are finite differences of the saved poses, with a backward
    difference at the final frame. Bodies and joints use breadth-first order.
    Only the floating robot subtree is exported (no MuJoCo world body).
    """
    poses = np.array(qpos, dtype=np.float64, copy=True)
    if poses.ndim != 2 or poses.shape[1] != model.nq or len(poses) < 2:
        raise ValueError(f"Run a trajectory test first: need at least 2 poses with {model.nq} columns.")
    if not np.isfinite(poses).all() or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Poses and recording interval must be finite; interval must be positive.")
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if root <= 0 or len(free) != 1 or model.jnt_bodyid[free[0]] != root:
        raise ValueError("Holosoma export requires one free joint on the configured root body.")
    bodies = [root]
    for parent in bodies:
        bodies.extend(i for i in range(1, model.nbody) if model.body_parentid[i] == parent)
    joints = [j for b in bodies for j in range(model.njnt)
              if model.jnt_bodyid[j] == b and j != free[0]]
    if any(int(model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
           for j in joints):
        raise ValueError("Only scalar robot joints are supported after the floating base.")
    joint_names = [model.joint(j).name for j in joints]
    body_names = [model.body(b).name for b in bodies]
    if not all(joint_names + body_names):
        raise ValueError("Exported joints and bodies must have names.")
    qa, va = int(model.jnt_qposadr[free[0]]), int(model.jnt_dofadr[free[0]])
    norms = np.linalg.norm(poses[:, qa + 3:qa + 7], axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("Root quaternion has zero length.")
    poses[:, qa + 3:qa + 7] /= norms[:, None]
    velocities = np.zeros((len(poses), model.nv), dtype=np.float64)
    for i in range(len(poses) - 1):
        mujoco.mj_differentiatePos(model, velocities[i], dt, poses[i], poses[i + 1])
    velocities[-1] = velocities[-2]
    qcols = list(range(qa, qa + 7)) + [int(model.jnt_qposadr[j]) for j in joints]
    vcols = list(range(va, va + 6)) + [int(model.jnt_dofadr[j]) for j in joints]
    shape = (len(poses), len(bodies))
    result = {
        "fps": np.array([1.0 / dt], dtype=np.float64),
        "joint_names": np.asarray(joint_names),
        "body_names": np.asarray(body_names),
        "joint_pos": poses[:, qcols].astype(np.float32),
        "joint_vel": velocities[:, vcols].astype(np.float32),
        "body_pos_w": np.empty((*shape, 3), dtype=np.float32),
        "body_quat_w": np.empty((*shape, 4), dtype=np.float32),
        "body_lin_vel_w": np.empty((*shape, 3), dtype=np.float32),
        "body_ang_vel_w": np.empty((*shape, 3), dtype=np.float32),
        "motion_format": np.array(["holosoma"]),
    }
    if np.isclose(1.0 / dt, round(1.0 / dt), rtol=0, atol=1e-9):
        result["fps"] = np.array([round(1.0 / dt)], dtype=np.int64)
    data = mujoco.MjData(model)
    velocity = np.empty(6, dtype=np.float64)
    for frame in range(len(poses)):
        data.qpos[:] = poses[frame]
        data.qvel[:] = velocities[frame]
        mujoco.mj_forward(model, data)
        result["body_pos_w"][frame] = data.xpos[bodies]
        result["body_quat_w"][frame] = data.xquat[bodies]
        for column, body in enumerate(bodies):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            result["body_ang_vel_w"][frame, column] = velocity[:3]
            result["body_lin_vel_w"][frame, column] = velocity[3:]
    if urdf_path is None and xml_path is not None:
        urdf_path = find_robot_urdf(xml_path)
    if urdf_path is not None:
        result = add_urdf_fixed_links(result, urdf_path, root_body)
    return result


def find_robot_urdf(xml_path):
    """Find the robot URDF next to its mjcf directory, including scene XML inputs."""
    xml_path = Path(xml_path).resolve()
    directory = xml_path.parent.parent / "urdf"
    candidates = sorted(directory.glob("*.urdf"))
    for basename in (xml_path.stem, directory.parent.name):
        match = directory / (basename + ".urdf")
        if match in candidates:
            return match
    if len(candidates) > 1:
        raise ValueError(f"Multiple URDF files in {directory}; use a matching robot XML/URDF basename.")
    return candidates[0] if candidates else None


def add_urdf_fixed_links(motion, urdf_path, root_body):
    """Recover missing fixed children using world poses and v_child = v_parent + omega x r.

    Existing body arrays and all joint arrays are preserved. Only the configured
    robot subtree is considered; disconnected scene objects are not exported.
    """
    tree = ET.parse(urdf_path).getroot()
    children = {}
    for joint in tree.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    if root_body not in {link.get("name") for link in tree.findall("link")}:
        raise ValueError(f"Root body {root_body!r} is missing from {urdf_path}")
    names = motion["body_names"].tolist()
    result = dict(motion)
    queue = [root_body]
    visited = set()
    for parent in queue:
        if parent in visited:
            raise ValueError("URDF contains a cycle or repeated child link")
        visited.add(parent)
        for joint in children.get(parent, []):
            child = joint.find("child").get("link")
            queue.append(child)
            if child in names:
                continue
            if joint.get("type") != "fixed":
                raise ValueError(f"Moving URDF link {child!r} is missing from the MuJoCo motion")
            if parent not in names:
                raise ValueError(f"Cannot recover {child!r}: parent {parent!r} is missing")
            index = names.index(parent)
            origin = joint.find("origin")
            attributes = {} if origin is None else origin.attrib
            xyz = np.fromstring(attributes.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(attributes.get("rpy", "0 0 0"), sep=" ")
            rotation = Rotation.from_quat(result["body_quat_w"][:, index][:, [1, 2, 3, 0]])
            offset = rotation.apply(xyz)
            orientation = (rotation * Rotation.from_euler("xyz", rpy)).as_quat()[:, [3, 0, 1, 2]]
            omega = result["body_ang_vel_w"][:, index]
            values = {
                "body_pos_w": result["body_pos_w"][:, index] + offset,
                "body_quat_w": orientation,
                "body_lin_vel_w": result["body_lin_vel_w"][:, index] + np.cross(omega, offset),
                "body_ang_vel_w": omega,
            }
            for key, value in values.items():
                result[key] = np.concatenate((result[key], value[:, None].astype(result[key].dtype)), axis=1)
            names.append(child)
    result["body_names"] = np.asarray(names)
    return result
