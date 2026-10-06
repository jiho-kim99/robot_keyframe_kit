#!/usr/bin/env python3
"""Single-arm MuJoCo PD payload bench. Run directly; see docs/arm_payload.md."""
from __future__ import annotations

import argparse
import csv
import json
import time
import threading
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

PARTS = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow_pitch',
         'wrist_roll', 'wrist_yaw', 'wrist_pitch')


def select_payload_model(xml, arm, payload, payload_com):
    """Select a total-mass wrist variant without adding a second payload."""
    xml = Path(xml).resolve()
    model_dir = Path(__file__).resolve().parents[1]/'prototype_v2.1.1/mjcf'
    masses = (10,15,20)
    bases = { (model_dir/'scene.xml').resolve(): 'scene',
              (model_dir/'prototype_v2.1.1.xml').resolve(): 'prototype_v2.1.1' }
    heavy = {(model_dir/f'{stem}_wrist{kg}kg.xml').resolve(): kg
             for stem in bases.values() for kg in masses}
    if payload in masses or xml in heavy:
        if arm != 'right':
            raise ValueError('wrist 질량 변경 모델은 right 팔 전용입니다.')
        if any(float(x) != 0 for x in payload_com):
            raise ValueError('모델 파일에 설정된 COM을 사용합니다. --payload-com을 생략하세요.')
        if xml in heavy:
            kg = heavy[xml]
            if payload not in (0,kg):
                raise ValueError('XML의 링크 질량과 --payload-kg가 다릅니다. 별도 payload는 추가하지 않습니다.')
        elif xml in bases:
            kg = int(payload)
            xml = model_dir/f'{bases[xml]}_wrist{kg}kg.xml'
        else:
            raise ValueError('wrist 질량 변경은 기본 scene.xml 또는 제공된 wrist XML을 사용하세요.')
        if not xml.is_file():
            raise ValueError(f'wrist XML을 찾을 수 없습니다: {xml}')
        return xml, 0.0, f'right_wrist_pitch_link total mass {kg}kg; COM y=-0.0315m (link frame); inertia scaled {kg/0.2:g}x; no added payload'
    return xml, payload, 'Rigid sphere at site-local payload_com; no payload contact'


def build_model(xml, arm, ee_site=None, payload=0.0, timestep=0.002, payload_com=(0, 0, 0), payload_radius=0.04):
    spec = mujoco.MjSpec.from_file(str(Path(xml).resolve()))
    original = spec.compile()
    names = [f'{arm}_{part}_joint' for part in PARTS]
    for name in names:
        if mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_JOINT, name) < 0:
            raise ValueError(f'Missing arm joint: {name}')
    # This bench deliberately fixes all non-arm DOFs at the XML reference pose.
    # Reject coupled models rather than silently changing their transmission.
    if list(spec.equalities) or list(spec.tendons):
        raise ValueError('This bench supports independent joints, not equality/tendon transmissions.')
    limits = []
    for name in names:
        joint = original.joint(name)
        if not original.jnt_actfrclimited[joint.id]:
            raise ValueError(f'{name}: explicit joint actuatorfrcrange is required.')
        limits.append(original.jnt_actfrcrange[joint.id].copy())
    for collection in (spec.actuators, spec.sensors, spec.keys):
        for element in list(collection):
            spec.delete(element)
    for joint in list(spec.joints):
        if joint.name not in names:
            spec.delete(joint)
    for name, limit in zip(names, limits):
        spec.add_actuator(name=f'bench_{name}', target=name,
                          trntype=mujoco.mjtTrn.mjTRN_JOINT,
                          gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[1] + [0]*9,
                          biastype=mujoco.mjtBias.mjBIAS_NONE,
                          dyntype=mujoco.mjtDyn.mjDYN_NONE, gear=[1, 0, 0, 0, 0, 0],
                          ctrllimited=True, ctrlrange=limit,
                          forcelimited=True, forcerange=limit)
    body_name = f'{arm}_wrist_pitch_link'
    added_site = None
    if ee_site is None:
        candidate = f'{arm}_welding_point'
        ee_site = candidate if spec.site(candidate) is not None else 'bench_ee'
        if ee_site == 'bench_ee':
            added_site = spec.body(body_name).add_site(name=ee_site, pos=[0, 0, 0], size=[0.01]*3)
    sid = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_SITE, ee_site)
    if sid >= 0:
        bid = int(original.site_bodyid[sid])
        ancestor = bid
        wrist = original.body(body_name).id
        while ancestor and ancestor != wrist:
            ancestor = int(original.body_parentid[ancestor])
        if ancestor != wrist:
            raise ValueError('EE site must be attached to the selected wrist or its descendants.')
        body_name = original.body(bid).name
    site = added_site if added_site is not None else spec.site(ee_site)
    if site is None:
        raise ValueError(f'Unknown EE site: {ee_site}')
    if payload > 0:
        site_rotation = Rotation.from_quat(np.roll(original.site_quat[sid], -1)) if sid >= 0 else Rotation.identity()
        com = np.asarray(site.pos) + site_rotation.apply(np.asarray(payload_com))
        body = spec.body(body_name).add_body(name='bench_payload', pos=com)
        body.add_geom(name='bench_payload_geom', type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[payload_radius, 0, 0], mass=payload, rgba=[1, 0.5, 0, 1],
                      contype=0, conaffinity=0)
    model = spec.compile()
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    model.opt.timestep = timestep
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, names, ee_site, np.asarray(limits)


class Bench:
    def __init__(self, model, data, names, ee_site, limits, kp, kd, gravity_comp=False):
        self.model, self.data, self.names = model, data, names
        self.sid = model.site(ee_site).id
        self.qadr = np.array([model.joint(n).qposadr[0] for n in names])
        self.dadr = np.array([model.joint(n).dofadr[0] for n in names])
        self.bounds = np.array([model.joint(n).range for n in names])
        self.limits, self.kp, self.kd = limits, kp, kd
        self.gravity_comp = gravity_comp
        self.gain_changes = []
        self.start = data.qpos[self.qadr].copy()
        self.target = self.start.copy()
        self.target_xyz = data.site_xpos[self.sid].copy()
        self.seconds = None

    def set_ik(self, xyz, seconds):
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        def residual(q):
            scratch.qpos[self.qadr] = q
            mujoco.mj_forward(self.model, scratch)
            return scratch.site_xpos[self.sid] - xyz
        def jac(q):
            residual(q)
            j = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, scratch, j, None, self.sid)
            return j[:, self.dadr]
        result = least_squares(residual, self.start, jac=jac,
                               bounds=(self.bounds[:, 0], self.bounds[:, 1]),
                               max_nfev=1000, gtol=1e-10, ftol=1e-10, xtol=1e-10)
        error = np.linalg.norm(residual(result.x))
        if error > 0.001:
            raise ValueError(f'IK target not reached: residual {error:.6f} m (limit 0.001 m).')
        self.target, self.target_xyz = result.x, np.asarray(xyz)
        self.seconds = seconds

    def solve_slide_pose(self, xyz, wxyz):
        """Solve a 6D EE pose from the current pose without mutating simulation data."""
        xyz, wxyz = np.asarray(xyz, dtype=float), np.asarray(wxyz, dtype=float)
        if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(wxyz)) or np.linalg.norm(wxyz) < 1e-12:
            raise ValueError('Invalid EE pose')
        target_rotation = Rotation.from_quat(np.roll(wxyz / np.linalg.norm(wxyz), -1))
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        def residual(q):
            scratch.qpos[self.qadr] = q
            mujoco.mj_forward(self.model, scratch)
            current = Rotation.from_matrix(scratch.site_xmat[self.sid].reshape(3, 3))
            angle = (target_rotation * current.inv()).as_rotvec()
            return np.concatenate((scratch.site_xpos[self.sid] - xyz, 0.4 * angle))
        result = least_squares(residual, self.data.qpos[self.qadr].copy(),
                               bounds=(self.bounds[:, 0], self.bounds[:, 1]),
                               max_nfev=80, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        error = residual(result.x)
        pos_error, angle_error = np.linalg.norm(error[:3]), np.linalg.norm(error[3:])/0.4
        if pos_error > 0.005 or angle_error > 0.03:
            raise ValueError(f'Unreachable pose: {pos_error*1000:.1f} mm / {np.degrees(angle_error):.1f} deg; keeping last pose')
        return result.x

    def reference(self):
        if self.seconds is None:
            return self.target.copy(), np.zeros(len(self.names))
        u = np.clip(self.data.time / self.seconds, 0, 1)
        blend = 10*u**3 - 15*u**4 + 6*u**5
        rate = (30*u**2 - 60*u**3 + 30*u**4) / self.seconds
        return self.start + blend*(self.target-self.start), rate*(self.target-self.start)

    def sample(self, mode='ik'):
        if mode == 'slide':
            # Kinematic posing only: no controller, gravity compensation, or integration.
            self.data.qpos[self.qadr] = np.clip(self.target, self.bounds[:, 0], self.bounds[:, 1])
            self.data.qvel[:] = 0
            self.data.ctrl[:] = 0
            self.data.qfrc_applied[:] = 0
            self.data.xfrc_applied[:] = 0
            mujoco.mj_forward(self.model, self.data)
            q = self.data.qpos[self.qadr].copy()
            zeros = np.zeros(len(self.names))
            xyz = self.data.site_xpos[self.sid].copy()
            row = np.concatenate(([self.data.time], q, zeros, zeros, zeros, q, xyz, zeros))
            if not np.all(np.isfinite(row)):
                raise RuntimeError('Non-finite slider pose.')
            return row, zeros.copy(), xyz, zeros.copy()
        qref, vref = self.reference()
        q = self.data.qpos[self.qadr].copy()
        v = self.data.qvel[self.dadr].copy()
        raw = self.kp*(qref-q) + self.kd*(vref-v)
        if self.gravity_comp:
            # Static gravity only, not velocity-dependent bias compensation.
            scratch = mujoco.MjData(self.model)
            scratch.qpos[:] = self.data.qpos
            mujoco.mj_forward(self.model, scratch)
            raw += scratch.qfrc_bias[self.dadr]
        self.data.ctrl[:] = np.clip(raw, self.limits[:, 0], self.limits[:, 1])
        mujoco.mj_forward(self.model, self.data)
        torque = self.data.qfrc_actuator[self.dadr].copy()
        xyz = self.data.site_xpos[self.sid].copy()
        saturated = (np.abs(raw-torque) > 1e-8).astype(float)
        row = np.concatenate(([self.data.time], q, v, raw, torque, qref, xyz, saturated))
        if not np.all(np.isfinite(row)):
            raise RuntimeError('Non-finite simulation state.')
        return row, torque, xyz, v

    def advance(self, mode='ik'):
        if mode == 'slide':
            # Logging clock only; mj_step would let the posed arm fall under gravity.
            self.data.time += self.model.opt.timestep
        else:
            mujoco.mj_step(self.model, self.data)


class ResidualBench(Bench):
    """Repeated relative EE moves, desired inverse dynamics + PD, then active hold."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_data = mujoco.MjData(self.model)
        self.move_start = 0.0
        self.duration = 1.0
        self.command_id = 0
        self.records = []
        self.sequence = []
        self.phase = 'READY'
        self.last_inverse = np.zeros(len(self.names))
        self.last_vref = np.zeros(len(self.names))
        self.last_aref = np.zeros(len(self.names))
        self.target_xyz, self.target_quat = self.ee_pose()
        self.ref_xyz, self.ref_quat = self.target_xyz.copy(), self.target_quat.copy()
        self.pos_error = self.angle_error = 0.0
        self.settled_since = None

    def ee_pose(self):
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.sid])
        return self.data.site_xpos[self.sid].copy(), quat

    def begin_move(self, translation_mm, rotation_deg, seconds, frame='World'):
        if self.phase != 'READY' or np.max(np.abs(self.data.qvel[self.dadr])) >= 0.02:
            raise ValueError('이동/정착 중입니다. READY가 된 후 다음 이동을 입력하세요.')
        trans = np.asarray(translation_mm, dtype=float)
        angles = np.asarray(rotation_deg, dtype=float)
        if trans.shape != (3,) or angles.shape != (3,) or not np.all(np.isfinite(np.r_[trans, angles, seconds])):
            raise ValueError('이동·회전·시간은 유한한 숫자로 입력하세요.')
        if seconds < 0.1:
            raise ValueError('이동 시간은 0.1초 이상이어야 합니다.')
        if frame not in ('World', 'EE local'):
            raise ValueError('Unknown residual frame')
        start_xyz, start_quat = self.ee_pose()
        rotation = Rotation.from_quat(np.roll(start_quat, -1))
        # Rotation-vector increment: direction is the axis, magnitude is the angle.
        delta = Rotation.from_rotvec(np.deg2rad(angles))
        target_xyz = start_xyz + (trans/1000 if frame == 'World' else rotation.apply(trans/1000))
        target_rotation = delta * rotation if frame == 'World' else rotation * delta
        target_quat = np.roll(target_rotation.as_quat(), 1)
        target = self.solve_slide_pose(target_xyz, target_quat)
        # Tighten endpoint acceptance for repeatable measurement commands.
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        scratch.qpos[self.qadr] = target
        mujoco.mj_forward(self.model, scratch)
        err = np.linalg.norm(scratch.site_xpos[self.sid]-target_xyz)
        rot_err = (target_rotation * Rotation.from_matrix(scratch.site_xmat[self.sid].reshape(3,3)).inv()).magnitude()
        if err > 0.001 or rot_err > 0.01:
            raise ValueError(f'목표 자세 도달 불가: 위치 오차 {err*1000:.2f} mm, 회전 오차 {np.degrees(rot_err):.2f} deg')
        # Commit only after validation; rejected commands leave the old hold target intact.
        self.start = self.data.qpos[self.qadr].copy()
        self.target, self.target_xyz, self.target_quat = target, target_xyz, target_quat
        self.move_start, self.duration = float(self.data.time), float(seconds)
        self.command_id += 1
        self.phase = 'MOVING'
        self.settled_since = None
        record = dict(command_id=self.command_id, start_time_s=self.move_start, seconds=float(seconds),
                      translation_mm=trans.tolist(), rotation_vector_deg=angles.tolist(), frame=frame,
                      start_ee_position_m=start_xyz.tolist(), start_ee_wxyz=start_quat.tolist(),
                      target_ee_position_m=target_xyz.tolist(), target_ee_wxyz=target_quat.tolist(),
                      start_joint_rad=self.start.tolist(), target_joint_rad=target.tolist())
        self.records.append(record)
        return record

    def begin_joint_move(self, joints, seconds, label='joint pose'):
        if self.phase != 'READY' or np.max(np.abs(self.data.qvel[self.dadr])) >= 0.02:
            raise ValueError('READY가 된 후 실행하세요.')
        joints = np.asarray(joints, dtype=float)
        if joints.shape != self.target.shape or not np.all(np.isfinite(joints)):
            raise ValueError('잘못된 관절 자세입니다.')
        if not np.isfinite(seconds) or seconds < .1:
            raise ValueError('이동 시간은 0.1초 이상이어야 합니다.')
        if np.any(joints < self.bounds[:,0]) or np.any(joints > self.bounds[:,1]):
            raise ValueError('관절 범위를 벗어난 자세입니다.')
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        scratch.qpos[self.qadr] = joints
        mujoco.mj_forward(self.model, scratch)
        start_xyz, start_quat = self.ee_pose()
        self.start = self.data.qpos[self.qadr].copy()
        self.target = joints.copy()
        self.target_xyz = scratch.site_xpos[self.sid].copy()
        mujoco.mju_mat2Quat(self.target_quat, scratch.site_xmat[self.sid])
        self.move_start, self.duration = float(self.data.time), float(seconds)
        self.command_id += 1
        self.phase, self.settled_since = 'MOVING', None
        record = dict(command_id=self.command_id, kind='joint_waypoint', label=label,
                      start_time_s=self.move_start, seconds=float(seconds),
                      start_joint_rad=self.start.tolist(), target_joint_rad=joints.tolist(),
                      start_ee_position_m=start_xyz.tolist(), start_ee_wxyz=start_quat.tolist(),
                      target_ee_position_m=self.target_xyz.tolist(), target_ee_wxyz=self.target_quat.tolist())
        self.records.append(record)
        return record

    def start_sequence(self, waypoints):
        if self.sequence or self.phase != 'READY':
            raise ValueError('현재 동작이 끝난 후 시퀀스를 실행하세요.')
        prepared = []
        for point in waypoints:
            q = np.asarray(point['joints'],dtype=float)
            seconds = float(point['seconds'])
            if q.shape != self.target.shape or not np.all(np.isfinite(q)) or not np.isfinite(seconds) or seconds < .1:
                raise ValueError('모든 자세와 구간 시간을 확인하세요.')
            if np.any(q < self.bounds[:,0]) or np.any(q > self.bounds[:,1]):
                raise ValueError('관절 한계를 벗어난 점이 있습니다.')
            prepared.append(dict(joints=q.copy(), seconds=seconds, label=point['label']))
        if not prepared:
            raise ValueError('시퀀스가 비어 있습니다.')
        self.begin_joint_move(**prepared[0])
        self.sequence = prepared[1:]

    def advance_sequence(self):
        if self.phase == 'READY' and self.sequence:
            self.begin_joint_move(**self.sequence[0])
            self.sequence.pop(0)

    def curl_presets(self):
        if not self.names[0].startswith('right_'):
            raise ValueError('기본 컬 자세는 오른팔용입니다. 왼팔은 현재 자세 저장으로 설정하세요.')
        # This XML has a 90-degree elbow bend at q=0; negative q lowers the forearm.
        down = np.deg2rad([5, -12, 0, -34, 0, 0, 0])
        up = down.copy()
        up[3] = np.deg2rad(65)
        return [down, up, down.copy()]

    def trajectory(self):
        if self.command_id == 0:
            return self.target.copy(), np.zeros(len(self.names)), np.zeros(len(self.names))
        u = np.clip((self.data.time-self.move_start)/self.duration, 0, 1)
        delta = self.target-self.start
        return (self.start+(10*u**3-15*u**4+6*u**5)*delta,
                (30*u**2-60*u**3+30*u**4)*delta/self.duration,
                (60*u-180*u**2+120*u**3)*delta/self.duration**2)

    def sample(self, mode='residual'):
        qref, vref, aref = self.trajectory()
        ref = self.ref_data
        ref.qpos[:] = self.data.qpos
        ref.qvel[:] = 0
        ref.qacc[:] = 0
        ref.qpos[self.qadr], ref.qvel[self.dadr], ref.qacc[self.dadr] = qref, vref, aref
        mujoco.mj_inverse(self.model, ref)
        ff = ref.qfrc_inverse[self.dadr].copy()
        q, v = self.data.qpos[self.qadr].copy(), self.data.qvel[self.dadr].copy()
        raw = ff + self.kp*(qref-q) + self.kd*(vref-v)
        self.data.ctrl[:] = np.clip(raw, self.limits[:, 0], self.limits[:, 1])
        mujoco.mj_forward(self.model, self.data)
        torque = self.data.qfrc_actuator[self.dadr].copy()
        xyz, quat = self.ee_pose()
        self.last_inverse, self.last_vref, self.last_aref = ff, vref, aref
        self.ref_xyz = ref.site_xpos[self.sid].copy()
        mujoco.mju_mat2Quat(self.ref_quat, ref.site_xmat[self.sid])
        self.pos_error = float(np.linalg.norm(xyz-self.target_xyz))
        self.angle_error = float((Rotation.from_quat(np.roll(self.target_quat,-1)) *
                                  Rotation.from_quat(np.roll(quat,-1)).inv()).magnitude())
        if self.command_id and self.data.time >= self.move_start+self.duration:
            settled = self.pos_error < 0.003 and self.angle_error < 0.03 and np.max(np.abs(v)) < 0.02
            if settled:
                if self.settled_since is None:
                    self.settled_since = float(self.data.time)
                self.phase = 'READY' if self.data.time-self.settled_since >= 0.2 else 'SETTLING'
            else:
                self.phase, self.settled_since = 'SETTLING', None
        saturated = (np.abs(raw-torque)>1e-8).astype(float)
        row = np.concatenate(([self.data.time], q, v, raw, torque, qref, xyz, saturated))
        if not np.all(np.isfinite(row)):
            raise RuntimeError('Non-finite residual experiment state')
        return row, torque, xyz, v

    def extra_row(self):
        _, quat = self.ee_pose()
        return [self.command_id, self.phase, *self.last_inverse, *self.last_vref, *self.last_aref,
                *quat, *self.ref_xyz, *self.ref_quat, *self.target_xyz, *self.target_quat,
                self.pos_error, self.angle_error]


class SlideControls:
    """Callbacks enqueue user input; only the main loop touches MuJoCo state."""
    def __init__(self, bench, server, widgets):
        self.bench, self.server, self.widgets = bench, server, widgets
        self.lock = threading.Lock()
        self.pending = None
        self.last_solve = 0.0
        self.message = server.gui.add_markdown(
            'Drag EE arrows/planes to translate; rings to rotate. Joint sliders stay synchronized.')
        xyz, quat = self.pose()
        self.actual = server.scene.add_frame('/ee_actual', position=xyz, wxyz=quat,
                                              axes_length=0.08, axes_radius=0.003)
        # Same transform controls and live/final callbacks as the keyframe editor.
        self.gizmo = server.scene.add_transform_controls('/ee_target', position=xyz, wxyz=quat,
                                                         scale=0.18, disable_rotations=False)
        self.gizmo.on_update(lambda event: self.queue_pose(event, False))
        self.gizmo.on_drag_end(lambda event: self.queue_pose(event, True))
        for widget in widgets:
            widget.on_update(self.queue_joints)

    def pose(self):
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.bench.data.site_xmat[self.bench.sid])
        return self.bench.data.site_xpos[self.bench.sid].copy(), quat

    def queue_pose(self, event, final):
        if event.client_id is None:
            return
        with self.lock:
            self.pending = ('pose', np.asarray(event.target.position).copy(),
                            np.asarray(event.target.wxyz).copy(), final)

    def queue_joints(self, event):
        if event.client_id is None:
            return
        with self.lock:
            self.pending = ('joints', np.array([w.value for w in self.widgets]))

    def update(self):
        with self.lock:
            command = self.pending
            if command is None:
                return
            if command[0] == 'pose' and not command[3] and time.monotonic()-self.last_solve < 0.06:
                return
            self.pending = None
        if command[0] == 'pose':
            self.last_solve = time.monotonic()
            try:
                self.bench.target = self.bench.solve_slide_pose(command[1], command[2])
            except ValueError as exc:
                self.message.content = str(exc)
                return
            self.message.content = 'EE pose applied (no PD control).'
        else:
            self.bench.target = command[1]
            self.message.content = 'Joint pose applied (no PD control).'
        self.bench.sample('slide')
        xyz, quat = self.pose()
        with self.server.atomic():
            for widget, value in zip(self.widgets, self.bench.target):
                widget.value = float(value)
            self.actual.position, self.actual.wxyz = xyz, quat
            # Leave the target in the user's hand during dragging.
            if command[0] == 'joints' or command[3]:
                self.gizmo.position, self.gizmo.wxyz = xyz, quat


class WristCOMVisual:
    """Display the right wrist body's inertial COM, not the EE or subtree COM."""
    def __init__(self, bench, server):
        self.bench = bench
        body_name = getattr(bench, 'com_body', 'right_wrist_pitch_link')
        self.bid = mujoco.mj_name2id(bench.model, mujoco.mjtObj.mjOBJ_BODY, body_name) if body_name else -1
        self.handles = []
        if self.bid < 0:
            return
        mass = float(bench.model.body_mass[self.bid])
        pos = bench.data.xipos[self.bid].copy()
        self.handles = [
            server.scene.add_icosphere('/right_wrist_com/marker',radius=.012,color=(255,0,200),position=pos),
            server.scene.add_label('/right_wrist_com/label',text=f'● {body_name} COM ({mass:g} kg)',
                                   position=pos,depth_test=False,anchor='bottom-left'),
        ]
        with server.gui.add_folder('Link COM',expand_by_default=False):
            self.visible = server.gui.add_checkbox('무게중심 표시',initial_value=True)
            self.info = server.gui.add_markdown('')
        self.update()

    def update(self):
        if self.bid < 0:
            return
        b = self.bench
        pos = b.data.xipos[self.bid].copy()
        for handle in self.handles:
            handle.position = pos
            handle.visible = self.visible.value
        self.info.content = (f'질량: **{b.model.body_mass[self.bid]:g} kg**\n\n'
                             f'링크 기준 COM (m): `{np.round(b.model.body_ipos[self.bid],6)}`\n\n'
                             f'World COM (m): `{np.round(pos,6)}`')


def make_ui(bench, port, mode):
    import viser
    import trimesh
    server = viser.ViserServer(host='127.0.0.1', port=port)
    title = 'Joint posing (no control)' if mode == 'slide' else 'EE pose / payload measurement' if mode == 'residual' else 'Single-arm PD payload bench'
    title = getattr(bench, 'platform_title', title)
    server.gui.add_markdown(f'## {title}\nAngles: rad · torque: N·m · position: world m')
    widgets = []
    if mode == 'slide':
        for name, limits, initial in zip(bench.names, bench.bounds, bench.start):
            widgets.append(server.gui.add_slider(name, min=float(limits[0]), max=float(limits[1]),
                                                step=0.001, initial_value=float(initial)))
    status = server.gui.add_markdown('Starting…')
    handles = []
    m = bench.model
    for g in range(m.ngeom):
        if m.geom_rgba[g, 3] <= 0 or int(m.geom_bodyid[g]) in getattr(bench, 'hidden_bodies', set()):
            continue
        kind = m.geom_type[g]
        size = m.geom_size[g]
        mesh = None
        if kind == mujoco.mjtGeom.mjGEOM_PLANE:
            # MuJoCo planes are infinite; show an 8 m square at the geom pose.
            extent = np.where(size[:2] > 0, size[:2], 4.0)
            vertices, faces, colors = [], [], []
            for ix in range(16):
                for iy in range(16):
                    x0, x1 = -extent[0] + np.array([ix, ix+1])*extent[0]/8
                    y0, y1 = -extent[1] + np.array([iy, iy+1])*extent[1]/8
                    base = len(vertices)
                    vertices.extend([[x0,y0,0], [x1,y0,0], [x1,y1,0], [x0,y1,0]])
                    faces.extend([[base,base+1,base+2], [base,base+2,base+3]])
                    colors.extend([[51,76,102] if (ix+iy)%2 else [26,51,76]]*4)
            floor_mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces),
                                         vertex_colors=np.asarray(colors, dtype=np.uint8), process=False)
            handle = server.scene.add_mesh_trimesh(f'/robot/geom_{g}', floor_mesh)
            handles.append((g, handle))
            continue
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            mesh = trimesh.Trimesh(vertices=m.mesh_vert[va:va+vn], faces=m.mesh_face[fa:fa+fn], process=False)
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = trimesh.creation.box(extents=2*size)
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            mesh = trimesh.creation.cylinder(radius=size[0], height=2*size[1])
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            mesh = trimesh.creation.capsule(radius=size[0], height=2*size[1])
        if mesh is not None:
            handle = server.scene.add_mesh_simple(f'/robot/geom_{g}', vertices=mesh.vertices,
                                                  faces=mesh.faces, color=tuple((255*m.geom_rgba[g, :3]).astype(int)))
            handles.append((g, handle))
    bench.com_visual = WristCOMVisual(bench, server)
    if mode == 'ik':
        server.scene.add_icosphere('/target', radius=0.015, color=(255, 30, 30), position=bench.target_xyz)
    if mode == 'residual':
        from payload_session import SessionControls
        controls = SessionControls(bench, server)
    else:
        controls = SlideControls(bench, server, widgets) if mode == 'slide' else None
    return server, widgets, status, handles, controls


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('xml', type=Path, nargs='?', default=Path(__file__).resolve().parents[1]/'prototype_v2.1.1/mjcf/scene.xml')
    p.add_argument('--mode', choices=['residual', 'slide', 'ik'], default='residual')
    p.add_argument('--arm', choices=['left', 'right'], default='right')
    p.add_argument('--ee-site')
    p.add_argument('--waypoints', type=Path, help='Load A/B/C/ordinary pos file (legacy waypoints.json also supported)')
    p.add_argument('--payload-kg', type=float, default=0, help='10/15/20: select matching right wrist link total-mass XML; other positive values: legacy added EE payload')
    p.add_argument('--payload-com', type=float, nargs=3, default=(0,0,0), metavar=('X','Y','Z'), help='Payload COM offset from EE in site-local meters')
    p.add_argument('--payload-radius', type=float, default=0.04, help='Sphere inertia radius, meters')
    p.add_argument('--target', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    p.add_argument('--seconds', type=float)
    p.add_argument('--kp', type=float, default=200)
    p.add_argument('--kd', type=float, default=40, help='Velocity damping gain (default 40; previous default 20)')
    p.add_argument('--gravity-comp', action='store_true')
    p.add_argument('--dt', type=float, default=0.002)
    p.add_argument('--print-hz', type=float, default=5)
    p.add_argument('--port', type=int, default=8766)
    p.add_argument('--output', type=Path, default=Path('payload_logs'))
    p.add_argument('--headless', action='store_true')
    p.add_argument('--run-seconds', type=float, help='Optional automatic stop; otherwise Ctrl+C.')
    return p


def main():
    p = parser()
    args = p.parse_args()
    for key in ('dt', 'print_hz', 'seconds', 'run_seconds', 'payload_radius'):
        value = getattr(args, key)
        if value is not None and (not np.isfinite(value) or value <= 0):
            p.error(f'--{key.replace("_", "-")} must be finite and positive')
    for key, maximum in (('kp', 1000), ('kd', 200)):
        value = getattr(args, key)
        if not np.isfinite(value) or not 0 <= value <= maximum:
            p.error(f'--{key} must be between 0 and {maximum}')
    if not np.isfinite(args.payload_kg) or args.payload_kg < 0:
        p.error('--payload-kg must be finite and nonnegative')
    if not np.all(np.isfinite(args.payload_com)):
        p.error('--payload-com must be finite')
    if args.waypoints is not None and (args.mode != 'residual' or args.headless):
        p.error('--waypoints requires residual UI mode')
    if args.mode == 'ik' and (args.target is None or args.seconds is None):
        p.error('IK requires --target X Y Z and --seconds T')
    if args.target is not None and not np.all(np.isfinite(args.target)):
        p.error('--target must be finite')
    args.requested_xml = args.xml
    try:
        args.xml, args.added_payload_kg, args.payload_definition = select_payload_model(
            args.xml, args.arm, args.payload_kg, args.payload_com)
    except ValueError as exc:
        p.error(str(exc))
    print(f'Model: {args.xml}\nMass configuration: {args.payload_definition}', flush=True)
    m, d, names, site, limits = build_model(args.xml, args.arm, args.ee_site, args.added_payload_kg, args.dt, args.payload_com, args.payload_radius)
    bench_type = ResidualBench if args.mode == 'residual' else Bench
    bench = bench_type(m, d, names, site, limits, args.kp, args.kd, args.gravity_comp)
    if args.mode == 'residual':
        from payload_session import run_session
        return run_session(args, bench, make_ui)
    if args.mode == 'ik':
        bench.set_ik(args.target, args.seconds)
    args.output.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    output = args.output / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir()
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(joint_names=names, ee_site=site, torque_limits_Nm=limits.tolist(),
                    initial_ee_world_m=d.site_xpos[bench.sid].tolist(),
                    target_joint_rad=bench.target.tolist(), mujoco_version=mujoco.__version__,
                    fixed_pose='XML reference pose; only selected arm joints retained',
                    contacts_enabled=False,
                    control_mode='kinematic_pose' if args.mode == 'slide' else 'inverse_dynamics_plus_PD' if args.mode == 'residual' else 'PD',
                    slide_log_note='Slide torque/velocity are zero placeholders; no physical motion is simulated.',
                    torque_definition='qfrc_actuator, joint-side applied torque after saturation',
                    payload_definition=args.payload_definition,
                    residual_rotation='Rotation vector in degrees; world left-multiply, EE local right-multiply',
                    trajectory='Quintic joint-space interpolation between full-pose IK endpoints')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    columns = ['time_s']
    for field in ('position_rad', 'velocity_rad_s', 'torque_requested_Nm', 'torque_Nm', 'target_rad'):
        columns += [f'{name}_{field}' for name in names]
    columns += ['ee_world_x_m', 'ee_world_y_m', 'ee_world_z_m']
    columns += [f'{name}_saturated' for name in names]
    if args.mode == 'residual':
        columns += ['command_id', 'phase']
        for field in ('inverse_torque_Nm', 'reference_velocity_rad_s', 'reference_acceleration_rad_s2'):
            columns += [f'{name}_{field}' for name in names]
        columns += ['ee_w', 'ee_qx', 'ee_qy', 'ee_qz']
        columns += ['reference_ee_'+x for x in ('x_m','y_m','z_m','w','qx','qy','qz')]
        columns += ['goal_ee_'+x for x in ('x_m','y_m','z_m','w','qx','qy','qz')]
        columns += ['goal_position_error_m', 'goal_orientation_error_rad']
    server = None
    count = 0
    commands_written = 0
    print(f'EE {site}: {d.site_xpos[bench.sid]} | output: {output.resolve()}', flush=True)
    try:
        if not args.headless:
            server, widgets, status, handles, controls = make_ui(bench, args.port, args.mode)
            if args.mode == 'residual' and args.waypoints is not None:
                controls.load_waypoints(args.waypoints)
        with (output / 'samples.csv').open('w', newline='') as stream, (output / 'commands.jsonl').open('w') as command_stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            wall_start = time.monotonic()
            next_print = next_ui = 0.0
            while args.run_seconds is None or d.time < args.run_seconds:
                if server is not None and controls is not None:
                    controls.update()
                row, torque, xyz, velocity = bench.sample(args.mode)
                if args.mode == 'residual':
                    while commands_written < len(bench.records):
                        command_stream.write(json.dumps(bench.records[commands_written])+'\n')
                        command_stream.flush()
                        commands_written += 1
                    writer.writerow([*row, *bench.extra_row()])
                else:
                    writer.writerow(row)
                count += 1
                if d.time >= next_print:
                    phase = 'HOLD' if bench.seconds is not None and d.time >= bench.seconds else args.mode.upper()
                    if args.mode == 'residual':
                        phase = f'#{bench.command_id} {bench.phase}'
                    print(f't={d.time:.3f}s {phase} EE={np.round(xyz, 5)}', flush=True)
                    if args.mode == 'slide':
                        for n, q in zip(names, d.qpos[bench.qadr]):
                            print(f'  {n}: position={q:+.4f} rad (no control)', flush=True)
                    else:
                        for n, t, v in zip(names, torque, velocity):
                            print(f'  {n}: torque={t:+.4f} Nm  velocity={v:+.4f} rad/s', flush=True)
                    if args.mode in ('ik', 'residual'):
                        print(f'  EE error={np.linalg.norm(xyz-bench.target_xyz):.5f} m', flush=True)
                    stream.flush()
                    next_print = d.time + 1/args.print_hz
                if server is not None and d.time >= next_ui:
                    with server.atomic():
                        for g, handle in handles:
                            quat = np.empty(4)
                            mujoco.mju_mat2Quat(quat, d.geom_xmat[g])
                            handle.position = d.geom_xpos[g].copy()
                            handle.wxyz = quat
                        bench.com_visual.update()
                        if args.mode == 'residual':
                            controls.refresh()
                        status.content = f'**t={d.time:.2f}s** · EE world: `{np.round(xyz, 4)}`\n\nCtrl+C in terminal to stop and save.'
                    next_ui = d.time + 1/30
                previous_time = d.time
                bench.advance(args.mode)
                if d.time <= previous_time or not np.all(np.isfinite(d.qpos)):
                    raise RuntimeError('Simulation diverged or reset; reduce gains/timestep.')
                delay = wall_start + d.time-time.monotonic()
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        print('\nCtrl+C: stopping and saving.', flush=True)
    finally:
        metadata.update(samples=count, final_sim_time_s=float(d.time), final_kd=float(bench.kd), gain_changes=bench.gain_changes)
        if args.mode == 'residual':
            metadata.update(command_count=bench.command_id, final_phase=bench.phase)
            if server is not None:
                (output / 'waypoints.json').write_text(json.dumps(dict(joint_names=bench.names,points=controls.waypoint_records()),indent=2)+'\n')
            # Also preserve a command accepted immediately before Ctrl+C.
            (output / 'commands.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in bench.records))
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        if server is not None:
            server.stop()
        print(f'Saved {count} samples: {output.resolve() / "samples.csv"}', flush=True)


if __name__ == '__main__':
    main()
