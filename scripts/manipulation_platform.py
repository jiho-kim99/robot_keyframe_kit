#!/usr/bin/env python3
"""Configurable fixed-base manipulation measurements: robot × domain × model variant."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import yaml
from arm_payload import ResidualBench, make_ui
from payload_session import run_session

ROOT = Path(__file__).resolve().parents[1]


def load_catalog(path):
    path=Path(path).expanduser().resolve()
    config=yaml.safe_load(path.read_text())
    if not isinstance(config,dict) or config.get('schema_version')!=1:
        raise ValueError('Platform config requires schema_version: 1')
    for collection in ('robots','domains'):
        if not isinstance(config.get(collection),dict) or not config[collection]:
            raise ValueError(f'Config requires nonempty {collection}')
        for name in config[collection]:
            if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',name):
                raise ValueError(f'Invalid {collection} ID: {name}')
    return config,path.parent


def build_experiment(catalog, base, selection, groups=None, kp=200, kd=40, dt=.002):
    robot_id,domain_id,variant,endpoint=selection
    robot=catalog['robots'][robot_id]
    domain=catalog['domains'][domain_id]
    if domain.get('kind') not in ('payload_lift','waypoint_motion'):
        raise ValueError('Unsupported domain kind; implement its physics before enabling it')
    variants=robot['models']
    if variant not in variants:raise ValueError(f'Unknown model variant: {variant}')
    if variant not in domain.get('variants',list(variants)):
        raise ValueError(f'{domain_id} does not support variant {variant}')
    model_file=domain.get('model_overrides',{}).get(robot_id,{}).get(variant,variants[variant])
    model_path=(base/model_file).resolve()
    chosen_groups=list(groups or robot['default_groups'])
    names=[]
    for group in chosen_groups:
        if group not in robot['joint_groups']:raise ValueError(f'Unknown joint group: {group}')
        names.extend(robot['joint_groups'][group])
    if not names or len(names)!=len(set(names)):raise ValueError('Active joints must be nonempty and unique')
    forbidden=set(robot.get('excluded_joints',[]))
    if forbidden.intersection(names):raise ValueError('Excluded joints cannot be activated')
    ee=robot['end_effectors'][endpoint]
    spec=mujoco.MjSpec.from_file(str(model_path))
    if list(spec.equalities) or list(spec.tendons):
        raise ValueError('Independent joints required; coupled transmissions are not supported yet')
    for configured in robot['end_effectors'].values():
        if spec.site(configured['site']) is None:
            if 'body' not in configured:raise ValueError(f"Missing EE site {configured['site']}; configure body and position_m")
            pos=np.asarray(configured.get('position_m',[0,0,0]),dtype=float)
            if pos.shape!=(3,) or not np.all(np.isfinite(pos)):raise ValueError('Invalid EE position_m')
            spec.body(configured['body']).add_site(name=configured['site'],pos=pos,size=[.01]*3)
    original=spec.compile()
    limits=[]
    for name in names:
        j=original.joint(name)
        if original.jnt_type[j.id]!=mujoco.mjtJoint.mjJNT_HINGE or not original.jnt_limited[j.id]:
            raise ValueError(f'{name}: a limited revolute joint is required')
        value=robot.get('torque_limits_Nm',{}).get(name)
        if value is None:
            if not original.jnt_actfrclimited[j.id]:raise ValueError(f'{name}: specify torque_limits_Nm in the robot profile')
            value=original.jnt_actfrcrange[j.id]
        value=np.asarray(value,dtype=float)
        if value.shape!=(2,) or not np.all(np.isfinite(value)) or value[0]>=0 or value[1]<=0:
            raise ValueError(f'{name}: torque limits must be [negative, positive]')
        limits.append(value)
    # Validate the EE belongs to a chain containing an active joint.
    body=int(original.site_bodyid[original.site(ee['site']).id]);ancestors=set()
    while body:
        ancestors.add(body);body=int(original.body_parentid[body])
    if not any(int(original.jnt_bodyid[original.joint(n).id]) in ancestors for n in names):
        raise ValueError('Selected EE cannot be moved by the active joint groups')
    for collection in (spec.actuators,spec.sensors,spec.keys):
        for item in list(collection):spec.delete(item)
    # Bake configured inactive hinge poses into local body transforms before
    # deleting their joints. Geometry AND mass/inertia follow the closed hand.
    fixed_positions=robot.get('fixed_joint_positions_rad',{})
    if fixed_positions:
        frozen=mujoco.MjData(original)
        bodies=set()
        for name,value in fixed_positions.items():
            if name in names:raise ValueError(f'Fixed joint cannot also be active: {name}')
            j=original.joint(name)
            if original.jnt_type[j.id]!=mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f'Fixed pose requires hinge joint: {name}')
            if not np.isfinite(value) or not j.range[0]<=value<=j.range[1]:
                raise ValueError(f'Invalid fixed joint position: {name}')
            bid=int(original.jnt_bodyid[j.id])
            if original.body_jntnum[bid]!=1:
                raise ValueError(f'Fixed pose requires one joint per body: {name}')
            frozen.qpos[j.qposadr[0]]=value
            bodies.add(bid)
        mujoco.mj_forward(original,frozen)
        for bid in bodies:
            parent=int(original.body_parentid[bid])
            rotation=frozen.xmat[parent].reshape(3,3).T
            body=spec.body(original.body(bid).name)
            body.pos=rotation@(frozen.xpos[bid]-frozen.xpos[parent])
            quat=np.empty(4)
            mujoco.mju_mat2Quat(quat,(rotation@frozen.xmat[bid].reshape(3,3)).ravel())
            body.quat=quat
    for joint in list(spec.joints):
        if joint.name not in names:spec.delete(joint)
    for name,limit in zip(names,limits):
        spec.add_actuator(name='measurement_'+name,target=name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,gaintype=mujoco.mjtGain.mjGAIN_FIXED,gainprm=[1]+[0]*9,
            biastype=mujoco.mjtBias.mjBIAS_NONE,dyntype=mujoco.mjtDyn.mjDYN_NONE,
            gear=[1,0,0,0,0,0],ctrllimited=True,ctrlrange=limit,forcelimited=True,forcerange=limit)
    model=spec.compile()
    model.opt.timestep=dt
    model.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # These initial domains measure free-space fixed-base manipulation.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    data=mujoco.MjData(model)
    for name,value in robot.get('initial_positions_rad',{}).items():
        if name not in names:raise ValueError(f'Initial position must name an active joint: {name}')
        j=model.joint(name)
        if not np.isfinite(value) or not j.range[0]<=value<=j.range[1]:raise ValueError(f'Invalid initial position: {name}')
        data.qpos[j.qposadr[0]]=value
    mujoco.mj_forward(model,data)
    bench=PlatformBench(model,data,names,ee['site'],np.asarray(limits),kp,kd)
    bench.presets=robot.get('presets',{}) if domain['kind']=='payload_lift' else {}
    bench.com_body=ee.get('com_body')
    bench.hidden_bodies=set()
    for root in robot.get('hidden_body_roots',[]):
        bid=model.body(root).id
        for i in range(1,model.nbody):
            parent=i
            while parent:
                if parent==bid:bench.hidden_bodies.add(i);break
                parent=int(model.body_parentid[parent])
    snapshot=dict(robot=robot,domain=domain,selection=list(selection),groups=chosen_groups)
    fingerprint=hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()+model_path.read_bytes()+model.body_mass.tobytes()+model.body_ipos.tobytes()+model.body_inertia.tobytes()).hexdigest()[:12]
    bench.pose_context=dict(robot=robot_id,domain=domain_id,variant=variant,endpoint=endpoint,
                            joints=names,profile_hash=fingerprint)
    bench.platform_metadata=dict(robot_profile=robot_id,domain=domain_id,domain_kind=domain['kind'],
        model_variant=variant,xml=str(model_path),end_effector=endpoint,ee_site=ee['site'],
        motor_ratings=str((base/robot['motor_ratings']).resolve()) if robot.get('motor_ratings') else None,
        active_groups=chosen_groups,fixed_base=True,contacts_enabled=False,profile_snapshot=snapshot,
        pose_context=bench.pose_context,measurement_scope='configured active revolute joints only')
    bench.platform_title=f'{robot_id} · {domain_id} · {variant} · {endpoint}'
    return bench


class PlatformBench(ResidualBench):
    def curl_presets(self):
        if not self.presets:raise ValueError('No presets for this robot/domain')
        result=[]
        for slot in ('A','B','C'):
            q=self.data.qpos[self.qadr].copy()
            for name,degrees in self.presets[slot].items():
                if name in self.names:q[self.names.index(name)]=np.deg2rad(degrees)
            if np.any(q<self.bounds[:,0]) or np.any(q>self.bounds[:,1]):raise ValueError('Preset exceeds joint limits')
            result.append(q)
        return result


class PlatformSelector:
    def __init__(self,bench,server,catalog,selection,prepare):
        self.bench,self.prepare=bench,prepare
        self.pending=None
        with server.gui.add_folder('Robot / domain'):
            self.robot=server.gui.add_dropdown('Robot',options=tuple(catalog['robots']),initial_value=selection[0])
            self.domain=server.gui.add_dropdown('Domain',options=tuple(catalog['domains']),initial_value=selection[1])
            self.variant=server.gui.add_dropdown('Model variant',options=tuple(catalog['robots'][selection[0]]['models']),initial_value=selection[2])
            self.endpoint=server.gui.add_dropdown('End effector',options=tuple(catalog['robots'][selection[0]]['end_effectors']),initial_value=selection[3])
            self.robot.on_update(lambda _:self.robot_changed(catalog))
            self.apply=server.gui.add_button('Apply robot / domain (reset pose)')
            self.apply.on_click(lambda _:self.request())
            self.message=server.gui.add_markdown('Save poses before switching. Switching is disabled during recording.')
            server.gui.add_markdown('Active joints: '+', '.join(bench.names)+'\n\nFixed base; contacts off. Legs are excluded.')
    def robot_changed(self,catalog):
        profile=catalog['robots'][self.robot.value]
        self.variant.options=tuple(profile['models']);self.variant.value=next(iter(profile['models']))
        self.endpoint.options=tuple(profile['end_effectors']);self.endpoint.value=next(iter(profile['end_effectors']))
    def request(self):
        if not self.bench.session.active:
            self.pending=(self.robot.value,self.domain.value,self.variant.value,self.endpoint.value)
    def poll(self):
        active=self.bench.session.active
        for h in (self.robot,self.domain,self.variant,self.endpoint,self.apply):h.disabled=active
        pending,self.pending=self.pending,None
        if pending is None or active:return None
        try:
            return pending,self.prepare(pending)
        except (ValueError,KeyError,OSError) as exc:
            self.message.content=f'Cannot switch: {exc}'
            return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/manipulation_platform.yaml')
    p.add_argument('--robot',default='prototype_barrett')
    p.add_argument('--domain',default='payload_lift')
    p.add_argument('--variant',default='base')
    p.add_argument('--ee',default='right')
    p.add_argument('--groups',nargs='+',help='Override active profile groups (e.g. waist right_arm)')
    p.add_argument('--kp',type=float,default=200)
    p.add_argument('--kd',type=float,default=40)
    p.add_argument('--dt',type=float,default=.002)
    p.add_argument('--port',type=int,default=8766)
    p.add_argument('--output',type=Path,default=ROOT/'manipulation_logs')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--run-seconds',type=float)
    p.add_argument('--check',action='store_true',help='Validate and print model scope without running UI')
    p.add_argument('--list',action='store_true')
    args=p.parse_args()
    try:
        catalog,base=load_catalog(args.config)
        if args.list:
            print(yaml.safe_dump(catalog,sort_keys=False));return
        if not np.isfinite(args.kp) or not 0<=args.kp<=1000 or not np.isfinite(args.kd) or not 0<=args.kd<=200:
            raise ValueError('Kp range: 0..1000; Kd: 0..200')
        if not np.isfinite(args.dt) or args.dt<=0:raise ValueError('dt must be positive')
        if args.run_seconds is not None and (not np.isfinite(args.run_seconds) or args.run_seconds<=0):raise ValueError('run-seconds must be positive')
        selection=(args.robot,args.domain,args.variant,args.ee)
        def prepare(value):return build_experiment(catalog,base,value,args.groups,args.kp,args.kd,args.dt)
        bench=prepare(selection)
        if args.check:
            print(json.dumps(bench.platform_metadata,indent=2));return
        while True:
            print(bench.platform_title+'\nJoints: '+', '.join(bench.names),flush=True)
            run=copy.copy(args)
            context=bench.pose_context
            run.output=args.output/selection[0]/selection[1]/selection[2]/selection[3]/context['profile_hash']
            run.waypoints=None;run.print_hz=5
            for key,value in bench.platform_metadata.items():setattr(run,key,value)
            def ui(b,port,mode):
                result=make_ui(b,port,mode)
                selector=PlatformSelector(b,result[0],catalog,selection,prepare)
                b.poll_switch=selector.poll
                return result
            change=run_session(run,bench,ui)
            if change is None:break
            selection,bench=change
    except (ValueError,KeyError,OSError) as exc:p.error(str(exc))

if __name__=='__main__':main()
