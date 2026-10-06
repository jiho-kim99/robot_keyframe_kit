"""Pose editing and explicitly bounded payload measurements for arm_payload.py."""
from __future__ import annotations

import csv
import json
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

NAMES = ('ordinary pos', 'A', 'B', 'C')
SKIP = '(skip)'


def columns_for(bench):
    columns = ['time_s']
    for field in ('position_rad','velocity_rad_s','torque_requested_Nm','torque_Nm','target_rad'):
        columns += [f'{j}_{field}' for j in bench.names]
    columns += ['ee_world_x_m','ee_world_y_m','ee_world_z_m']
    columns += [f'{j}_saturated' for j in bench.names]
    columns += ['command_id','phase']
    for field in ('inverse_torque_Nm','reference_velocity_rad_s','reference_acceleration_rad_s2'):
        columns += [f'{j}_{field}' for j in bench.names]
    columns += ['ee_w','ee_qx','ee_qy','ee_qz']
    columns += ['reference_ee_'+x for x in ('x_m','y_m','z_m','w','qx','qy','qz')]
    columns += ['goal_ee_'+x for x in ('x_m','y_m','z_m','w','qx','qy','qz')]
    return columns+['goal_position_error_m','goal_orientation_error_rad']


class ExperimentSession:
    def __init__(self, bench, output, metadata):
        self.bench = bench
        self.output = Path(output)
        self.metadata = metadata
        self.mode = 'edit'
        self.state = 'EDIT'
        self.active = False
        self.poses = {name: None for name in NAMES}
        self.poses['ordinary pos'] = self.pose_record()
        self.file = None
        self.path = None
        self.count = 0
        self.plan = []
        self.index = 0
        self.hold_until = None
        self.elapsed = 0.0
        self.saved_path = self.output/'ee_poses.json'

    def validate_joints(self, joints):
        q = np.asarray(joints,dtype=float)
        b = self.bench
        if q.shape != b.target.shape or not np.all(np.isfinite(q)):
            raise ValueError('잘못된 관절 자세입니다.')
        if np.any(q<b.bounds[:,0]) or np.any(q>b.bounds[:,1]):
            raise ValueError('관절 범위를 벗어난 자세입니다.')
        return q.copy()

    def pose_record(self):
        xyz, quat = self.bench.ee_pose()
        return dict(joints=self.bench.data.qpos[self.bench.qadr].tolist(),
                    position_m=xyz.tolist(), wxyz=quat.tolist())

    def set_pose(self, joints):
        if self.active or self.mode != 'edit':
            raise ValueError('자세 변경은 편집 모드에서만 가능합니다.')
        b = self.bench
        q = self.validate_joints(joints)
        b.data.qpos[b.qadr] = q
        b.data.qvel[:] = 0
        b.data.ctrl[:] = 0
        b.data.qfrc_applied[:] = 0
        b.data.xfrc_applied[:] = 0
        mujoco.mj_forward(b.model,b.data)
        b.start = b.target = q.copy()
        b.target_xyz,b.target_quat = b.ee_pose()
        b.command_id,b.phase = 0,'READY'
        b.sequence.clear()

    def edit_frame(self, xyz, quat):
        if self.mode != 'edit' or self.active:
            raise ValueError('EE frame 편집 모드에서 조작하세요.')
        self.set_pose(self.bench.solve_slide_pose(xyz,quat))

    def save_pose(self, name):
        if self.mode != 'edit' or self.active or name not in NAMES:
            raise ValueError('편집 모드에서 저장할 점을 선택하세요.')
        self.poses[name] = self.pose_record()
        self.persist_poses()

    def persist_poses(self):
        self.output.mkdir(parents=True,exist_ok=True)
        tmp = self.saved_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(joint_names=self.bench.names,poses=self.poses,pose_context=getattr(self.bench,'pose_context',None)),indent=2)+'\n')
        tmp.replace(self.saved_path)

    def set_gains(self, kp, kd):
        if self.active:
            raise ValueError('측정 종료 후 gain을 변경하세요.')
        if not np.all(np.isfinite([kp,kd])) or not (0 <= kp <= 1000 and 0 <= kd <= 200):
            raise ValueError('Kp는 0~1000, Kd는 0~200이어야 합니다.')
        self.bench.kp,self.bench.kd=float(kp),float(kd)

    def load_poses(self, path):
        if self.active or self.mode != 'edit':
            raise ValueError('EE frame 파일은 편집 모드에서 불러오세요.')
        saved = json.loads(Path(path).read_text())
        if getattr(self.bench,'pose_context',None) is not None and saved.get('pose_context') != self.bench.pose_context:
            raise ValueError('Saved poses belong to a different robot/domain/variant/joint configuration.')
        if saved.get('joint_names') != self.bench.names:
            raise ValueError('저장 파일의 관절 이름이 현재 팔과 다릅니다.')
        loaded = dict(self.poses)
        entries = saved.get('poses')
        if entries is None:  # Previous A/B/C waypoint files.
            entries = {label: item for label,item in zip(('A','B','C'),saved.get('points',[]))}
        scratch = mujoco.MjData(self.bench.model)
        for name,value in entries.items():
            if name not in NAMES or value is None:
                continue
            q = self.validate_joints(value['joints'])
            scratch.qpos[:] = self.bench.data.qpos
            scratch.qpos[self.bench.qadr] = q
            mujoco.mj_forward(self.bench.model,scratch)
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat,scratch.site_xmat[self.bench.sid])
            loaded[name] = dict(joints=q.tolist(),position_m=scratch.site_xpos[self.bench.sid].tolist(),wxyz=quat.tolist())
        self.poses = loaded

    def switch_mode(self, mode):
        if self.active:
            raise ValueError('측정 중에는 모드를 바꿀 수 없습니다. Stop으로 먼저 종료하세요.')
        if mode not in ('edit','measure'):
            raise ValueError('Unknown mode')
        self.mode = mode
        self.state = 'EDIT' if mode=='edit' else 'IDLE'

    def start(self, rows):
        if self.mode != 'measure' or self.active:
            raise ValueError('측정 모드에서 Start를 누르세요.')
        prepared=[]
        for row in rows:
            if row['name']==SKIP:
                continue
            name=row['name']
            if name not in self.poses or self.poses[name] is None:
                raise ValueError(f'{name} 자세를 먼저 저장하세요.')
            seconds,hold=float(row['seconds']),float(row['hold'])
            if not np.all(np.isfinite([seconds,hold])) or seconds<.1 or hold<0:
                raise ValueError('이동 시간은 0.1초 이상, 유지 시간은 0초 이상이어야 합니다.')
            prepared.append(dict(name=name,seconds=seconds,hold=hold,
                                 joints=self.validate_joints(self.poses[name]['joints']).tolist()))
        if not prepared:
            raise ValueError('실행할 점을 하나 이상 선택하세요.')
        prepared[-1]['hold']=0.0  # Stop recording upon final arrival, never append a final dwell.
        b=self.bench
        b.data.qvel[:]=0
        b.command_id=0
        b.records.clear()
        b.sequence.clear()
        b.phase='READY'
        self.start_time=float(b.data.time)
        self.elapsed=0
        self.plan,self.index,self.hold_until=prepared,0,None
        b.begin_joint_move(prepared[0]['joints'],prepared[0]['seconds'],prepared[0]['name'])
        self.path=self.output/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.path.mkdir(parents=True)
        self.file=(self.path/'samples.csv').open('w',newline='')
        self.writer=csv.writer(self.file)
        self.writer.writerow(columns_for(b))
        self.count=0
        self.active=True
        self.state='MOVING'
        self.write_metadata('recording')
        (self.path/'poses.json').write_text(json.dumps(dict(joint_names=b.names,poses=self.poses,pose_context=getattr(b,'pose_context',None)),indent=2)+'\n')
        self.write_commands()
        print(f'START recording: {self.path.resolve()}',flush=True)

    def write_commands(self):
        records=[]
        for i,source in enumerate(self.bench.records):
            r=dict(source)
            r['start_time_s']-=self.start_time
            r['hold_seconds']=self.plan[i]['hold']
            records.append(r)
        (self.path/'commands.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))

    def write_metadata(self,status):
        value=dict(self.metadata)
        value.update(mode='residual',workflow='edit_then_measure',joint_names=self.bench.names,
                     torque_limits_Nm=self.bench.limits.tolist(),samples=self.count,run_status=status,
                     command_count=self.bench.command_id,final_phase=self.state,final_sim_time_s=self.elapsed,
                     measurement_start_sim_time_s=self.start_time,plan=self.plan,
                     contacts_enabled=False,kp=self.bench.kp,kd=self.bench.kd,
                     time_origin='Start button; time_s=0 at first recorded sample',
                     arrival='3mm position, 0.03rad orientation, 0.02rad/s velocity sustained for 0.2s',
                     final_point_hold_s=0)
        (self.path/'metadata.json').write_text(json.dumps(value,indent=2)+'\n')

    def finish(self,status='completed'):
        if not self.active:
            return
        self.state='COMPLETED' if status=='completed' else 'ABORTED'
        self.active=False
        self.file.flush()
        self.file.close()
        self.file=None
        self.write_commands()
        self.write_metadata(status)
        print(f'{self.state}: {self.count} samples saved to {self.path.resolve()}',flush=True)

    def tick(self):
        b=self.bench
        if not self.active:
            b.data.time+=b.model.opt.timestep  # UI clock only: no control or logging outside a run.
            return None
        row,torque,xyz,velocity=b.sample()
        final=False
        next_leg=False
        if b.phase=='READY':
            if self.index==len(self.plan)-1:
                final=True
                self.state='COMPLETED'
            else:
                if self.hold_until is None:
                    self.hold_until=float(b.data.time)+self.plan[self.index]['hold']
                self.state='HOLDING'
                next_leg=b.data.time>=self.hold_until
        else:
            self.state=b.phase
            self.hold_until=None
        row[0]-=self.start_time
        self.elapsed=float(row[0])
        extra=b.extra_row()
        extra[1]=self.state
        self.writer.writerow([*row,*extra])
        self.count+=1
        if self.count%100==0:
            self.file.flush()
        if final:
            self.finish()
            return torque,xyz,velocity
        if next_leg:
            self.index+=1
            point=self.plan[self.index]
            b.begin_joint_move(point['joints'],point['seconds'],point['name'])
            self.hold_until=None
            self.write_commands()
        previous=b.data.time
        mujoco.mj_step(b.model,b.data)
        if b.data.time<=previous or not np.all(np.isfinite(b.data.qpos)):
            self.finish('simulation_error')
            raise RuntimeError('Simulation diverged; partial recording saved.')
        return torque,xyz,velocity


class SessionControls:
    def __init__(self,bench,server):
        self.bench,self.server,self.session=bench,server,bench.session
        self.actions=deque()
        self.lock=threading.Lock()
        self.last_edit=0.0
        self.edit_button=server.gui.add_button('① EE frame 편집 모드')
        self.measure_button=server.gui.add_button('② torque / velocity 측정 모드')
        self.edit_button.on_click(lambda _:self.queue('mode',value='edit'))
        self.measure_button.on_click(lambda _:self.queue('mode',value='measure'))
        self.message=server.gui.add_markdown('프레임을 직접 움직여 자세를 만들고 저장하세요. 편집 중에는 데이터를 기록하지 않습니다.')
        with server.gui.add_folder('EE frame 편집') as self.edit_folder:
            self.slot=server.gui.add_dropdown('저장할 이름',options=NAMES,initial_value='A')
            self.save=server.gui.add_button('현재 EE frame 저장')
            self.save.on_click(lambda _:self.queue('save',name=self.slot.value))
            self.load=server.gui.add_button('선택한 EE frame 적용')
            self.load.on_click(lambda _:self.queue('load',name=self.slot.value))
            self.pose_path=server.gui.add_text('EE frame JSON 경로',initial_value=str(self.session.saved_path.resolve()))
            self.reload=server.gui.add_button('파일에서 EE frames 불러오기')
            self.reload.on_click(lambda _:self.queue('reload',path=self.pose_path.value))
            server.gui.add_markdown('ee_poses.json 또는 실험 폴더의 poses.json을 불러온 뒤 이름을 선택하고 적용하세요.')
            self.preset=server.gui.add_button('Load robot presets into A/B/C',disabled=not bool(getattr(bench,'presets',bench.names[0].startswith('right_'))))
            self.preset.on_click(lambda _:self.queue('preset'))
            self.saved=server.gui.add_markdown('')
            with server.gui.add_folder('관절 각도로 자세 만들기',expand_by_default=False):
                self.joints=[server.gui.add_slider(name,min=float(lo),max=float(hi),step=.1,initial_value=float(q))
                             for name,(lo,hi),q in zip(bench.names,np.rad2deg(bench.bounds),np.rad2deg(bench.target))]
                for w in self.joints:
                    w.on_update(self.joint_event)
                server.gui.add_markdown('관절 각도 단위: deg')
        with server.gui.add_folder('측정 경로와 시간') as self.measure_folder:
            server.gui.add_markdown('위에서 아래 순서로 실행합니다. 이동 시간은 이전 점 → 이 점입니다. 마지막 점 유지 시간은 0으로 처리합니다.')
            self.rows=[]
            for i,name in enumerate(NAMES):
                with server.gui.add_folder(f'{i+1}번째 점'):
                    select=server.gui.add_dropdown('점',options=(*NAMES,SKIP),initial_value=name)
                    seconds=server.gui.add_number('이 점까지 이동 (s)',initial_value=2.,min=.1,step=.1)
                    hold=server.gui.add_number('도착 후 유지 (s)',initial_value=0.,min=0.,step=.1)
                    self.rows.append((select,seconds,hold))
            self.kp=server.gui.add_slider('P gain (Kp)',initial_value=float(bench.kp),min=0.,max=1000.,step=1.)
            self.kd=server.gui.add_slider('D gain (Kd)',initial_value=float(bench.kd),min=0.,max=200.,step=1.)
            server.gui.add_markdown('Kp: 0~1000 / Kd: 0~200, Start 시 적용. 측정 중에는 변경할 수 없습니다.')
            self.start=server.gui.add_button('Start · 이동 및 기록 시작')
            self.start.on_click(lambda _:self.queue('start',rows=[dict(name=n.value,seconds=t.value,hold=h.value) for n,t,h in self.rows],kp=float(self.kp.value),kd=float(self.kd.value)))
            self.stop=server.gui.add_button('Stop · 중단 기록 저장')
            self.stop.on_click(lambda _:self.queue('stop'))
        xyz,quat=bench.ee_pose()
        self.actual=server.scene.add_frame('/ee_actual',position=xyz,wxyz=quat,axes_length=.08,axes_radius=.003)
        self.gizmo=server.scene.add_transform_controls('/ee_edit',position=xyz,wxyz=quat,scale=.18,disable_rotations=False)
        self.gizmo.on_update(self.frame_event)
        self.gizmo.on_drag_end(lambda e:self.frame_event(e,True))
        self.refresh()

    def queue(self,kind,**kwargs):
        with self.lock:
            item=dict(kind=kind,**kwargs)
            if kind in ('frame','joints') and self.actions and self.actions[-1]['kind']==kind:
                self.actions[-1]=item
            else:
                self.actions.append(item)

    def frame_event(self,event,final=False):
        if event.client_id is not None and self.session.mode=='edit':
            self.queue('frame',xyz=np.array(event.target.position),quat=np.array(event.target.wxyz),final=final)

    def joint_event(self,event):
        if event.client_id is not None and self.session.mode=='edit':
            self.queue('joints',joints=np.deg2rad([w.value for w in self.joints]))

    def sync_pose(self,frame=True):
        xyz,quat=self.bench.ee_pose()
        self.actual.position,self.actual.wxyz=xyz,quat
        if frame:self.gizmo.position,self.gizmo.wxyz=xyz,quat
        for w,q in zip(self.joints,self.bench.data.qpos[self.bench.qadr]):w.value=float(np.rad2deg(q))

    def update(self):
        with self.lock:
            if not self.actions:return
            action=self.actions[0]
            if action['kind']=='frame' and not action['final'] and time.monotonic()-self.last_edit<.06:return
            self.actions.popleft()
        s=self.session
        try:
            kind=action['kind']
            if kind=='mode':
                s.switch_mode(action['value']);self.sync_pose()
                self.message.content='편집 중 · 기록 안 함' if s.mode=='edit' else '측정 대기 · Start를 누르면 기록합니다.'
            elif kind=='frame':
                self.last_edit=time.monotonic()
                s.edit_frame(action['xyz'],action['quat'])
                self.sync_pose(frame=action['final'])
                self.message.content='프레임 자세 적용 · 원하는 이름으로 저장하세요.'
            elif kind=='joints':s.set_pose(action['joints']);self.sync_pose()
            elif kind=='save':
                s.save_pose(action['name']);self.message.content=f'{action["name"]} 저장 완료 · {s.saved_path}'
            elif kind=='reload':
                s.load_poses(Path(action['path']).expanduser())
                self.message.content='EE frames 불러오기 완료 · 이름 선택 후 선택한 EE frame 적용을 누르세요.'
            elif kind=='load':
                value=s.poses[action['name']]
                if value is None:raise ValueError('아직 저장하지 않은 점입니다.')
                s.set_pose(value['joints']);self.sync_pose()
            elif kind=='preset':
                if s.mode!='edit':raise ValueError('편집 모드에서 사용하세요.')
                original=self.bench.data.qpos[self.bench.qadr].copy()
                for name,q in zip(('A','B','C'),self.bench.curl_presets()):
                    s.set_pose(q);s.save_pose(name)
                s.set_pose(original);self.sync_pose();self.message.content='A/B/C 기본 컬 자세 저장 완료'
            elif kind=='start':
                if not s.active:
                    s.set_gains(action['kp'],action['kd']);s.start(action['rows'])
                    self.message.content=f'기록 중: {s.path}'
            elif kind=='stop':s.finish('user_stopped');self.message.content='중단된 실험을 저장했습니다.'
        except (ValueError,OSError,KeyError,TypeError,AttributeError) as exc:
            self.message.content=f'실행하지 않았습니다: {exc}'
            if action['kind']=='frame' and action['final']:self.sync_pose()

    def refresh(self):
        s=self.session
        self.edit_folder.visible=s.mode=='edit'
        self.measure_folder.visible=s.mode=='measure'
        self.gizmo.visible=s.mode=='edit' and not s.active
        self.edit_button.disabled=self.measure_button.disabled=s.active
        self.start.disabled=s.active
        self.stop.disabled=not s.active
        self.kp.disabled=self.kd.disabled=s.active
        last=max((i for i,(n,_,_) in enumerate(self.rows) if n.value!=SKIP),default=-1)
        for i,(name,seconds,hold) in enumerate(self.rows):
            name.disabled=seconds.disabled=s.active
            hold.disabled=s.active or i==last or name.value==SKIP
        self.saved.content='저장 상태: '+', '.join(f'{name} {"✓" if s.poses[name] is not None else "미저장"}' for name in NAMES)
        xyz,quat=self.bench.ee_pose()
        self.actual.position,self.actual.wxyz=xyz,quat
        if s.state=='COMPLETED':self.message.content=f'완료 · 마지막 점 도착 즉시 기록 종료 · {s.count} samples · {s.path}'


def run_session(args,bench,make_ui):
    meta={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    session=ExperimentSession(bench,args.output,meta)
    bench.session=session
    if args.waypoints is not None:session.load_poses(args.waypoints)
    elif session.saved_path.exists():session.load_poses(session.saved_path)
    server=None
    try:
        if not args.headless:
            server,widgets,status,handles,controls=make_ui(bench,args.port,'residual')
        wall_start=time.monotonic()-bench.data.time
        next_ui=next_print=0.
        while args.run_seconds is None or bench.data.time<args.run_seconds:
            if server is not None:
                controls.update()
                change=getattr(bench,'poll_switch',lambda:None)()
                if change is not None and not session.active:return change
            result=session.tick()
            if result is not None and bench.data.time>=next_print:
                torque,xyz,velocity=result
                print(f't={session.elapsed:.3f}s {session.state} point={session.plan[session.index]["name"]} torque={np.round(torque,3)} Nm velocity={np.round(velocity,3)} rad/s',flush=True)
                next_print=bench.data.time+1/args.print_hz
            if server is not None and bench.data.time>=next_ui:
                with server.atomic():
                    for g,handle in handles:
                        quat=np.empty(4);mujoco.mju_mat2Quat(quat,bench.data.geom_xmat[g])
                        handle.position=bench.data.geom_xpos[g].copy();handle.wxyz=quat
                    bench.com_visual.update()
                    controls.refresh()
                    status.content=f'**{session.state}** · {"기록 중" if session.active else "기록 안 함"} · samples {session.count}\n\nEE: {np.round(bench.ee_pose()[0],4)}'
                next_ui=bench.data.time+1/30
            delay=wall_start+bench.data.time-time.monotonic()
            if delay>0:time.sleep(delay)
    except KeyboardInterrupt:
        session.finish('interrupted')
    except Exception:
        session.finish('error')
        raise
    finally:
        session.finish('program_exit')
        if server is not None:server.stop()
