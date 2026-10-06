#!/usr/bin/env python3
"""18 representative upper-body actuator tasks with user-defined cycle duration."""
import argparse
from collections import deque
from pathlib import Path
import threading
import time
import mujoco
import numpy as np
from manipulation_platform import load_catalog,build_experiment
from arm_payload import make_ui
from evaluation.tasks import load_tasks,compile_task
from evaluation.runtime import TaskRunner
from evaluation.environments import TaskEnvironment
ROOT=Path(__file__).resolve().parents[1]

class TaskUI:
    def __init__(self,runner,server):
        self.r=runner;self.queue=deque();self.lock=threading.Lock()
        with server.gui.add_folder('Actuator evaluation tasks'):
            self.task=server.gui.add_dropdown('Task',options=tuple(runner.config['tasks']),initial_value=runner.task_id)
            self.task.on_update(lambda e:self.put('select',task=self.task.value) if e.client_id is not None else None)
            self.duration=server.gui.add_number('Motion Duration / Cycle Time [s]',initial_value=runner.duration,min=2*runner.b.model.opt.timestep,step=.1)
            server.gui.add_markdown('One complete cycle. Duration is applied on Start; geometric path stays fixed. Selection changes stop the previous log.')
            self.object_mass=server.gui.add_number('Object Mass [kg]',initial_value=float(runner.task.get('load',{}).get('object_mass_kg',0)),min=0.,step=.1)
            server.gui.add_markdown('Mass is applied on Start. Pick/place: gravity load only while carrying. No object acceleration/rotation inertia or contact grasp.')
            self.start=server.gui.add_button('Start / Resume Task')
            self.start.on_click(lambda _:self.put('start',duration=self.duration.value,kp=self.kp.value,kd=self.kd.value,object_mass=self.object_mass.value))
            server.gui.add_button('Pause Task').on_click(lambda _:self.put('pause'))
            server.gui.add_button('Stop Task').on_click(lambda _:self.put('stop'))
            server.gui.add_button('Reset Task').on_click(lambda _:self.put('reset'))
            self.kp=server.gui.add_number('Kp',initial_value=float(runner.b.kp),min=0.,max=1000.,step=10.)
            self.kd=server.gui.add_number('Kd',initial_value=float(runner.b.kd),min=0.,max=200.,step=1.)
            self.log_start=server.gui.add_button('Start Logging')
            self.log_start.on_click(lambda _:self.put('log_start'))
            self.log_stop=server.gui.add_button('Stop Logging')
            self.log_stop.on_click(lambda _:self.put('log_stop'))
            self.info=server.gui.add_markdown('Ready. Logging OFF.')
            self.error=server.gui.add_markdown('')
    def put(self,kind,**values):
        with self.lock:self.queue.append((kind,values))
    def update(self):
        with self.lock:action=self.queue.popleft() if self.queue else None
        if action:
            kind,v=action
            try:
                if kind=='select':
                    self.r.select(v['task']);self.duration.value=self.r.duration
                    self.object_mass.value=float(self.r.task.get('load',{}).get('object_mass_kg',0))
                elif kind=='start':
                    kp,kd=float(v['kp']),float(v['kd'])
                    if not np.isfinite([kp,kd]).all() or not 0<=kp<=1000 or not 0<=kd<=200:raise ValueError('Invalid gains')
                    self.r.set_object_mass(v['object_mass'])
                    if (kp,kd)!=(self.r.b.kp,self.r.b.kd):self.r.logger.stop('gains_changed')
                    self.r.b.kp,self.r.b.kd=kp,kd
                    self.r.start(v['duration'])
                elif kind=='pause':self.r.pause()
                elif kind=='stop':self.r.stop()
                elif kind=='reset':self.r.reset()
                elif kind=='log_start':self.r.start_logging()
                elif kind=='log_stop':self.r.logger.stop()
                self.error.content=''
            except (ValueError,KeyError,OSError) as exc:self.error.content=f'Not applied: {exc}'
        self.log_start.disabled=self.r.logger.active;self.log_stop.disabled=not self.r.logger.active
        self.kp.disabled=self.kd.disabled=self.r.logger.active
        self.object_mass.disabled=self.r.logger.active or self.r.state in ('RUNNING','PREPARING')
        layout=getattr(self.r.path,'pick_place',None)
        stage=''
        if layout:
            phase=(self.r.task_time/self.r.duration)%1
            index=max(0,min(len(layout['waypoint_names'])-1,int(np.searchsorted(layout['waypoint_phases'],phase,side='right')-1)))
            stage=f"Stage: **{layout['waypoint_names'][index]}** · Front → Right 90° → Front → Rear 180°\n\n"
        self.info.content=(stage+f'**{self.r.task["label"]} · {self.r.state}**\n\n'
            f"Object: **{self.r.task.get('load',{}).get('object_mass_kg',0):g} kg** · applied load: **{self.r.active_object_mass():g} kg**\n\n"
            f'Applied cycle: **{self.r.duration:g} s** · frequency: {1/self.r.duration:.3f} Hz\n\n'
            f'Logging **{"ON" if self.r.logger.active else "OFF"}** · {self.r.logger.count} samples\n\n'
            f'{self.r.logger.path or "No recording yet"}\n\n'
            'Representative motion + prescribed wrench; no grasp/contact success model.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/manipulation_platform.yaml')
    p.add_argument('--tasks',type=Path,default=ROOT/'configs/evaluation_tasks.yaml')
    p.add_argument('--robot',default='prototype_barrett');p.add_argument('--variant',default='base')
    p.add_argument('--groups',nargs='+');p.add_argument('--ee',default='right')
    p.add_argument('--task',default='pick_and_place');p.add_argument('--motion-duration',type=float)
    p.add_argument('--object-mass-kg',type=float,help='Prescribed gravity load, not object inertia/contact')
    p.add_argument('--transition-duration',type=float,default=1.)
    p.add_argument('--kp',type=float,default=200);p.add_argument('--kd',type=float,default=40)
    p.add_argument('--dt',type=float,default=.002);p.add_argument('--port',type=int,default=8766)
    p.add_argument('--output',type=Path,default=ROOT/'task_logs')
    p.add_argument('--headless',action='store_true');p.add_argument('--run-seconds',type=float)
    p.add_argument('--log',action='store_true',help='Explicitly start logging (also in headless runs)')
    p.add_argument('--autostart',action='store_true');p.add_argument('--check-all',action='store_true')
    args=p.parse_args();server=None;runner=None
    try:
        if not np.isfinite([args.dt,args.kp,args.kd]).all() or args.dt<=0 or not 0<=args.kp<=1000 or not 0<=args.kd<=200:raise ValueError('Invalid timestep or gains')
        if args.run_seconds is not None and (not np.isfinite(args.run_seconds) or args.run_seconds<=0):raise ValueError('run-seconds must be positive')
        if args.headless and args.run_seconds is None and not args.check_all:raise ValueError('--headless requires --run-seconds')
        catalog,base=load_catalog(args.config)
        b=build_experiment(catalog,base,(args.robot,'waypoint_motion',args.variant,args.ee),args.groups,args.kp,args.kd,args.dt)
        config=load_tasks(args.tasks)
        if args.check_all:
            for task in config['tasks']:
                runner=TaskRunner(b,config,task,args.output,args.transition_duration)
                print(task,runner.path.points.shape,runner.path.fingerprint[:12])
            return
        runner=TaskRunner(b,config,args.task,args.output,args.transition_duration)
        if args.motion_duration is not None:runner.set_duration(args.motion_duration)
        if args.object_mass_kg is not None:runner.set_object_mass(args.object_mass_kg)
        if args.autostart:runner.start()
        if args.log:runner.start_logging()
        if not args.headless:
            b.platform_title='Upper-body actuator evaluation · '+args.robot+' · '+args.variant
            server,_,status,handles,_=make_ui(b,args.port,'evaluation')
            ui=TaskUI(runner,server)
            environment=TaskEnvironment(runner,server)
        start=time.monotonic();next_ui=0.
        while args.run_seconds is None or b.data.time<args.run_seconds:
            if server:ui.update()
            runner.tick()
            if server and b.data.time>=next_ui:
                # Update FK after integration, then renderer at 30Hz.
                mujoco.mj_forward(b.model,b.data)
                with server.atomic():
                    for g,h in handles:
                        quat=np.empty(4);mujoco.mju_mat2Quat(quat,b.data.geom_xmat[g]);h.position=b.data.geom_xpos[g].copy();h.wxyz=quat
                    b.com_visual.update()
                    environment.update()
                    status.content=f'**{runner.task["label"]} — {runner.duration:g} s/cycle** · {runner.state} · Logging {"ON" if runner.logger.active else "OFF"}'
                next_ui=b.data.time+1/30
            if server:
                delay=start+b.data.time-time.monotonic()
                if delay>0:time.sleep(delay)
    except KeyboardInterrupt:
        if runner:runner.logger.stop('interrupted')
    except (ValueError,KeyError,OSError) as exc:
        if runner:runner.logger.stop('error')
        p.error(str(exc))
    except Exception:
        if runner:runner.logger.stop('simulation_error')
        raise
    finally:
        if runner:runner.logger.stop('program_exit')
        if server:server.stop()

if __name__=='__main__':main()
