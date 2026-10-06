"""Clock-driven tasks, independent logging, external wrench and dynamics sampling."""
import copy
import numpy as np
import mujoco
from .trajectory import TimedPath
from .tasks import compile_task
from .logger import TaskLogger

class TaskRunner:
    def __init__(self,bench,config,task_id,output,transition_duration=1.):
        self.b=bench;self.config=config;self.logger=TaskLogger(output)
        self.transition_duration=float(transition_duration)
        if not np.isfinite(self.transition_duration) or self.transition_duration<=0:raise ValueError('transition_duration must be positive')
        self.state='IDLE';self.task_time=0.;self.prep_time=0.;self.log_start=0.
        self.ref=mujoco.MjData(bench.model)
        self.hold=bench.data.qpos[bench.qadr].copy()
        profile=bench.platform_metadata['profile_snapshot']['robot']
        self.ee_sites={}
        for name,entry in profile['end_effectors'].items():
            sid=mujoco.mj_name2id(bench.model,mujoco.mjtObj.mjOBJ_SITE,entry['site'])
            if sid>=0:self.ee_sites[name]=sid
        self.groups={j:g for g,js in profile['joint_groups'].items() for j in js if j in bench.names}
        self.select(task_id)
    def select(self,task_id):
        # Compile before mutating current task or closing its recording.
        path=compile_task(self.b,self.config,task_id)
        task=copy.deepcopy(self.config['tasks'][task_id])
        self.validate_load(task.get('load',{}))
        resume=self.state in ('RUNNING','PREPARING')
        self.logger.stop('task_changed')
        self.task_id,self.task,self.path=task_id,task,path
        self.duration=float(task['motion_duration']);self.task_time=0
        self.hold=self.b.data.qpos[self.b.qadr].copy();self.state='IDLE'
        if resume:self.start()
    def validate_load(self,load):
        for key in ('resistance_force_N','resistance_torque_Nm','axial_force_N','reaction_torque_Nm','object_mass_kg'):
            value=load.get(key,0)
            if not np.isfinite(value) or value<0:raise ValueError(f'{key} must be finite and nonnegative')
        for key in ('force_world_N','torque_world_Nm'):
            v=np.asarray(load.get(key,[0,0,0]),dtype=float)
            if v.shape!=(3,) or not np.isfinite(v).all():raise ValueError(f'Invalid {key}')
        for ee in load.get('sites',[self.b.platform_metadata['end_effector']]):
            if ee not in self.ee_sites:raise ValueError(f'Task load needs EE site {ee}')
    def set_object_mass(self,mass):
        mass=float(mass)
        if not np.isfinite(mass) or mass<0:
            raise ValueError('Object mass must be finite and nonnegative')
        if mass==self.task.get('load',{}).get('object_mass_kg',0):return
        if self.state in ('RUNNING','PREPARING'):
            raise ValueError('Pause or Stop before changing object mass')
        self.logger.stop('object_mass_changed')
        self.task.setdefault('load',{})['object_mass_kg']=mass
    def active_object_mass(self):
        mass=float(self.task.get('load',{}).get('object_mass_kg',0))
        layout=getattr(self.path,'pick_place',None)
        if layout is None:return mass
        if self.state=='IDLE':return 0.
        phase=(self.task_time/self.duration)%1
        return mass if any(a<=phase<z for a,z in zip(layout['attach_phases'],layout['release_phases'])) else 0.
    def set_duration(self,duration):
        duration=float(duration)
        if not np.isfinite(duration) or duration<2*self.b.model.opt.timestep:
            raise ValueError('Motion Duration must be finite and at least two simulation timesteps')
        if duration!=self.duration:
            self.logger.stop('duration_changed')
            self.duration=duration;self.task_time=0;self.state='IDLE'
            self.hold=self.b.data.qpos[self.b.qadr].copy()
    def start(self,duration=None):
        if duration is not None:self.set_duration(duration)
        if self.state in ('RUNNING','PREPARING'):return
        target=self.path.at(self.task_time,self.duration,self.task.get('repeat',True))[0]
        self.preparation=TimedPath([self.b.data.qpos[self.b.qadr].copy(),target])
        self.prep_time=0.;self.state='PREPARING';self.reset_only=False
    def pause(self):
        if self.state not in ('RUNNING','PREPARING'):return
        self.hold=self.b.data.qpos[self.b.qadr].copy();self.state='PAUSED'
    def reset(self):
        self.logger.stop('reset')
        self.task_time=0.;self.state='IDLE';self.hold=self.b.data.qpos[self.b.qadr].copy()
        self.start();self.reset_only=True
    def stop(self):
        self.pause();self.logger.stop('task_stopped');self.task_time=0.;self.state='IDLE'
    def metadata(self):
        b=self.b
        masses={b.model.body(i).name:float(b.model.body_mass[i]) for i in range(1,b.model.nbody)}
        return dict(**b.platform_metadata,task_name=self.task_id,task_label=self.task['label'],
            motion_duration=self.duration,cycle_time=self.duration,
            trajectory_frequency_hz=1/self.duration if self.task.get('repeat',True) else None,
            trajectory_speed_scale=self.task['reference_duration']/self.duration,
            reference_duration=self.task['reference_duration'],task_parameters=self.task,
            task_layout=getattr(self.path,'pick_place',None),
            path_hash=self.path.fingerprint,path_joint_positions_rad=self.path.points.tolist(),
            path_phases=self.path.phases.tolist(),joint_names=b.names,joint_groups=self.groups,
            simulation_timestep=b.model.opt.timestep,kp=b.kp,kd=b.kd,
            robot_body_masses_kg=masses,
            end_effector_body_mass_kg=masses.get(getattr(b,'com_body',None)),
            payload_mass_kg=(0.0 if 'hand' in b.platform_metadata['profile_snapshot']['robot'] else masses.get(getattr(b,'com_body',None))),
            payload_definition=('No attached payload; hand mass belongs to robot' if 'hand' in b.platform_metadata['profile_snapshot']['robot'] else 'Total mass of configured COM body; additional static object weight is object_mass_kg'),
            object_mass_kg=self.task.get('load',{}).get('object_mass_kg',0),
            object_mass_application=('Gravity at EE during reference attach/release intervals; retained while paused' if hasattr(self.path,'pick_place') else 'Prescribed gravity at configured EE sites'),
            environment_mode='Visual objects; mass input produces prescribed EE gravity only, no simulated contact or object inertia',
            external_load_model='Prescribed EE wrench; object weight is static force only; no impact/grasp/object inertia simulation',
            external_force=self.task.get('load',{}),transition_duration=self.transition_duration,
            logging_start_sim_time_s=float(b.data.time),logging_start_task_time_s=self.task_time,
            cycle_duration_definition='Reference clock; settling/tracking error never stretches the cycle',
            torque_definition='joint-side applied actuator torque after saturation',
            power_definition='applied actuator torque times actual joint velocity; signed power; peak is absolute')
    def start_logging(self):
        if self.logger.active:return
        self.log_start=float(self.b.data.time)
        self.logger.start(self.metadata(),self.columns())
    def columns(self):
        names=['time_s','simulation_time_s','task_name','phase','task_time_s','cycle_index','cycle_phase','motion_duration_s','active_object_mass_kg']
        for j in self.b.names:
            names += [j+'_'+f for f in ('position_rad','velocity_rad_s','acceleration_rad_s2','target_rad',
                'reference_velocity_rad_s','reference_acceleration_rad_s2','torque_requested_Nm','torque_Nm',
                'inverse_torque_Nm','external_load_Nm','power_W','position_error_rad','saturated')]
        for ee in self.ee_sites:
            names += [ee+'_'+f for f in ('ee_x_m','ee_y_m','ee_z_m','ee_w','ee_qx','ee_qy','ee_qz',
                'ee_vx_m_s','ee_vy_m_s','ee_vz_m_s','ee_wx_rad_s','ee_wy_rad_s','ee_wz_rad_s',
                'force_x_N','force_y_N','force_z_N','moment_x_Nm','moment_y_Nm','moment_z_Nm')]
        return names
    def reference(self):
        if self.state=='PREPARING':return self.preparation.at(self.prep_time,self.transition_duration,False)
        if self.state=='RUNNING':return self.path.at(self.task_time,self.duration,self.task.get('repeat',True))
        return self.hold.copy(),np.zeros(len(self.b.names)),np.zeros(len(self.b.names))
    def load_wrenches(self,vref):
        b=self.b;load=self.task.get('load',{})
        wrenches={ee:(np.zeros(3),np.zeros(3)) for ee in self.ee_sites}
        sites=load.get('sites',[b.platform_metadata['end_effector']])
        qload=np.zeros(b.model.nv)
        full_v=np.zeros(b.model.nv);full_v[b.dadr]=vref
        for ee in sites:
            sid=self.ee_sites[ee]
            jp,jr=np.zeros((3,b.model.nv)),np.zeros((3,b.model.nv))
            mujoco.mj_jacSite(b.model,b.data,jp,jr,sid)
            linear,angular=jp@full_v,jr@full_v
            # Smooth sign changes near reversal instead of discontinuous Coulomb loads.
            force=np.asarray(load.get('force_world_N',[0,0,0]),dtype=float).copy()
            moment=np.asarray(load.get('torque_world_Nm',[0,0,0]),dtype=float).copy()
            axis=b.data.site_xmat[sid].reshape(3,3)[:,0]
            force-=load.get('axial_force_N',0)*axis
            moment-=load.get('reaction_torque_Nm',0)*axis
            force-=load.get('resistance_force_N',0)*linear/np.sqrt(linear@linear+1e-6)
            moment-=load.get('resistance_torque_Nm',0)*angular/np.sqrt(angular@angular+1e-6)
            force+=self.active_object_mass()*b.model.opt.gravity/len(sites)
            mujoco.mj_applyFT(b.model,b.data,force,moment,b.data.site_xpos[sid],int(b.model.site_bodyid[sid]),qload)
            wrenches[ee]=(force,moment)
        return qload,wrenches
    def tick(self):
        b=self.b;dt=b.model.opt.timestep
        qref,vref,aref=self.reference()
        mujoco.mj_forward(b.model,b.data)
        qload,wrenches=self.load_wrenches(vref)
        b.data.qfrc_applied[:]=qload
        self.ref.qpos[:]=b.data.qpos;self.ref.qvel[:]=0;self.ref.qacc[:]=0
        self.ref.qpos[b.qadr]=qref;self.ref.qvel[b.dadr]=vref;self.ref.qacc[b.dadr]=aref
        mujoco.mj_inverse(b.model,self.ref)
        ff=self.ref.qfrc_inverse[b.dadr].copy()
        q=b.data.qpos[b.qadr].copy();v=b.data.qvel[b.dadr].copy()
        raw=ff-qload[b.dadr]+b.kp*(qref-q)+b.kd*(vref-v)
        b.data.ctrl[:]=np.clip(raw,b.limits[:,0],b.limits[:,1])
        mujoco.mj_forward(b.model,b.data)
        tau=b.data.qfrc_actuator[b.dadr].copy();acc=b.data.qacc[b.dadr].copy()
        if not np.isfinite(np.concatenate([q,v,tau,acc])).all():
            self.logger.stop('simulation_error');raise RuntimeError('Non-finite simulation state')
        row=dict(time_s=float(b.data.time)-self.log_start,simulation_time_s=float(b.data.time),
            task_name=self.task_id,phase=self.state,task_time_s=self.task_time,
            cycle_index=int(self.task_time/self.duration),cycle_phase=(self.task_time/self.duration)%1,motion_duration_s=self.duration,active_object_mass_kg=self.active_object_mass())
        fields=('position_rad','velocity_rad_s','acceleration_rad_s2','target_rad','reference_velocity_rad_s',
            'reference_acceleration_rad_s2','torque_requested_Nm','torque_Nm','inverse_torque_Nm','external_load_Nm',
            'power_W','position_error_rad','saturated')
        for i,j in enumerate(b.names):
            values=(q[i],v[i],acc[i],qref[i],vref[i],aref[i],raw[i],tau[i],ff[i],qload[b.dadr[i]],tau[i]*v[i],qref[i]-q[i],float(abs(raw[i]-tau[i])>1e-8))
            row.update({j+'_'+f:float(x) for f,x in zip(fields,values)})
        for ee,sid in self.ee_sites.items():
            quat=np.empty(4);mujoco.mju_mat2Quat(quat,b.data.site_xmat[sid])
            jp,jr=np.zeros((3,b.model.nv)),np.zeros((3,b.model.nv));mujoco.mj_jacSite(b.model,b.data,jp,jr,sid)
            values=np.concatenate([b.data.site_xpos[sid],quat,jp@b.data.qvel,jr@b.data.qvel,*wrenches[ee]])
            fields=('ee_x_m','ee_y_m','ee_z_m','ee_w','ee_qx','ee_qy','ee_qz','ee_vx_m_s','ee_vy_m_s','ee_vz_m_s',
                'ee_wx_rad_s','ee_wy_rad_s','ee_wz_rad_s','force_x_N','force_y_N','force_z_N','moment_x_Nm','moment_y_Nm','moment_z_Nm')
            row.update({ee+'_'+f:float(x) for f,x in zip(fields,values)})
        self.logger.write(row)
        old=float(b.data.time);mujoco.mj_step(b.model,b.data)
        if b.data.time<=old or not np.isfinite(b.data.qpos).all():
            self.logger.stop('simulation_error');raise RuntimeError('Simulation time reset/diverged')
        if self.state=='PREPARING':
            self.prep_time+=dt
            if self.prep_time>=self.transition_duration:
                self.state='PAUSED' if getattr(self,'reset_only',False) else 'RUNNING'
                self.hold=self.preparation.points[-1].copy();self.reset_only=False
        elif self.state=='RUNNING':
            self.task_time+=dt
            if not self.task.get('repeat',True) and self.task_time>=self.duration:
                self.hold=self.path.points[-1].copy();self.state='COMPLETED';self.logger.stop('completed')
        self.last_row=row
        return row
