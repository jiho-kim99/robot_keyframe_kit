"""Top-down front -> right -> front -> rear pick/place reference choreography.
Objects are event-driven visual previews, not contact or grasp dynamics.
"""
import numpy as np
import mujoco
from scipy.optimize import least_squares
from .trajectory import TimedPath
from scipy.interpolate import CubicSpline
import hashlib

class PickPlacePath(TimedPath):
    def __init__(self,points,phases,arc):
        super().__init__(points,phases)
        self.arc=np.asarray(arc)
        self.arc_spline=CubicSpline(np.linspace(0,1,len(arc)),self.arc)
        self.fingerprint=hashlib.sha256(self.points.tobytes()+self.phases.tobytes()+self.arc.tobytes()).hexdigest()
    def at(self,elapsed,duration,repeat=True):
        q,v,a=super().at(elapsed,duration,repeat)
        phase=(max(0.,elapsed)/duration)%1 if repeat else np.clip(elapsed/duration,0,1)
        segment=min(np.searchsorted(self.phases,phase,side="right")-1,len(self.points)-2)
        if segment not in (3,7):return q,v,a
        width=self.phases[segment+1]-self.phases[segment]
        u=np.clip((phase-self.phases[segment])/width,0,1);dt=width*duration
        f=10*u**3-15*u**4+6*u**5
        df=(30*u**2-60*u**3+30*u**4)/dt
        ddf=(60*u-180*u**2+120*u**3)/dt**2
        if segment==7:f,df,ddf=1-f,-df,-ddf
        q=self.arc_spline(f);v=self.arc_spline(f,1)*df
        a=self.arc_spline(f,2)*df**2+self.arc_spline(f,1)*ddf
        return q,v,a



def compile_pick_place(bench, task, home, roles):
    cfg=task.get('layout',{})
    if bench.platform_metadata['end_effector']!='right':
        raise ValueError('Front/right/rear pick-and-place requires the right end effector')
    arm_roles=['r_'+part for part in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_pitch','wrist_roll','wrist_yaw','wrist_pitch')]
    required=arm_roles+['waist_yaw','waist_pitch']
    if any(roles.get(role) not in bench.names for role in required):
        raise ValueError('Pick-and-place requires right_arm and waist groups')
    ids=np.array([bench.names.index(roles[role]) for role in arm_roles]);yaw=bench.names.index(roles['waist_yaw']);pitch=bench.names.index(roles['waist_pitch'])
    m=bench.model;data=mujoco.MjData(m);home=home.copy();home[yaw]=home[pitch]=0
    data.qpos[bench.qadr]=home;mujoco.mj_forward(m,data)
    waist_joint=m.joint(roles['waist_pitch']);z=float(data.xanchor[waist_joint.id,2])+float(cfg.get('table_height_offset_m',0))
    radius=float(cfg.get('radius_m',.38));height=float(cfg.get('object_height_m',.06));grasp_clearance=float(cfg.get('grasp_offset_m',.025))
    lift=float(cfg.get('lift_clearance_m',.20));rim=float(cfg.get('basket_height_m',.10))
    if not np.isfinite([z,radius,height,grasp_clearance,lift,rim]).all() or min(z,radius,height,lift,rim)<=0 or grasp_clearance<0 or lift<=rim+height:
        raise ValueError('Invalid pick/place layout dimensions or insufficient rim clearance')
    front=np.array([radius,0,z+height+grasp_clearance]);side=np.array([0,-radius,z+rim+height+grasp_clearance]);high_z=z+height+grasp_clearance+lift
    low,high=bench.bounds[ids].T.copy()
    # Deterministic continuation: same geometry for every duration and restart.
    seed=np.clip(np.deg2rad([0,-30,0,90,0,0,45]),low+1e-8,high-1e-8)
    def solve(target,previous,top_down=True):
        anchor=previous.copy()
        def residual(q):
            data.qpos[bench.qadr]=home;data.qpos[bench.qadr[ids]]=q;mujoco.mj_forward(m,data)
            axis=data.site_xmat[bench.sid].reshape(3,3)[:,0]
            return np.r_[data.site_xpos[bench.sid]-target,(.25*(axis-[0,0,-1]) if top_down else np.zeros(3)),1e-3*(q-anchor)]
        result=least_squares(residual,previous,bounds=(low+1e-9,high-1e-9),max_nfev=500,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        err=residual(result.x)
        if np.linalg.norm(err[:3])>.002 or np.linalg.norm(err[3:6])/.25>.025:
            raise ValueError(f'Pick/place top-down target is unreachable: {target.tolist()} (position error {np.linalg.norm(err[:3]):.4f} m). Adjust layout.radius_m/height.')
        q=home.copy();q[ids]=result.x;return q
    front_low=solve(front,seed)
    front_high=solve(np.array([*front[:2],high_z]),front_low[ids])
    arc=[front_high]
    for theta in np.linspace(0,-np.pi/2,25)[1:]:
        arc.append(solve(np.array([radius*np.cos(theta),radius*np.sin(theta),high_z]),arc[-1][ids],False))
    side_high=arc[-1]
    side_low=solve(side,side_high[ids],False)
    rear_high=front_high.copy();rear_high[yaw]=-np.pi
    rear_low=solve(np.array([radius,0,side[2]]),front_high[ids]);rear_low[yaw]=-np.pi
    # Segment fractions include grasp/place dwells and return to front for repetition.
    frames=[('front_approach',front_high),('front_pick_1',front_low),('grasp_1',front_low),
            ('lift_1',front_high),('right_approach',side_high),('right_place',side_low),('release_right',side_low),
            ('right_retreat',side_high),('front_return',front_high),('front_pick_2',front_low),('grasp_2',front_low),
            ('lift_2',front_high),('rear_approach',rear_high),('rear_place',rear_low),('release_rear',rear_low),
            ('rear_retreat',rear_high),('cycle_front',front_high)]
    phases=np.array([0,.055,.075,.13,.23,.275,.295,.34,.42,.475,.495,.55,.73,.775,.795,.84,1.])
    points=np.array([q for _,q in frames])
    if np.any(points<bench.bounds[:,0]) or np.any(points>bench.bounds[:,1]):raise ValueError('Pick/place exceeds robot joint limits')
    path=PickPlacePath(points,phases,arc)
    sampled=np.array([path.at(t,1,False)[0] for t in np.linspace(0,1,1001)])
    if np.any(sampled<bench.bounds[:,0]-1e-6) or np.any(sampled>bench.bounds[:,1]+1e-6):raise ValueError("Pick/place interpolated path exceeds joint limits")
    path.pick_place=dict(transfer_arc_joint_positions_rad=np.asarray(arc).tolist(),table_height_m=z,radius_m=radius,object_height_m=height,basket_height_m=rim,
        front_object_xyz=[radius,0,z+height/2],right_basket_xyz=[0,-radius,z],rear_basket_xyz=[-radius,0,z],
        grasp_offset_m=height/2+grasp_clearance,sequence=['front','right_90','front','rear_180'],
        waypoint_names=[name for name,_ in frames],waypoint_phases=phases.tolist(),
        attach_phases=[.075,.495],release_phases=[.295,.795],
        waist_yaw_rad={'front':0.,'right_90':0.,'rear_180':-float(np.pi)},
        object_model='event-driven visual attachment; GUI mass applies prescribed EE gravity while carrying; no object inertia/contact')
    return path
