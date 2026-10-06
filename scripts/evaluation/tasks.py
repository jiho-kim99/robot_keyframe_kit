"""Robot role mappings and duration-independent representative task paths."""
from pathlib import Path
import numpy as np
import yaml
from .trajectory import TimedPath

def load_tasks(path):
    config=yaml.safe_load(Path(path).read_text())
    if not isinstance(config,dict) or config.get('schema_version')!=1:raise ValueError('Tasks require schema_version: 1')
    tasks=config['tasks']
    for name,task in tasks.items():
        if not name.replace('_','').isalnum():raise ValueError('Invalid task ID')
        for key in ('motion_duration','reference_duration'):
            if not np.isfinite(task[key]) or task[key]<=0:raise ValueError(f'{name}: invalid {key}')
    return config

def compile_task(bench,config,task_id):
    task=config['tasks'][task_id]
    roles=bench.platform_metadata['profile_snapshot']['robot'].get('task_joint_roles',{})
    if not roles:raise ValueError('Robot profile needs task_joint_roles for evaluation tasks')
    home=bench.model.qpos0[bench.qadr].copy()
    for role,value in config.get('home_deg',{}).items():
        name=roles.get(role)
        if name in bench.names:home[bench.names.index(name)]=np.deg2rad(value)
    if task.get('planner')=='front_right_rear':
        from .pick_place import compile_pick_place
        return compile_pick_place(bench,task,home,roles)
    points=[]
    for frame in task['keyframes_deg']:
        q=home.copy()
        for role,value in frame.items():
            name=roles.get(role)
            if name not in bench.names:raise ValueError(f'{task_id}: active joint role required: {role}')
            q[bench.names.index(name)]+=np.deg2rad(value)
        points.append(q)
    points=np.asarray(points)
    if np.any(points<bench.bounds[:,0]) or np.any(points>bench.bounds[:,1]):
        raise ValueError(f'{task_id}: path exceeds joint limits; edit amplitude/keyframes')
    if task.get('repeat',True) and not np.allclose(points[0],points[-1],atol=1e-12):
        raise ValueError(f'{task_id}: repeating path must close')
    return TimedPath(points,task.get('phases'))
