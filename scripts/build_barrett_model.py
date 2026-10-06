from pathlib import Path
import sys, shutil, copy, subprocess
import xml.etree.ElementTree as E
import numpy as np
from scipy.spatial.transform import Rotation
import xacro
import argparse, tempfile
parser=argparse.ArgumentParser(description='Generate Barrett model from a local upstream checkout; requires xacro, numpy, scipy, PyYAML.')
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True,help='Staging output directory')
args=parser.parse_args()
SRC=args.source.resolve(); REPO=Path(__file__).resolve().parents[1]; OUT=args.output.resolve()
HAND=OUT/'third_party/barrett_model'; HAND.mkdir(parents=True,exist_ok=True)
for name in ['README.md','package.xml'] : shutil.copy2(SRC/name,HAND/name)
shutil.copytree(SRC/'models',HAND/'models',dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.blend'))
# Vendor only BHand meshes and its two macro sources.
for p in (HAND/'models').iterdir():
 if p.name not in ['hand.urdf.xacro','common.urdf.xacro','sw_meshes']: shutil.rmtree(p) if p.is_dir() else p.unlink()
for p in (HAND/'models/sw_meshes').iterdir():
 if p.name!='bhand':
  if p.is_dir(): shutil.rmtree(p)
  else:p.unlink()
sha=subprocess.check_output(['git','-C',str(SRC),'rev-parse','HEAD'],text=True).strip()
(HAND/'PROVENANCE.md').write_text(f'Source: https://github.com/jhu-lcsr/barrett_model\nCommit: {sha}\nLicense: GPL; common.urdf.xacro specifies GPL-3.0-or-later.\nOriginal source and mesh files retained. Generated integration changes: transparent materials made opaque, DAE collisions replaced with supplied STL convex pieces, palm mounted to right wrist pitch.\n')
shutil.copy2('/usr/share/common-licenses/GPL-3',HAND/'COPYING')
# Resolve ROS package discovery only in temporary build input.
temporary=tempfile.TemporaryDirectory(prefix='barrett_expand_');tmp=Path(temporary.name)
(tmp/'common.urdf.xacro').write_text((HAND/'models/common.urdf.xacro').read_text())
(tmp/'hand.urdf.xacro').write_text((HAND/'models/hand.urdf.xacro').read_text().replace('$(find barrett_model)/models/common.urdf.xacro',str(tmp/'common.urdf.xacro')))
(tmp/'entry.xacro').write_text(f'<robot name="bhand" xmlns:xacro="http://www.ros.org/wiki/xacro"><xacro:include filename="{tmp}/hand.urdf.xacro"/><xacro:bhand parent_link="mount" prefix="right_hand" xyz="0 0 0" rpy="0 0 0"/></robot>')
h=E.fromstring(xacro.process_file(str(tmp/'entry.xacro')).toxml())
def vec(s):return np.fromstring(s,sep=' ')
def fmt(v):return ' '.join(f'{float(x):.12g}' for x in v)
def origin(el):
 o=el.find('origin');return (np.zeros(3),np.eye(3)) if o is None else (vec(o.get('xyz','0 0 0')),Rotation.from_euler('xyz',vec(o.get('rpy','0 0 0'))).as_matrix())
def setorigin(el,p,r):
 o=el.find('origin')
 if o is None:o=E.SubElement(el,'origin')
 o.set('xyz',fmt(p));o.set('rpy',fmt(Rotation.from_matrix(r).as_euler('xyz')))
palm='right_hand/bhand_palm_link'; wrist='right_wrist_pitch_link'
R=Rotation.from_euler('y',np.pi/2).as_matrix(); offset=np.array([0,-.0315,0])
for node in h.iter():
 for k,v in list(node.attrib.items()):
  if v==palm:node.set(k,wrist)
  elif v.startswith('right_hand/'):node.set(k,v.replace('/','_'))
for link in h.findall('link'):
 if link.get('name')==wrist:
  for part in list(link):
   p,r=origin(part);setorigin(part,offset+R@p,R@r)
 for vis in link.findall('visual'):
  vis.find('material/color').set('rgba','.75 .78 .82 1')
 for col in list(link.findall('collision')):
  mesh=col.find('geometry/mesh'); stem=Path(mesh.get('filename')).stem
  parts=sorted((HAND/'models/sw_meshes/bhand').glob(stem+'*.stl'))
  if not parts:raise ValueError(stem)
  link.remove(col)
  for p in parts:
   c=copy.deepcopy(col);c.find('geometry/mesh').set('filename','../../third_party/barrett_model/models/sw_meshes/bhand/'+p.name);link.append(c)
 for mesh in link.findall('.//mesh'):
  if mesh.get('filename').startswith('package://'):mesh.set('filename','../../third_party/barrett_model/models/sw_meshes/bhand/'+Path(mesh.get('filename')).name)
for joint in list(h.findall('joint')):
 if joint.get('name')=='right_hand_bhand_base_joint':h.remove(joint);continue
 if joint.find('parent').get('link')==wrist:
  p,r=origin(joint);setorigin(joint,offset+R@p,R@r)
# URDF: wrist joint retained; old terminal body geometry/inertia removed entirely.
u=E.parse(REPO/'prototype_v2.1.1/urdf/prototype_v2.1.1.urdf').getroot()
for n in list(u):
 if n.get('name') in [wrist,'right_welding_point','right_welder_joint']:u.remove(n)
for n in h:
 if n.tag in ['link','joint']:u.append(copy.deepcopy(n))
def write(root,path):path.parent.mkdir(parents=True,exist_ok=True);E.indent(root);E.ElementTree(root).write(path,encoding='unicode',xml_declaration=True)
write(u,OUT/'prototype_v2.1.1/urdf/prototype_v2.1.1_barrett.urdf')
# MJCF from same transformed URDF components, with exact inertial matrices.
m=E.parse(REPO/'prototype_v2.1.1/mjcf/prototype_v2.1.1.xml').getroot();body=m.find(f'.//body[@name="{wrist}"]')
for n in list(body):
 if n.tag!='joint':body.remove(n)
links={n.get('name'):n for n in h.findall('link')}; joints=h.findall('joint');assets=m.find('asset'); meshes={}
def addlink(name,b):
 l=links[name];it=l.find('inertial')
 if it is not None:
  p,r=origin(it);a=it.find('inertia').attrib
  I=np.array([[float(a['ixx']),float(a['ixy']),float(a['ixz'])],[float(a['ixy']),float(a['iyy']),float(a['iyz'])],[float(a['ixz']),float(a['iyz']),float(a['izz'])]])
  I=r@I@r.T
  E.SubElement(b,'inertial',pos=fmt(p),mass=it.find('mass').get('value'),fullinertia=fmt([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]))
 for kind in ['visual','collision']:
  for i,g in enumerate(l.findall(kind)):
   mesh=g.find('geometry/mesh');file=mesh.get('filename');key=Path(file).stem
   if key not in meshes:E.SubElement(assets,'mesh',name=key,file=file);meshes[key]=True
   p,r=origin(g);quat=Rotation.from_matrix(r).as_quat()[[3,0,1,2]]
   E.SubElement(b,'geom',name=f'{name}_{kind}_{i}',type='mesh',mesh=key,pos=fmt(p),quat=fmt(quat),rgba='.75 .78 .82 1' if kind=='visual' else '.75 .78 .82 0',contype='0' if kind=='visual' else '1',conaffinity='0' if kind=='visual' else '1',group='2' if kind=='visual' else '3')
 for j in joints:
  if j.find('parent').get('link')!=name:continue
  child=j.find('child').get('link');p,r=origin(j)
  cb=E.SubElement(b,'body',name=child,pos=fmt(p),quat=fmt(Rotation.from_matrix(r).as_quat()[[3,0,1,2]]))
  if j.get('type')!='fixed':
   lim=j.find('limit');E.SubElement(cb,'joint',name=j.get('name'),axis=j.find('axis').get('xyz'),range=lim.get('lower')+' '+lim.get('upper'),actuatorfrcrange='-5 5',damping='.11',armature='.001')
  addlink(child,cb)
addlink(wrist,body)
# Tool frame +X points out of palm, retaining evaluator's axial force convention.
E.SubElement(body,'site',name='right_hand_grasp',pos=fmt(offset+R@np.array([0,0,.12])),size='.008',rgba='1 .5 0 1')
write(m,OUT/'prototype_v2.1.1/mjcf/prototype_v2.1.1_barrett.xml')
scene=E.parse(REPO/'prototype_v2.1.1/mjcf/scene.xml').getroot();scene.find('include').set('file','./prototype_v2.1.1_barrett.xml');write(scene,OUT/'prototype_v2.1.1/mjcf/scene_barrett.xml')
import yaml
cfg=yaml.safe_load((REPO/'configs/manipulation_platform.yaml').read_text());profile=copy.deepcopy(cfg['robots']['prototype_v211']);profile['models']={'base':'../prototype_v2.1.1/mjcf/scene_barrett.xml'};profile['end_effectors']['right']={'site':'right_hand_grasp','com_body':wrist};profile['hand']={'model':'Barrett BHand-280','source_commit':sha,'additional_payload_kg':0,'evaluation_fingers':'fixed_closed','mount_xyz_m':offset.tolist(),'mount_rpy_rad':[0,float(np.pi/2),0]};profile['fixed_joint_positions_rad']={f'right_hand_finger_{i}_{part}_joint':float(np.deg2rad(angle)) for i in (1,2,3) for part,angle in [('med',90),('dist',30)]};profile['fixed_joint_positions_rad'].update({f'right_hand_finger_{i}_prox_joint':0.0 for i in (1,2)});cfg['robots']['prototype_barrett']=profile;(OUT/'configs').mkdir(exist_ok=True);(OUT/'configs/manipulation_platform.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
(OUT/'scripts').mkdir(exist_ok=True)
for name in ['evaluate_tasks.py','manipulation_platform.py']:
 (OUT/'scripts'/name).write_text((REPO/'scripts'/name).read_text().replace("default='prototype_v211'","default='prototype_barrett'"))
print('Built',OUT,'commit',sha)
