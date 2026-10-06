"""Task-specific visual workcells. No collision/mass/contact changes to MuJoCo."""
import numpy as np
import mujoco
import trimesh
from scipy.spatial.transform import Rotation

KINDS={
 'pick_and_place':'factory','heavy_box_lift':'lifting','package_carrying':'warehouse',
 'push_pull':'push_station','door_opening':'door','drawer':'drawer',
 'screwdriver':'assembly','valve_turning':'valve','jar_opening':'jar','drilling':'drilling',
 'wrench_turning':'wrench','lever_operation':'lever','wiping':'wall','stirring':'mixing',
 'pouring':'pouring','hammering':'hammer','throwing':'throwing','overhead_work':'overhead'}

class TaskEnvironment:
    def __init__(self,runner,server):
        self.r,self.server=runner,server;self.handles=[];self.task=None;self.anim=[];self.index=0
        with server.gui.add_folder('Task environment'):
            self.enabled=server.gui.add_checkbox('Show environment',initial_value=True)
            self.caption=server.gui.add_markdown('Visual workcell only. Forces remain configured task loads; no object contact physics.')
        self.update()
    def node(self,kind,**kw):
        self.index+=1
        h=getattr(self.server.scene,'add_'+kind)(f'/task_environment/{self.index}',**kw)
        self.handles.append(h);return h
    def box(self,pos,size,color=(90,110,125),**kw):
        return self.node('box',position=np.asarray(pos),dimensions=size,color=color,**kw)
    def cylinder(self,pos,radius,height,color=(100,110,120),rotation=None):
        mesh=trimesh.creation.cylinder(radius=radius,height=height,sections=32)
        return self.node('mesh_simple',vertices=mesh.vertices,faces=mesh.faces,position=pos,color=color,
                         wxyz=(1,0,0,0) if rotation is None else rotation)
    def label(self,text,pos):return self.node('label',text=text,position=pos,font_screen_scale=.7)
    def table(self,x,y,z,width=.8,depth=.55):
        self.box((x,y,z-.025),(depth,width,.05),(100,119,128))
        for dx in (-depth*.4,depth*.4):
            for dy in (-width*.4,width*.4):self.box((x+dx,y+dy,(z-.05)/2),(.035,.035,max(.03,z-.05)),(45,58,67))
    def tray(self,pos,color=(36,128,155),size=(.2,.25,.09)):
        x,y,z=pos;d,w,h=size
        self.box((x,y,z),(d,w,.015),color)
        for yy in (-w/2,w/2):self.box((x,y+yy,z+h/2),(d,.012,h),color)
        for xx in (-d/2,d/2):self.box((x+xx,y,z+h/2),(.012,w,h),color)
    def reference_ee(self,q):
        b=self.r.b;d=mujoco.MjData(b.model);d.qpos[:]=b.data.qpos;d.qpos[b.qadr]=q;mujoco.mj_forward(b.model,d)
        return d.site_xpos[b.sid].copy()
    def build(self):
        for h in reversed(self.handles):h.remove()
        self.handles=[];self.anim=[];self.index=0
        self.task=self.r.task_id
        self.caption.content='Visual workcell only. Forces remain configured task loads; no object contact physics.'
        cfg=self.r.task.get('environment',{})
        self.kind=cfg.get('kind',KINDS.get(self.task,'assembly'))
        self.anchor=self.reference_ee(self.r.path.points[0])+np.asarray(cfg.get('offset_m',[0,0,0]),dtype=float)
        x,y,z=self.anchor;z=max(z,.65);self.anchor[2]=z
        self.box((x+.3,y,.007),(1.65,2.3,.012),(62,72,82))
        for yy in (-1.12,1.12):self.box((x+.3,y+yy,.016),(1.65,.018,.006),(235,183,38))
        self.label(self.r.task['label']+' · '+self.kind,(x+.6,y+.6,z+.55))
        k=self.kind
        if k=='factory':
            layout=getattr(self.r.path,'pick_place',None)
            if layout is None:raise ValueError('Factory environment requires pick/place layout')
            z=layout['table_height_m'];radius=layout['radius_m']
            self.caption.content='Front → RIGHT 90° (arm only) → front → REAR 180° (waist yaw). Object attachment is a visual preview; GUI object mass adds gravity while carrying. No contact/object inertia.'
            self.table(radius,0,z,.30,.26)
            self.label('PICK · TOP DOWN',(radius,.19,z+.12))
            self.label(f'Table: {z:.3f} m · waist pitch height',(radius,.18,z-.18))
            for key,title,color in [('right_basket_xyz','RIGHT 90° · ARM ONLY',(45,156,100)),('rear_basket_xyz','REAR 180° · WAIST YAW',(55,126,194))]:
                pos=np.array(layout[key]);self.table(pos[0],pos[1],z,.30,.28)
                self.tray(pos,color,(.25,.25,layout['basket_height_m']))
                self.label(title,pos+np.array([0,.17,.18]))
            size=(.045,.045,layout['object_height_m'])
            for index,color in enumerate([(235,158,49),(232,193,72)]):
                obj=self.box(layout['front_object_xyz'],size,color)
                self.anim.append(('pick_object',obj,index))
        elif k=='drawer':
            # Open cabinet shell and a separate moving drawer, front and handle.
            cx=x+.24;floor=z-.24
            self.table(cx,y,z-.52,.57,.54)
            self.box((cx,y,z-.12),(.5,.52,.035))
            self.box((cx,y,z-.5),(.5,.52,.035))
            self.box((cx+.24,y,z-.3),(.025,.52,.4))
            for yy in (-.26,.26):self.box((cx,y+yy,z-.3),(.5,.025,.4))
            for yy in (-.24,.24):self.box((cx,y+yy,floor),(.45,.012,.012),(190,200,210))
            pieces=[((cx,y,floor),(.43,.45,.02),(152,170,180)),
                    ((x,y,z-.09),(.025,.5,.25),(55,120,155)),
                    ((x-.035,y,z),(.025,.18,.022),(225,229,232))]
            for pos,size,col in pieces:self.anim.append(('drawer',self.box(pos,size,col),np.array(pos)))
        elif k=='door':
            pivot=np.array([x+.18,y+.36,0.]);self.pivot=pivot
            height=max(1.7,z+.4)
            for yy in (pivot[1],pivot[1]-.76):self.box((pivot[0],yy,height/2),(.07,.05,height),(50,62,74))
            self.box((pivot[0],pivot[1]-.38,height),(.07,.81,.06),(50,62,74))
            self.anim.append(('door',self.box(pivot+np.array([0,-.36,height/2]),(.035,.7,height-.06),(159,125,91)),np.array([0,-.36,height/2])))
            self.anim.append(('door',self.box(pivot+np.array([-.045,-.64,z]),(.05,.12,.025),(220,224,228)),np.array([-.045,-.64,z])))
        elif k in ('lifting','warehouse'):
            self.table(x+.08,y,z-.25,.85,.65)
            for i in range(3):self.box((x+.48,y+.48,z-.18+i*.17),(.26,.3,.16),(171,117,61))
            obj=self.box((x,y,z),(.25,.48,.22),(203,150,83));self.anim.append(('carried',obj,None))
            if k=='warehouse':
                for height in (.4,.8,1.2):self.box((x+.85,y,height),(.4,1.5,.045),(84,104,119))
                for yy in (-.75,.75):self.box((x+.85,y+yy,.7),(.055,.055,1.4),(227,157,36))
        elif k in ('wall','overhead'):
            if k=='wall':self.box((x+.04,y,z),(.025,.75,.7),(169,194,204));self.label('WORK SURFACE',(x+.05,y+.35,z+.38))
            else:
                self.box((x,y,z+.13),(.7,.85,.045),(139,158,175))
                for yy in (-.43,.43):self.box((x+.25,y+yy,z/2),(.05,.05,z),(78,91,103))
        elif k in ('valve','lever'):
            self.table(x+.12,y,z-.25)
            self.box((x+.12,y,z),(.08,.5,.42),(77,103,123))
            if k=='valve':
                mesh=trimesh.creation.annulus(r_min=.12,r_max=.15,height=.025)
                quat=tuple(np.roll(Rotation.from_euler('y',90,degrees=True).as_quat(),1))
                self.node('mesh_simple',vertices=mesh.vertices,faces=mesh.faces,position=(x,y,z),wxyz=quat,color=(200,60,48))
                self.box((x,y,z),(.025,.26,.02),(200,60,48));self.box((x,y,z),(.025,.02,.26),(200,60,48))
            else:self.box((x,y,z+.08),(.035,.035,.3),(215,178,65))
        elif k=='throwing':
            self.tray((x+.8,y,z-.4),(47,140,151),(.55,.65,.4))
            self.label('TARGET BIN',(x+.8,y,z+.05))
            obj=self.node('icosphere',radius=.05,position=self.anchor,color=(230,133,47));self.anim.append(('carried',obj,None))
        else:
            self.table(x+.08,y,z-.12)
            if k in ('mixing','pouring','jar'):
                self.cylinder((x,y,z-.02),.13,.14,(140,165,176))
                self.cylinder((x,y,z+.054),.112,.003,(61,134,167))
                if k=='pouring':self.cylinder((x,y+.25,z-.015),.07,.15,(217,204,177))
                if k=='jar':self.cylinder((x,y,z+.065),.13,.025,(193,82,57))
            elif k=='push_station':self.box((x+.05,y,z),(.17,.35,.22),(185,125,57))
            elif k=='hammer':
                self.box((x,y,z-.025),(.25,.3,.11),(134,89,47));self.cylinder((x,y,z+.05),.009,.07,(188,199,210))
            else:
                self.box((x,y,z-.055),(.3,.3,.06),(166,180,191))
                for yy in (-.075,.075):self.cylinder((x,y+yy,z-.01),.022,.035,(60,67,73))
                self.label({'assembly':'FASTENER FIXTURE','drilling':'DRILLING FIXTURE','wrench':'BOLT FIXTURE'}.get(k,'WORKPIECE'),(x+.15,y+.2,z+.1))
    def update(self):
        if self.task!=self.r.task_id:self.build()
        phase=(self.r.task_time/self.r.duration)%1
        travel=.5-.5*np.cos(2*np.pi*phase)
        for kind,h,origin in self.anim:
            if kind=='pick_object':
                layout=self.r.path.pick_place
                attach=layout['attach_phases'][origin];release=layout['release_phases'][origin]
                if phase<attach:
                    pos=np.asarray(layout['front_object_xyz']).copy()
                    # The second workpiece is supplied after the first lift.
                    if origin==1 and phase<layout['attach_phases'][0]:pos[1]+=.09
                elif phase<release:
                    pos=self.r.b.data.site_xpos[self.r.b.sid].copy()
                    pos[2]-=layout['grasp_offset_m']
                else:
                    key='right_basket_xyz' if origin==0 else 'rear_basket_xyz'
                    pos=np.asarray(layout[key])+np.array([0,0,.0075+layout['object_height_m']/2])
                h.position=pos
            elif kind=='drawer':h.position=origin+np.array([-.2*travel,0,0])
            elif kind=='door':
                rotation=Rotation.from_euler('z',-65*travel,degrees=True)
                h.position=self.pivot+rotation.apply(origin);h.wxyz=tuple(np.roll(rotation.as_quat(),1))
            elif kind=='carried':
                if self.r.state=='IDLE':
                    pos=self.anchor.copy()
                elif self.r.task_id in ('heavy_box_lift','package_carrying') and len(self.r.ee_sites)>1:
                    pos=np.mean([self.r.b.data.site_xpos[s] for s in self.r.ee_sites.values()],axis=0)
                else:pos=self.r.b.data.site_xpos[self.r.b.sid]
                h.position=np.asarray(pos)+np.array([0,0,-.055])
        for h in self.handles:h.visible=self.enabled.value
