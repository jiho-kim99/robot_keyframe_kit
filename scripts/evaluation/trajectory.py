"""Fixed geometric joint paths with analytic quintic time scaling."""
import hashlib
import numpy as np

class TimedPath:
    def __init__(self, points, phases=None):
        self.points=np.asarray(points,dtype=float)
        if self.points.ndim!=2 or len(self.points)<2 or not np.isfinite(self.points).all():
            raise ValueError('Path needs at least two finite joint vectors')
        self.phases=np.linspace(0,1,len(self.points)) if phases is None else np.asarray(phases,dtype=float)
        if self.phases.shape!=(len(self.points),) or not np.isfinite(self.phases).all() or self.phases[0]!=0 or self.phases[-1]!=1 or np.any(np.diff(self.phases)<=0):
            raise ValueError('Path phases must strictly increase from 0 to 1')
        self.fingerprint=hashlib.sha256(self.points.tobytes()+self.phases.tobytes()).hexdigest()
    def at(self, elapsed, duration, repeat=True):
        if not np.isfinite(duration) or duration<=0:raise ValueError('motion_duration must be finite and positive')
        phase=(max(0.,elapsed)/duration)%1 if repeat else np.clip(elapsed/duration,0,1)
        i=min(np.searchsorted(self.phases,phase,side='right')-1,len(self.points)-2)
        width=self.phases[i+1]-self.phases[i]
        u=np.clip((phase-self.phases[i])/width,0,1)
        delta=self.points[i+1]-self.points[i]
        segment_time=duration*width
        q=self.points[i]+delta*(10*u**3-15*u**4+6*u**5)
        v=delta*(30*u**2-60*u**3+30*u**4)/segment_time
        a=delta*(60*u-180*u**2+120*u**3)/segment_time**2
        return q,v,a
