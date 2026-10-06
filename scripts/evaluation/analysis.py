"""Task/duration-aware metrics, plotting and worst-case comparison."""
import csv
import json
from pathlib import Path
import numpy as np

METRICS=('peak_torque_Nm','rms_torque_Nm','peak_velocity_rad_s','rms_velocity_rad_s',
         'peak_acceleration_rad_s2','peak_mechanical_power_W','rms_mechanical_power_W')

def read_run(path,include_transitions=False):
    path=Path(path).expanduser().resolve()
    file=path if path.is_file() else path/'samples.csv'
    meta=json.loads(file.with_name('metadata.json').read_text())
    with file.open() as f:
        reader=csv.DictReader(f);rows=list(reader);fields=reader.fieldnames
    if len(rows)<2:raise ValueError(f'{file}: at least two samples required')
    if any(None in row or any(x is None or x=='' for x in row.values()) for row in rows):raise ValueError('Incomplete recording; stop logging before plotting')
    d={k:np.array([r[k] for r in rows],dtype=str if k in ('task_name','phase') else float) for k in fields}
    if any(not np.isfinite(v).all() for k,v in d.items() if k not in ('task_name','phase')):raise ValueError('Non-finite log')
    if np.any(np.diff(d['time_s'])<=0):raise ValueError('Time must strictly increase')
    if not np.all(d['task_name']==meta['task_name']):raise ValueError('Mixed tasks in one recording')
    if not np.allclose(d['motion_duration_s'],meta['motion_duration']):raise ValueError('Mixed durations in one recording')
    idx=np.arange(len(rows)) if include_transitions else np.flatnonzero(d['phase']=='RUNNING')
    if len(idx)<2:raise ValueError('At least two RUNNING samples required; use --include-transitions for preparation/paused data')
    segments=np.split(idx,np.flatnonzero(np.diff(idx)>1)+1)
    return file.parent,meta,d,segments

def label(meta):
    return f"{meta.get('task_label',meta['task_name'])} — {meta['motion_duration']:g} s [{meta.get('model_variant','?')}]"

def summaries(meta,d,segments):
    idx=np.concatenate(segments);t=d['time_s']
    elapsed=sum(t[s[-1]]-t[s[0]] for s in segments)
    def stats(key):
        y=d[key]
        integral=sum(np.sum(np.diff(t[s])*(y[s[:-1]]**2+y[s[1:]]**2)/2) for s in segments)
        rms=np.sqrt(integral/elapsed) if elapsed>0 else np.sqrt(np.mean(y[idx]**2))
        return float(np.max(np.abs(y[idx]))),float(rms)
    result=[]
    for joint in meta['joint_names']:
        pt,rt=stats(joint+'_torque_Nm');pv,rv=stats(joint+'_velocity_rad_s');pa,_=stats(joint+'_acceleration_rad_s2');pp,rp=stats(joint+'_power_W')
        result.append(dict(task_name=meta['task_name'],motion_duration=meta['motion_duration'],cycle_time=meta['cycle_time'],
            joint=joint,group=meta.get('joint_groups',{}).get(joint,''),
            **dict(zip(METRICS,[pt,rt,pv,rv,pa,pp,rp])),
            peak_requested_torque_Nm=stats(joint+'_torque_requested_Nm')[0],
            peak_tracking_error_rad=stats(joint+'_position_error_rad')[0],
            saturation_sample_fraction=float(np.mean(d[joint+'_saturated'][idx])),
            analyzed_samples=len(idx),analyzed_time_s=float(elapsed)))
    return result

def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def plot_run(path,output=None,include_transitions=False,show=False):
    import matplotlib
    if not show:matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_arm_payload import load_ratings,motor_envelope,outside_envelopes
    folder,meta,d,segments=read_run(path,include_transitions)
    output=Path(output) if output else folder/'plots';output.mkdir(parents=True,exist_ok=True)
    metrics=summaries(meta,d,segments);write_csv(output/'summary.csv',metrics)
    (output/'analysis_metadata.json').write_text(json.dumps(dict(source=str(folder),label=label(meta),include_transitions=include_transitions,
        metric_scope='All selected recorded RUNNING samples; may include partial cycles' if not include_transitions else 'All recorded states',
        path_hash=meta['path_hash'],motion_duration=meta['motion_duration']),indent=2)+'\n')
    ratings=load_ratings(meta.get('motor_ratings'),meta['joint_names'])
    n=len(meta['joint_names']);figures=[]
    for field,title,unit in [('position_rad','Position','rad'),('velocity_rad_s','Velocity','rad/s'),
        ('acceleration_rad_s2','Acceleration','rad/s²'),('torque_Nm','Torque','N·m'),('power_W','Mechanical power','W')]:
        fig,axes=plt.subplots((n+1)//2,2,figsize=(14,2.6*((n+1)//2)),squeeze=False,layout='constrained')
        fig.suptitle(label(meta)+' · '+title)
        for ax,j in zip(axes.flat,meta['joint_names']):
            for segment in segments:ax.plot(d['time_s'][segment],d[j+'_'+field][segment],lw=.8)
            ax.set(title=j,xlabel='Time [s]',ylabel=unit);ax.grid(alpha=.2)
        for ax in axes.flat[n:]:ax.set_visible(False)
        fig.savefig(output/(title.lower().replace(' ','_')+'_time.png'),dpi=130);figures.append(fig)
    fig,axes=plt.subplots((n+1)//2,2,figsize=(14,3*((n+1)//2)),squeeze=False,layout='constrained')
    fig.suptitle(label(meta)+' · Torque–speed operating points')
    idx=np.concatenate(segments)
    for ax,j in zip(axes.flat,meta['joint_names']):
        v=np.abs(d[j+'_velocity_rad_s'][idx]);tau=np.abs(d[j+'_torque_Nm'][idx]);r=ratings.get(j,{})
        grid=np.linspace(0,r.get('no_load_velocity_rad_s') or max(v.max(),1),300)
        peak,rated=motor_envelope(grid,r)
        if peak is not None:ax.plot(grid,peak,label='Peak',color='tab:blue')
        if rated is not None:ax.plot(grid,rated,label='Rated',color='tab:orange')
        ax.scatter(v,tau,s=3,alpha=.4,label='Measured')
        outr,outp=outside_envelopes(v,tau,r)
        ax.scatter(v[outr],tau[outr],s=8,marker='x',color='tab:orange');ax.scatter(v[outp],tau[outp],s=8,marker='x',color='red')
        ax.set(title=j,xlabel='|velocity| [rad/s]',ylabel='|torque| [N·m]',xlim=(0,None),ylim=(0,None));ax.grid(alpha=.2);ax.legend(fontsize=7)
    for ax in axes.flat[n:]:ax.set_visible(False)
    fig.savefig(output/'torque_speed.png',dpi=130);figures.append(fig)
    print(label(meta))
    for row in metrics:print(row['joint'], ' | '.join(f'{k}={row[k]:.3f}' for k in METRICS))
    if show:plt.show()
    for f in figures:plt.close(f)
    return metrics

def compare_runs(paths,output,include_transitions=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    runs=[];rows=[]
    for path in paths:
        folder,meta,data,segments=read_run(path,include_transitions)
        summary=summaries(meta,data,segments)
        run_label=label(meta)+' · '+folder.name
        runs.append((folder,meta,summary,run_label))
        rows.extend(dict(run=str(folder),label=run_label,**r) for r in summary)
    if len(runs)<2:raise ValueError('Comparison requires at least two logs')
    signatures={(m.get('robot_profile'),tuple(m['joint_names'])) for _,m,_,_ in runs}
    if len(signatures)!=1:raise ValueError('Compare logs with the same robot and joint scope')
    write_csv(output/'comparison.csv',rows)
    worst=[]
    for j in runs[0][1]['joint_names']:
        subset=[row for row in rows if row['joint']==j]
        fig,axes=plt.subplots(2,4,figsize=(18,9),layout='constrained')
        fig.suptitle(j+' · Task / motion duration comparison')
        ticklabels=[f"{r['task_name']}\n{r['motion_duration']:g}s\n#{i+1}" for i,r in enumerate(subset)]
        for ax,metric in zip(axes.flat,METRICS):
            ax.bar(np.arange(len(subset)),[r[metric] for r in subset]);ax.set_title(metric)
            ax.set_xticks(np.arange(len(subset)),ticklabels,rotation=35,ha='right',fontsize=7);ax.grid(axis='y',alpha=.2)
            best=max(subset,key=lambda row:row[metric])
            worst.append(dict(joint=j,metric=metric,value=best[metric],task=best['task_name'],motion_duration=best['motion_duration'],run=best['run']))
        axes.flat[-1].set_visible(False);fig.savefig(output/(j+'_comparison.png'),dpi=130);plt.close(fig)
    write_csv(output/'worst_cases.csv',worst)
    warnings=[]
    for task in {m['task_name'] for _,m,_,_ in runs}:
        hashes={m['path_hash'] for _,m,_,_ in runs if m['task_name']==task}
        if len(hashes)>1:warnings.append(f'{task}: geometric paths differ; not a pure duration-only comparison')
    (output/'comparison_metadata.json').write_text(json.dumps(dict(runs=[dict(path=str(p),label=l,path_hash=m['path_hash'],
        motion_duration=m['motion_duration'],model_variant=m.get('model_variant'),parameters=m['task_parameters']) for p,m,_,l in runs],warnings=warnings,include_transitions=include_transitions),indent=2)+'\n')
    for warning in warnings:print('NOTE:',warning)
    print('Saved comparison and worst_cases.csv to',output.resolve())
