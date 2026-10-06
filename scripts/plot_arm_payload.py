#!/usr/bin/env python3
"""Plot arm_payload logs: torque-time, angular velocity-time, torque-speed, EE velocity-time."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def find_log(path):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path
    if (path / 'samples.csv').is_file():
        return path / 'samples.csv'
    candidates = list(path.glob('*/samples.csv'))
    if not candidates:
        raise ValueError(f'No samples.csv under {path}')
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def read_log(path):
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if 'time_s' not in fields:
            raise ValueError('CSV has no time_s column')
        rows = list(reader)
    if not rows:
        raise ValueError('CSV contains no samples')
    if any(None in row or any(v is None or v == '' for v in row.values()) for row in rows):
        raise ValueError('Incomplete CSV row. Stop the recorder with Ctrl+C before plotting.')
    try:
        data = {k: np.array([r[k] for r in rows], dtype=str if k == 'phase' else float) for k in fields}
    except ValueError as exc:
        raise ValueError(f'Invalid numeric CSV value: {exc}') from exc
    if any(not np.all(np.isfinite(v)) for k,v in data.items() if k != 'phase'):
        raise ValueError('CSV contains non-finite numeric values')
    if np.any(np.diff(data['time_s']) <= 0):
        raise ValueError('time_s must be strictly increasing')
    # Exact suffix excludes requested/inverse torque columns.
    suffix = '_position_rad'
    joints = [k[:-len(suffix)] for k in fields if k.endswith(suffix)]
    joints = [j for j in joints if j+'_torque_Nm' in data and j+'_velocity_rad_s' in data]
    if not joints:
        raise ValueError('No joint torque/velocity columns found')
    metadata_path = path.with_name('metadata.json')
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    command_path = path.with_name('commands.jsonl')
    commands = [json.loads(line) for line in command_path.read_text().splitlines() if line.strip()] if command_path.exists() else []
    return data, joints, metadata, commands


def select_samples(data, commands=None, start=None, end=None):
    mask = np.ones(len(data['time_s']), dtype=bool)
    if commands:
        if 'command_id' not in data:
            raise ValueError('This log does not contain command_id')
        mask &= np.isin(data['command_id'], commands)
    if start is not None:
        mask &= data['time_s'] >= start
    if end is not None:
        mask &= data['time_s'] <= end
    indices = np.flatnonzero(mask)
    if len(indices) < 2:
        raise ValueError('Select at least two samples')
    # Keep disjoint selections as separate lines instead of bridging missing intervals.
    return np.split(indices, np.flatnonzero(np.diff(indices) > 1)+1)


def select_joints(joints, requested):
    if not requested:
        return joints
    result = []
    for token in requested:
        found = [j for j in joints if j == token or j.removeprefix('right_').removeprefix('left_').removesuffix('_joint') == token]
        if len(found) != 1:
            raise ValueError(f'Unknown or ambiguous joint {token}; choices: {", ".join(joints)}')
        if found[0] not in result:
            result.append(found[0])
    return result


def load_ratings(path, joints):
    if path is None or not Path(path).exists():
        return {}
    path = Path(path)
    if path.suffix.lower() in ('.yaml', '.yml'):
        import yaml
        config = yaml.safe_load(path.read_text())
        if not isinstance(config,dict) or config.get('reference') != 'joint_output':
            raise ValueError('YAML reference must be joint_output (after gearbox)')
    else:
        config = json.loads(path.read_text())
        if config.get('units') != 'joint_output_Nm':
            raise ValueError('Ratings units must be joint_output_Nm (after gearbox)')
    result = {}
    for joint in joints:
        entry = config.get('joints', {}).get(joint, {})
        values = {}
        for key,alias in [('rated_torque_Nm','rated_torque_nm'),('peak_torque_Nm','peak_torque_nm'),
                          ('rated_velocity_rad_s','rated_velocity_rad_s'),('no_load_velocity_rad_s','no_load_velocity_rad_s')]:
            v = entry.get(key,entry.get(alias))
            if v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0):
                raise ValueError(f'{joint}: {key} must be positive or null')
            values[key] = v
        rated,peak = values['rated_torque_Nm'],values['peak_torque_Nm']
        if rated is not None and peak is not None and rated > peak:
            raise ValueError(f'{joint}: rated torque exceeds peak torque')
        result[joint] = values
    return result


def motor_envelope(speed, entry):
    """Visualization approximation specified in motor_ratings.yaml, not measured curves."""
    peak,free = entry.get('peak_torque_Nm'),entry.get('no_load_velocity_rad_s')
    if peak is None or free is None:
        return None,None
    peak_curve = peak*np.maximum(0.,1.-np.abs(np.asarray(speed))/free)
    rated = entry.get('rated_torque_Nm')
    return peak_curve, None if rated is None else np.minimum(rated,peak_curve)


def outside_envelopes(speed, torque, entry):
    speed,torque=np.abs(speed),np.abs(torque)
    peak,rated=motor_envelope(speed,entry)
    no=np.zeros_like(speed,dtype=bool)
    if peak is None:return no,no
    overspeed=speed>entry['no_load_velocity_rad_s']
    outside_peak=(torque>peak+1e-9)|overspeed
    outside_rated=no if rated is None else ((torque>rated+1e-9)|overspeed)&~outside_peak
    return outside_rated,outside_peak


def measured_summary(data, joint, segments):
    """RMS over recorded time only; never integrate across excluded gaps."""
    t=data['time_s']
    indices=np.concatenate(segments)
    duration=sum(t[s[-1]]-t[s[0]] for s in segments)
    def stats(key):
        values=data[key]
        energy=sum(np.sum(np.diff(t[s])*(values[s[:-1]]**2+values[s[1:]]**2)/2) for s in segments)
        rms=np.sqrt(energy/duration) if duration>0 else np.sqrt(np.mean(values[indices]**2))
        return float(rms),float(np.max(np.abs(values[indices])))
    return (*stats(joint+'_torque_Nm'),*stats(joint+'_velocity_rad_s'))


def draw(data, joints, meta, commands, segments, output, show=False, dpi=160, ratings=None):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ratings = ratings or {}
    output.mkdir(parents=True, exist_ok=True)
    t = data['time_s']
    indices = np.concatenate(segments)
    limits = dict(zip(meta.get('joint_names', []), meta.get('torque_limits_Nm', [])))
    palette = ['#dbeafe', '#dcfce7', '#ffedd5', '#f3e8ff']
    payload = meta.get('payload_kg', '?')
    model_label = f"Model variant {meta['model_variant']}" if 'model_variant' in meta else f'Payload {payload} kg'
    subtitle = f'{model_label} | joint-side values | {t[indices[0]]:.2f}–{t[indices[-1]]:.2f} s'
    figures = []

    def axes_grid(title):
        ncols = 2 if len(joints) > 1 else 1
        nrows = (len(joints)+ncols-1)//ncols
        fig, axes = plt.subplots(nrows,ncols,figsize=(13 if ncols==2 else 9,2.9*nrows+1),squeeze=False,layout='constrained')
        fig.suptitle(title+'\n'+subtitle,fontsize=13)
        for ax in axes.flat[len(joints):]:
            ax.set_visible(False)
        for ax,j in zip(axes.flat,joints):
            ax.set_title(j.replace('_joint','').replace('_',' '),fontsize=10)
            ax.grid(alpha=.2)
        figures.append(fig)
        return fig,list(axes.flat[:len(joints)])

    def annotate(ax):
        for i,cmd in enumerate(commands):
            start = float(cmd['start_time_s'])
            end = start+float(cmd['seconds'])
            if end < t[indices[0]] or start > t[indices[-1]]:
                continue
            label = str(cmd.get('label', '')).split(' · ')[0]
            if not label.isascii(): label = ''
            ax.axvspan(start,end,color=palette[i%len(palette)],alpha=.5,zorder=0)
            ax.axvline(start,color='#6b7280',lw=.6,ls=':')
            ax.text(max(start,t[indices[0]]),.98,f'#{cmd["command_id"]} {label}',transform=ax.get_xaxis_transform(),va='top',fontsize=7)
        ax.set_xlim(t[indices[0]],t[indices[-1]])
        ax.set_xlabel('Time (s)')

    def series(ax, key, label, **style):
        if key not in data:
            return
        for i,segment in enumerate(segments):
            ax.plot(t[segment],data[key][segment],label=label if i==0 else None,**style)

    def torque_lines(ax, joint):
        entry = ratings.get(joint,{})
        for field,label,color,style in [('rated_torque_Nm','Rated','#d97706','--'),
                                       ('peak_torque_Nm','Peak','#dc2626','-.')]:
            value = entry.get(field)
            if value is not None:
                ax.axhline(value,color=color,ls=style,lw=1.2,label=f'{label}: ±{value:g} N·m')
                ax.axhline(-value,color=color,ls=style,lw=1.2)
        if joint in limits:
            for k,value in enumerate(limits[joint]):
                ax.axhline(value,color='#64748b',ls=':',lw=.8,label='XML control limits' if k==0 else None)
        missing = [label for field,label in [('rated_torque_Nm','Rated'),('peak_torque_Nm','Peak')]
                   if entry.get(field) is None]
        if missing:
            ax.text(.02,.04,' / '.join(missing)+': not specified',transform=ax.transAxes,fontsize=7,color='#92400e')

    fig,axes = axes_grid('Joint torque vs time')
    for ax,j in zip(axes,joints):
        annotate(ax)
        series(ax,j+'_torque_Nm','Applied',color='#2563eb',lw=1.2)
        series(ax,j+'_inverse_torque_Nm','Inverse dynamics',color='#f97316',lw=.9,ls='--')
        torque_lines(ax,j)
        ax.set_ylabel('Torque (N·m)')
        ax.legend(loc='lower right',fontsize=7)
    fig.savefig(output/'torque_time.png',dpi=dpi)

    fig,axes = axes_grid('Joint angular velocity vs time (v–t)')
    for ax,j in zip(axes,joints):
        annotate(ax)
        series(ax,j+'_velocity_rad_s','Actual',color='#059669',lw=1.2)
        series(ax,j+'_reference_velocity_rad_s','Reference',color='#6b7280',lw=.8,ls='--')
        ax.set_ylabel('Angular velocity (rad/s)')
        ax.legend(loc='lower right',fontsize=7)
    fig.savefig(output/'velocity_time.png',dpi=dpi)

    def draw_tv(ax,j):
        entry=ratings.get(j,{})
        speed=np.abs(data[j+'_velocity_rad_s'][indices])
        torque=np.abs(data[j+'_torque_Nm'][indices])
        free=entry.get('no_load_velocity_rad_s')
        if free is not None:
            grid=np.linspace(0,free,500)
            rated_value,peak_value=entry.get('rated_torque_Nm'),entry.get('peak_torque_Nm')
            if rated_value is not None and peak_value is not None:
                grid=np.sort(np.append(grid,free*(1-rated_value/peak_value)))
            peak,rated=motor_envelope(grid,entry)
            if peak is not None:ax.plot(grid,peak,color='tab:blue',lw=1.8,label='Peak T-V curve')
            if rated is not None:ax.plot(grid,rated,color='tab:orange',lw=1.8,label='Rated T-V curve')
        if free is None or entry.get('peak_torque_Nm') is None:
            ax.text(.02,.95,'T-V curve unavailable: missing peak torque or no-load speed',
                    transform=ax.transAxes,va='top',fontsize=7)
        ax.scatter(speed,torque,color='tab:blue',s=7,alpha=.5,label='Measured data',rasterized=True)
        out_rated,out_peak=outside_envelopes(speed,torque,entry)
        ax.scatter(speed[out_rated],torque[out_rated],color='tab:orange',marker='x',s=20,lw=.8,label='Outside rated curve')
        ax.scatter(speed[out_peak],torque[out_peak],color='red',marker='x',s=20,lw=.8,label='Outside peak curve')
        ax.set(xlabel='Velocity |v| [rad/s]',ylabel='Torque |T| [Nm]',xlim=(0,max(float(speed.max()),free or 0,.1)*1.07),ylim=(0,None))
        ax.grid(alpha=.2)
        ax.legend(loc='upper right',fontsize=7)

    fig,axes = axes_grid('Joint T-V curves (absolute velocity and torque)')
    for ax,j in zip(axes,joints):draw_tv(ax,j)
    fig.savefig(output/'torque_velocity.png',dpi=dpi)
    fig.savefig(output/'v_t.png',dpi=dpi)
    for j in joints:
        single,ax=plt.subplots(figsize=(10,7.5),layout='constrained')
        ax.set_title(j+' T-V curve')
        draw_tv(ax,j)
        single.savefig(output/(j+'_t_v.png'),dpi=dpi)
        plt.close(single)

    xyz_keys = ['ee_world_'+axis+'_m' for axis in 'xyz']
    if all(k in data for k in xyz_keys):
        # Differentiate the original full recording before filtering to avoid gap artifacts.
        xyz = np.column_stack([data[k] for k in xyz_keys])
        velocity = np.gradient(xyz,t,axis=0,edge_order=2 if len(t)>2 else 1)
        speed = np.linalg.norm(velocity,axis=1)
        fig,ax = plt.subplots(figsize=(12,4),layout='constrained')
        figures.append(fig)
        fig.suptitle('End-effector world velocity vs time (finite difference)\n'+subtitle,fontsize=12)
        annotate(ax)
        for part,values,color in zip(['vx','vy','vz','speed'],[*velocity.T,speed],['#2563eb','#f97316','#059669','#111827']):
            for i,segment in enumerate(segments):
                ax.plot(t[segment],values[segment],color=color,label=part if i==0 else None,lw=1)
        ax.set_ylabel('Velocity / speed (m/s)')
        ax.grid(alpha=.2)
        ax.legend()
        fig.savefig(output/'ee_velocity_time.png',dpi=dpi)

    table_rows = []
    with (output/'summary.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['joint','rated_torque_Nm','peak_torque_Nm','rated_velocity_rad_s','peak_velocity_rad_s',
                         'motor_rated_torque_Nm','motor_peak_torque_Nm','motor_rated_velocity_rad_s','motor_no_load_velocity_rad_s'])
        for j in joints:
            values=measured_summary(data,j,segments)
            entry=ratings.get(j,{})
            specs=[entry.get(k) for k in ('rated_torque_Nm','peak_torque_Nm','rated_velocity_rad_s','no_load_velocity_rad_s')]
            writer.writerow([j,*values,*specs])
            for source,row_values in [('Measured (RMS / max)',values),('Motor (YAML)',specs)]:
                table_rows.append([j,source,*['N/A' if v is None else f'{v:.2f}' for v in row_values]])
    headers=['Joint','Source','Rated / RMS torque\nN·m','Peak torque\nN·m','Rated / RMS velocity\nrad/s','Peak / no-load velocity*\nrad/s']
    md_headers = [h.replace('\n',' ').replace('|', r'\|') for h in headers]
    markdown = '| '+' | '.join(md_headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'
    markdown += ''.join('| '+' | '.join(row)+' |\n' for row in table_rows)
    markdown += '\nMeasured: time-weighted RMS / maximum absolute value. Motor: supplied YAML ratings. *Motor last column is no-load speed, not a verified peak velocity rating. N/A = missing specification.\n'
    (output/'torque_table.md').write_text(markdown)
    print(markdown,flush=True)
    fig,ax = plt.subplots(figsize=(18,max(4,.85*len(joints)+2.5)),layout='constrained')
    figures.append(fig)
    ax.axis('off')
    ax.set_title('Measured values and motor specifications\n'+subtitle,pad=18)
    table = ax.table(cellText=table_rows,colLabels=headers,cellLoc='center',loc='center',
                     colWidths=[.27,.17,.14,.12,.14,.16])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1,2.2)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#e2e8f0')
        if row==0:
            cell.set_facecolor('#1e293b');cell.set_text_props(color='white',weight='bold')
        elif ((row-1)//2)%2==0: cell.set_facecolor('#f1f5f9')
    fig.text(.5,.02,'Measured: RMS / absolute maximum. Motor: YAML specifications. *Motor last column = no-load speed. N/A = missing.',
             ha='center',fontsize=9,color='#92400e')
    fig.savefig(output/'torque_table.png',dpi=dpi)

    if show:
        plt.show()
    for fig in figures:
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log',nargs='?',type=Path,default=Path(__file__).resolve().parents[1]/'payload_logs',help='Log folder, samples.csv, or logs root (newest selected)')
    p.add_argument('--ratings',type=Path,help='Motor ratings YAML (or legacy JSON); default configs/motor_ratings.yaml')
    p.add_argument('--output',type=Path,help='Default: selected log folder/plots')
    p.add_argument('--joint',nargs='+',help='Full joint name or short name, e.g. elbow_pitch')
    p.add_argument('--command',type=int,nargs='+',help='Only selected command IDs, e.g. 1 2 3')
    p.add_argument('--start',type=float,help='Start time (s) in recording')
    p.add_argument('--end',type=float,help='End time (s) in recording')
    p.add_argument('--show',action='store_true',help='Also open plot windows; close them to exit')
    p.add_argument('--dpi',type=int,default=160)
    args = p.parse_args()
    if args.dpi<=0 or any(x is not None and not np.isfinite(x) for x in (args.start,args.end)):
        p.error('dpi must be positive and time limits finite')
    if args.start is not None and args.end is not None and args.end<=args.start:
        p.error('--end must be greater than --start')
    try:
        path=find_log(args.log)
        data,joints,meta,commands=read_log(path)
        joints=select_joints(joints,args.joint)
        segments=select_samples(data,args.command,args.start,args.end)
        if args.ratings is not None:
            rating_path = args.ratings
        elif 'robot_profile' in meta:
            rating_path = Path(meta['motor_ratings']) if meta.get('motor_ratings') else None
        else:
            rating_path = Path(__file__).resolve().parents[1]/'configs/motor_ratings.yaml'
        if args.ratings is not None and not rating_path.is_file():
            raise ValueError(f'Ratings file not found: {rating_path}')
        ratings = load_ratings(rating_path,joints)
        output=args.output or path.parent/'plots'
        if meta.get('mode')=='slide':
            print('NOTE: slide mode torque/velocity are zero placeholders, not load measurements.')
        print(f'Ratings: {rating_path}\nInput: {path}\nJoints: {", ".join(joints)}\nOutput: {output.resolve()}',flush=True)
        draw(data,joints,meta,commands,segments,output,args.show,args.dpi,ratings)
        print('Saved PNG plots and summary.csv',flush=True)
    except (ValueError,OSError,KeyError) as exc:
        p.error(str(exc))
    except ImportError as exc:
        p.error(f'{exc}. Install plotting dependencies: python -m pip install numpy matplotlib pyyaml')


if __name__=='__main__':
    main()
