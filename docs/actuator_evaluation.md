# Upper-body actuator task evaluation

Run in the existing robot_keyframe_kit environment:

```bash
cd /home/jiho/robot_keyframe_kit
python scripts/evaluate_tasks.py --variant 15kg
```

Open http://127.0.0.1:8766. Stop another process occupying that port or use `--port 8770`. Both arms and waist are active; legs are fixed, hidden, and excluded from logs. The original payload and manipulation-platform launchers remain available.

## GUI workflow

1. Choose one of 18 entries in **Task**.
2. Enter **Motion Duration / Cycle Time [s]** numerically (0.5, 1, 2, 5, 10, etc.). Press **Start / Resume Task** to apply it.
3. Use **Pause**, **Stop**, or **Reset Task**. Reset smoothly moves to the initial task pose and pauses. Selecting a different task while running prepares and starts the new task.
4. Press **Start Logging** only when needed, then **Stop Logging**. Running does not automatically enable logging. Logging ON/OFF and its path are displayed.

Changing task, applied duration or gains closes an active recording with the reason in metadata. It does not silently start a new log. Pausing keeps logging available but labels records PAUSED. Changing task while paused loads it idle; press Start to execute. A separate smooth preparation move (default 1 s; `--transition-duration`) brings the robot from its current configuration to the task path. Preparation is not part of motion_duration or a cycle. Resume uses the same preparation mechanism to avoid jumping back to the paused reference.

## Timing contract

A full closed task path is one cycle, e.g. A → B → A. Motion duration T scales the fixed normalized path. Piecewise quintic interpolation gives zero reference velocity and acceleration at its keyframes, including cycle boundaries. Segment phase fractions and joint points do not change with T:

- q(t) = Q(t/T)
- qdot(t) = Q'(t/T) / T
- qddot(t) = Q''(t/T) / T²

All 18 default tasks repeat until Pause, Stop, task selection or process exit. There are no settling waits that extend the cycle. Actual tracking error is recorded; if an aggressive duration saturates torque, the actual robot may not follow the requested path. Reference timing remains unchanged. Duration must be at least two simulation timesteps. A timestep of 0.002 s is the default. No waypoint decimation is used.

RMS/peak statistics use recorded RUNNING samples by default, excluding PREPARING, IDLE and PAUSED. Recording may begin/end mid-cycle; analysis reports its sample count and time coverage rather than assuming complete cycles. Use `--include-transitions` when intentionally evaluating the approach or paused hold.

## Task configuration

`configs/evaluation_tasks.yaml` contains 18 individually configurable entries:

Pick & place, heavy box lift, package carrying, push/pull, door opening, drawer, screwdriver, valve turning, jar opening, drilling, wrench turning, lever operation, wiping/polishing, stirring, pouring, hammering, throwing, overhead work.

These are **representative joint-space trajectories**, permitted by the evaluation specification. In particular, door/lever arcs and wiping/stirring loops are joint-space surrogates, not exact Cartesian hinge/circle constraints. Tools, object release, grasp and impact contact are not simulated. Hammering measures acceleration/load from the prescribed strike-return motion, not a physical impact impulse. Jar opening holds the other arm at its reference posture.

- `keyframes_deg`: joint-role offsets from `home_deg` in degrees, defining geometric amplitude and shape.
- `phases`: optional fixed normalized knot times, 0 through 1. Default: evenly spaced.
- `motion_duration`: default seconds per cycle for this task.
- `reference_duration`: nominal duration used to report speed scale = reference_duration / motion_duration.
- `repeat`: true by default; false stops after one path traversal.
- `load`: prescribed wrench parameters described below.

Robot-specific role names are mapped under `task_joint_roles` in `configs/manipulation_platform.yaml`. No robot joint names are hardcoded in the task engine. Required roles must be among active joints; unsupported tasks fail explicitly instead of silently dropping required joints. `--robot`, `--variant`, `--ee`, `--groups`, `--config` and `--tasks` allow other configurations. The default 10/15/20kg models continue to mean total right wrist link mass, not an added sphere.

## External loads

External forces and moments are applied at the configured EE site using MuJoCo's generalized-force mapping, not direct state edits. They are logged separately from actuator torque. Inverse dynamics feedforward compensates the prescribed load; actual actuator torque still respects configured limits.

Load parameters:

- `sites`: endpoint IDs, default selected EE. Heavy box/carrying use both endpoints.
- `force_world_N`, `torque_world_Nm`: world-frame constant vectors.
- `axial_force_N`: force opposing the tool's local +X axis.
- `reaction_torque_Nm`: constant reaction moment opposing tool +X, also present at zero speed (drilling).
- `resistance_force_N`, `resistance_torque_Nm`: opposing reference EE linear/angular motion with a smooth reversal near zero speed.
- `object_mass_kg`: optional **static weight only**, distributed evenly between listed EE sites. No object rotational inertia or grasp is modeled. Default is 0 to avoid adding a hidden load to the selected heavy-link model.

Defaults for each task are visible in YAML and saved in metadata. Nonnegative scalar loads are required. Rendering and physics use existing fixed-base, contact-disabled models.

## Logging

`task_logs/<task>/duration_<seconds>s/<unique UTC timestamp>/`

Each session has `samples.csv` and `metadata.json`. Samples include task/state/phase, simulation and task time, all active joint positions, actual velocities/accelerations, commanded positions/velocities/accelerations, requested and applied actuator torques, inverse dynamics torque, external generalized loads, signed mechanical power (tau × qdot), tracking errors and saturation flags. Available EE sites include position, quaternion, world linear/angular velocity and applied wrench.

Metadata includes motion_duration, cycle_time, trajectory_frequency_hz, trajectory_speed_scale, task parameters, path points/phases/hash, robot/profile/model, body masses, payload_mass_kg (configured COM body's total mass), separate object_mass_kg, external loads, gains, timestep and actuator groups. Each recording contains one task/duration configuration. `samples.csv` retains actual and requested torque to distinguish physical load requirements from actuator saturation.

## Analysis

```bash
python scripts/plot_task_log.py --log "task_logs/pick_and_place/duration_2s/<run>"
python scripts/compare_task_logs.py \
  "task_logs/pick_and_place/duration_1s/<run>" \
  "task_logs/pick_and_place/duration_2s/<run>" \
  --output task_comparison
```

Single-log PNGs: position, velocity, acceleration, torque, signed power vs time and absolute torque–speed scatter, with robot motor-rating curves when available. Titles contain task, duration and model variant. `summary.csv` includes peak/RMS torque, peak/RMS velocity, peak acceleration, peak/RMS mechanical power, peak requested torque, tracking error and saturation fraction. Peak values use absolute magnitude; RMS is time-weighted trapezoidal integration and never bridges excluded time gaps.

Comparison creates joint-specific metric charts, `comparison.csv`, `comparison_metadata.json` and `worst_cases.csv`. Worst cases include task, motion duration and source run. Comparisons require matching robot and joint scope. A different geometric-path hash for the same task produces a warning: it is not a pure time-scaling experiment. Review model variants and external-load parameters as well when interpreting duration comparisons.

## Headless checks

```bash
python scripts/evaluate_tasks.py --check-all
python scripts/evaluate_tasks.py --task pick_and_place --motion-duration 2 \
  --headless --autostart --log --run-seconds 9
```

`--log` is the explicit headless logging request. `--autostart` starts preparation and repeating motion. Without these flags the application does not implicitly run or record. `--run-seconds` includes preparation time.

Implementation: `scripts/evaluation/{trajectory,tasks,runtime,logger,analysis}.py`, `scripts/evaluate_tasks.py`, plotting/comparison entry points, and the task YAML. Existing manually edited A/B/C sessions are separate from this representative-task engine.

## Task workcell visualization

Task selection automatically replaces the visual workcell. Pick & place uses a factory conveyor, infeed/place trays and cartons; Drawer uses a cabinet with moving drawer/handle; Door uses a hinged door; other tasks include lifting/carrying workstations, fixtures, valve/lever, wall, mixing/pouring containers, hammer workpiece, target bin and overhead panel. Drawer/door motion follows the configured cycle phase. Carried object visuals follow the EE. These are schematic visual animations, not grasp/contact-derived object dynamics.

`Task environment > Show environment` toggles visibility. Models are positioned from the task's initial reference EE pose. An optional `environment: {offset_m: [x, y, z]}` entry in a task's YAML shifts the workcell anchor. The visual geometry adds no mass, collision or friction to MuJoCo. Logged `environment_mode` is explicitly visual_only. Wrenches remain those specified by the task configuration.
