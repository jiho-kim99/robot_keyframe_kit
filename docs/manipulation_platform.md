# Configurable manipulation measurements

Run from `/home/jiho/robot_keyframe_kit` in the existing `robot_keyframe_kit` environment:

```bash
python scripts/manipulation_platform.py --variant 15kg
```

Open http://127.0.0.1:8766. Stop an old process using that port first, or add `--port 8770`. The original `scripts/arm_payload.py` entry point continues to work.

## What is included

- `payload_lift`: original saved EE/joint pose workflow with robot-specific A/B/C presets and base/10kg/15kg/20kg model variants. The heavy variants change the right wrist link's total mass; they do not attach an additional object.
- `waypoint_motion`: generic free-space manipulation between saved poses. Same timing, dwell, Start/Stop, torque/velocity recording, and completion logic.
- Default scope: both arms (14 joints) + waist yaw/pitch (2 joints). Legs and base are fixed at their XML reference pose; leg geometry is hidden and leg joints are excluded from control and logs. This is a fixed-base experiment, not a balance/contact simulation.
- EE editing solves IK using all active joints, including waist. The inactive arm holds its saved joint target. This is a single selected EE editor, not simultaneous dual-EE constrained IK.
- Right EE uses the existing welding site. The source robot has no left welding site: the left endpoint is explicitly at the left wrist link origin. Configure its `body`/`position_m` to match a real left tool frame.

## Change experiments

Use the **Robot / domain** folder: select robot, domain, variant, and EE; press **Apply robot / domain (reset pose)**. The next setup is validated before switching. Save poses first; switching resets the simulation and the browser reconnects to the same port. Switching is disabled while recording.

CLI equivalents:

```bash
python scripts/manipulation_platform.py --domain payload_lift --variant 20kg
python scripts/manipulation_platform.py --domain waypoint_motion --variant base --ee left
python scripts/manipulation_platform.py --groups waist right_arm --variant 10kg
python scripts/manipulation_platform.py --check --variant 15kg
python scripts/manipulation_platform.py --list
```

`--groups` overrides the profile's default active groups. Each joint must occur only once. Excluded joints cannot be enabled. Gains retain Kp 0–1000 and Kd 0–200. Without `--variant`, the original base model is used.

## Add a robot without editing controller code

Edit `configs/manipulation_platform.yaml` or pass `--config /path/to/catalog.yaml`. Model and ratings paths are relative to the catalog directory. Add an entry under `robots` containing:

```yaml
my_robot:
  models:
    base: ../models/my_robot/scene.xml
  joint_groups:
    waist: [torso_joint]
    arm: [joint_1, joint_2, joint_3]
  default_groups: [waist, arm]
  excluded_joints: [leg_joint]
  hidden_body_roots: [leg_root]
  end_effectors:
    tool:
      site: tool_tip
      body: tool_link          # used only if site does not exist
      position_m: [0, 0, 0.1] # site offset in this body
      com_body: tool_link      # optional COM visualization
  torque_limits_Nm:           # required if XML lacks joint actuatorfrcrange
    torso_joint: [-100, 100]
    joint_1: [-30, 30]
    joint_2: [-30, 30]
    joint_3: [-10, 10]
  motor_ratings: my_robot_ratings.yaml # optional; never inferred from another robot
```

Use `--robot my_robot --variant base --ee tool`. Supported robot models have finite-range revolute joints and independent joint torque actuation. Existing actuators are replaced by unit-gear torque actuators with the configured joint-output limits. Free-base and other inactive joints are fixed. Coupled tendons/equalities are rejected explicitly. Runtime input is MuJoCo XML; convert URDF and define the scene/EE sites first. `initial_positions_rad` optionally sets active joint starting positions. Robot `presets` can supply A/B/C mappings from joint names to degrees.

## Add a domain

Domain entries are independent from robot entries. Each declares `kind: payload_lift` or `kind: waypoint_motion` and may restrict `variants`. `model_overrides` can select a domain-specific scene per robot and variant:

```yaml
reach_workspace:
  kind: waypoint_motion
  model_overrides:
    my_robot:
      base: ../models/my_robot/reach_scene.xml
```

These two initial kinds share the tested edit/sequence/record runner. Future physical pick-and-place needs a new domain implementation for object DOFs, grasp/release, collision/contact, and termination. It is intentionally not listed as a completed domain: the current model builder disables contacts and fixes non-selected DOFs. Adding YAML alone does not implement grasp physics.

## Data and compatibility

Poses and experiment folders are separated under:
`manipulation_logs/<robot>/<domain>/<variant>/<ee>/<configuration hash>/`.

All selected joints, including waist, appear in `samples.csv`. Metadata records the resolved model, full profile snapshot, active scope, domain, EE and motor ratings path. Pose files require matching robot/domain/variant/joint configuration. Old seven-joint pose files are rejected by the new platform; the legacy script still reads them. Different configurations must be posed/saved separately.

Start begins recording; final arrival ends it. Intermediate dwell and final arrival criteria are unchanged. There is currently no settling timeout. No measurement is recorded during editing or idle/headless waiting.

```bash
python scripts/plot_arm_payload.py "manipulation_logs/<robot>/<domain>/<variant>/<ee>/<hash>/<run>"
```

Pass the actual run folder shown by the recorder (not the top-level manipulation_logs directory). Plotting supports arbitrary joint counts and uses the robot ratings path in metadata; `--ratings` explicitly overrides it. Missing robot ratings remain unspecified.

## Files

- `scripts/manipulation_platform.py`: profile loading/validation, model preparation, selectable platform UI and launcher.
- `configs/manipulation_platform.yaml`: robot, joint groups, model variants, EE frames, ratings, presets and domain catalog.
- Existing `arm_payload.py`: shared dynamics/controller and renderer, preserved legacy launcher.
- Existing `payload_session.py`: shared editing, recording and sequence lifecycle.

Validated with all four prototype variants and an unrelated two-joint test robot. The legs are absent from the prototype's 16 active DOFs. A full arms-and-waist A/B/C measurement, UI creation, domain selection and recording-time switch lock have been exercised.
