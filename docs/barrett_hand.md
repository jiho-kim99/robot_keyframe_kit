# Barrett hand model

The default robot in `evaluate_tasks.py` and `manipulation_platform.py` is now `prototype_barrett` (base variant only, no attached payload). The legacy `prototype_v211` profile and 10/15/20 kg files remain available explicitly for old experiments. `arm_payload.py` is a legacy launcher and retains its original model default.

Generated files:
- `prototype_v2.1.1/urdf/prototype_v2.1.1_barrett.urdf`
- `prototype_v2.1.1/mjcf/prototype_v2.1.1_barrett.xml`
- `prototype_v2.1.1/mjcf/scene_barrett.xml`

Only the right terminal link is replaced. `right_wrist_pitch_joint` and its limits remain intact; `right_wrist_pitch_link` now contains the Barrett palm, not the original terminal geometry/inertia. Left arm is unchanged. Hand mass including fingers and original tiny frame inertias: 1.098463 kg, additional payload: 0 kg. The displayed wrist COM is the palm COM, not the aggregate hand COM.

Mount transform in the original right wrist pitch frame: xyz=(0, -0.0315, 0) m, rpy=(0, pi/2, 0). Barrett +Z points along robot wrist +X. This is an explicit simulation mounting assumption, not a manufacturer-verified mechanical adapter. Grasp site `right_hand_grasp` is (0.12, -0.0315, 0) m in the wrist frame; its +X points out of the palm. Upstream grasp frame/link is also retained. Old saved EE frames must be recorded again because tool geometry/reference changed.

The URDF/MJCF retain eight finger joints and supplied convex collision pieces. The existing arm/waist evaluator freezes non-selected joints, so fingers are fixed closed during current torque measurements (middle joints 90°, distal joints 30°, spread joints 0°; configured in `fixed_joint_positions_rad`). Existing 18 tasks remain representative motion/wrench tests; environment props are visual only and cannot yet be physically grasped. Hand self-mass/inertia affects arm torque, but no object mass is added to the wrist. A future contact-based payload experiment must use a separate dynamic object, finger actuators, contact/friction, and object-lift success checks; do not interpret the moving visual box as a physical grasp.

Run:
```sh
python scripts/evaluate_tasks.py
python scripts/manipulation_platform.py --robot prototype_barrett --variant base
```

Source/license/commit: `third_party/barrett_model/PROVENANCE.md`. Original xacro and BHand meshes are vendored, GPL notice preserved. DAE collision meshes were replaced with the matching upstream STL convex pieces, materials made opaque. Regenerate into a staging directory using `python scripts/build_barrett_model.py --source /path/to/barrett_model --output /tmp/barrett_generated` (requires xacro).
