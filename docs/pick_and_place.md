# Front / right / rear pick and place

Launch `python scripts/evaluate_tasks.py --task pick_and_place`. Restart an existing server to load changes.

Default layout (`configs/evaluation_tasks.yaml`, `pick_and_place.layout`): table surface at waist pitch axis height (1.01001 m for prototype); 4 cm tall objects; front source x=0.38, y=0; right basket x=0, y=-0.38; rear basket x=-0.38, y=0. Basket floor at table height, rim 6 cm high.

Cycle: front overhead approach → descend → pick preview → lift → right basket (arm only, waist yaw/pitch zero) → release preview → return front → second overhead pickup → lift → waist yaw -180° with arm posture held → rear basket → release preview → retreat → return front for next cycle. Default complete cycle is 16 seconds, including both transfers, dwells, and return. GUI Motion Duration scales the same geometric path.

Pickup targets constrain the grasp frame +X downward. Side transfer follows an IK-sampled outside arc; tool orientation is allowed to change during transfer. Joint-space quintic interpolation handles other segments. Arc geometry is a cubic spline with quintic timing; velocity/acceleration are analytic and zero at major segment boundaries. All path geometry and layout are logged in `task_layout`, with duration and a geometry hash. Unreachable top-down targets or joint-limit violations are rejected.

This is a motion/layout preview, as agreed before implementation: fingers remain fixed closed. The two visual workpieces attach to the EE at configured phases, appear inside each basket at release, and replenish each cycle. No grasp/contact physics, collision-avoidance guarantee, grip force, or object inertia is simulated. Recorded torques include robot/hand dynamics and the configured object gravity load during carrying. Very short cycle times can saturate actuators and cause actual motion to lag reference timing; the preview events follow the reference clock, not verified successful grasp.

## Object mass input
GUI `Object Mass [kg]` (or CLI `--object-mass-kg`) sets the weight load on Start. Pause/Stop and stop logging before editing. Pick/place applies m*g only during the two carrying intervals; it removes the load on release, and retains it while paused during a carry. The setting is recorded as `object_mass_kg`; CSV `active_object_mass_kg` records whether it is currently applied. Changing mass closes the prior log. Other tasks apply their configured static weight continuously, as before. Robot/hand inertia remains unchanged; object acceleration/rotation inertia and physical grasp remain unmodeled.
