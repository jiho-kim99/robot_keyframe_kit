# Payload 로그 그래프

```bash
cd /home/jiho/robot_keyframe_kit
python scripts/plot_arm_payload.py "payload_logs/20260923_bicep curl"
```

경로를 생략하면 payload_logs의 최신 samples.csv를 선택한다. 공백이 있는 경로는 따옴표로 감싼다.

기본 사양 파일은 `configs/motor_ratings.yaml`이다. 사용자가 제공한 `/home/jiho/whole_body_tracking_log_visual/prototype_v2.1.1/motor_ratings.yaml`을 복사한 파일이며, 원본의 이후 변경은 자동 동기화하지 않는다. 원본을 직접 사용하려면:

```bash
python scripts/plot_arm_payload.py "payload_logs/20260923_bicep curl" --ratings /home/jiho/whole_body_tracking_log_visual/prototype_v2.1.1/motor_ratings.yaml
```

출력은 실험 폴더의 `plots/`이며 `--output`으로 변경 가능하다.

- `v_t.png`, `torque_velocity.png`: 모든 관절의 T-V 곡선. x=|velocity| [rad/s], y=|applied torque| [N·m].
- `<joint>_t_v.png`: 각 관절의 개별 T-V 곡선.
- 파란 곡선: `peak_torque_nm * max(0, 1 - |v| / no_load_velocity_rad_s)`.
- 주황 곡선: `min(rated_torque_nm, peak_curve)`.
- 파란 점: 측정값. 주황 x: rated 초과·peak 이내. 빨간 x: peak 초과. 무부하 속도를 넘는 점도 peak 초과로 표시한다.
- `torque_table.png`, `torque_table.md`, `summary.csv`: 측정 데이터에서 계산한 rated torque(RMS), peak torque(절댓값 최대), rated velocity(RMS), peak velocity(절댓값 최대)만 표시한다. Rated는 실제 모터 정격이 아니라 선택 구간의 시간 가중 RMS이다. 제곱값을 시간에 대해 사다리꼴 적분하고 구간 길이로 나눈 뒤 제곱근을 취한다. 제외된 구간의 시간은 적분하지 않는다. CSV의 rated_* 열도 같은 RMS 값을 저장한다.
- `torque_time.png`, `velocity_time.png`, `ee_velocity_time.png`: 기존 시간 그래프.

곡선은 YAML에서 정의한 선형 근사이며 제조사 실측 곡선은 아니다. YAML은 그래프의 한계 곡선에만 사용하며 표는 측정 데이터로 계산한다. rated_velocity_rad_s는 곡선의 꺾임점 계산에는 사용하지 않는다. 필요한 사양이 없으면 N/A 또는 곡선 없음으로 표시하며 XML 토크 한도를 모터 사양으로 대신 쓰지 않는다. 기존 JSON 사양 파일도 --ratings로 읽을 수 있지만 속도 정보가 없으면 T-V 곡선을 그리지 않는다.

필요 패키지: numpy, matplotlib, pyyaml. `--joint wrist_pitch`, `--command 1 2`, `--start 0 --end 5`, `--show` 옵션도 사용 가능하다.

## 측정값과 모터 사양 비교 표

각 관절을 Measured / Motor (YAML) 두 행으로 표시한다. Measured 행은 선택 구간의 시간 가중 RMS 및 절댓값 최대 토크·속도이고, Motor 행은 YAML의 정격/피크 토크, 정격 속도, 무부하 속도이다. 모터 행의 마지막 열은 실제 peak velocity 사양이 아닌 no_load_velocity_rad_s임을 명시한다. CSV는 기존 측정값 4개 열과 motor_ 접두사의 사양 4개 열을 함께 저장한다.
