# EE 자세 편집 / payload 측정

```bash
cd /home/jiho/robot_keyframe_kit
conda activate robot_keyframe_kit
python scripts/arm_payload.py --payload-kg 2
```

브라우저에서 http://127.0.0.1:8766 접속. 기존 프로세스가 실행 중이면 Ctrl+C로 종료하고 다시 실행한다. 다른 포트는 `--port 8770`처럼 지정한다. 기본은 오른팔이며 `--payload-kg 20`은 right_wrist_pitch_link 자체의 총질량이 20kg인 별도 XML을 선택한다. EE에 추를 추가하지 않는다. 10/15/20은 해당 링크 질량 모델, 0은 원본 모델, 그 외 양수는 기존 EE 추가 질량 방식이다.

## ① EE frame 편집 모드

- EE의 화살표/평면을 드래그하여 위치, 회전 링을 드래그하여 방향을 변경한다. IK로 관절 자세를 직접 적용하며 PD 제어와 데이터 기록은 하지 않는다.
- `저장할 이름`에서 **A, B, C, ordinary pos** 중 선택하고 `현재 EE frame 저장`을 누른다. 같은 이름으로 다시 저장하면 덮어쓴다.
- `ordinary pos`는 처음에는 시작 자세이다. 다른 자세로 덮어쓸 수 있다.
- `선택한 EE frame 적용`로 해당 자세를 편집할 수 있다. `관절 각도로 자세 만들기` 슬라이더도 사용 가능하며 단위는 deg이다.
- `기본 컬 자세를 A/B/C에 넣기`는 A=팔 내림, B=컬 올림, C=다시 내림을 저장한다. 기존 A/B/C를 덮어쓴다.
- 도달할 수 없는 frame은 적용하지 않고 오류를 표시한다. 드래그가 끝나면 실제 적용된 frame으로 돌아온다.
- 저장한 위치·방향·관절 자세는 `payload_logs/ee_poses.json`에 보존하고 다음 실행 시 자동으로 읽는다. 다른 파일은 `--waypoints 파일.json`으로 지정한다.

## ② torque / velocity 측정 모드

1. 네 행의 `점`에서 실행 순서를 정한다. 예: ordinary pos → A → B → C. 같은 점을 반복하거나 `(skip)`으로 행을 건너뛸 수 있다.
2. 각 행의 `이 점까지 이동 (s)`를 설정한다. 첫 행은 현재 자세 → 첫 점, 이후는 이전 선택 점 → 해당 점의 시간이다.
3. `도착 후 유지 (s)`는 해당 점에 도착한 뒤 다음 이동 전까지 유지할 시간이다. **마지막 선택 점은 도착 시 기록을 끝내므로 유지 시간을 0으로 처리하고 입력을 비활성화한다.** 마지막 점을 바꾸면 새 마지막 점에 이 규칙을 적용한다.
4. **Start**를 누르면 이동과 기록이 함께 시작한다. 대기·편집 중에는 측정 CSV를 생성하지 않는다.
5. 마지막 점 도착 판정 시 자동으로 기록을 닫고 `COMPLETED` 및 저장 경로를 표시한다. 이후 자세를 고정하고 기록하지 않는다. 다시 Start를 누르면 새로운 폴더에 독립된 실험을 저장한다.

이동은 저장된 관절 자세 사이의 부드러운 quintic 궤적이며 EE가 직선으로 움직인다는 뜻은 아니다. 이동 시간은 기준 궤적의 시간이다. 실제 도착 판정은 위치 오차 3mm 미만, 방향 오차 0.03rad 미만, 각 관절 속도 0.02rad/s 미만을 0.2초 유지한 시점이므로 실제 구간에는 짧은 안정화 시간이 추가될 수 있다. 중간 점 유지 시간은 도착 판정 후 시작한다.

측정 중에는 모드·경로·시간 변경을 잠근다. `Stop`은 부분 기록을 `user_stopped`로 저장하며, Ctrl+C는 `interrupted`로 저장한다. 최종점에 도달하지 못한 기록을 완료로 표시하지 않는다. 토크 제한 등으로 정지 조건을 만족하지 못하면 SETTLING 상태로 계속 기록하므로 Stop으로 종료할 수 있다.

## 제어 및 기록

측정 중 inverse-dynamics feedforward + PD를 사용한다. 기본 Kp=200, Kd=40이며 측정 화면에서 Kp를 0~1000, Kd를 0~200으로 조절할 수 있다. Start 시 적용되고 측정 중에는 잠긴다. 0도 허용하며, 두 값이 0이어도 inverse-dynamics feedforward는 남아 있다. payload COM 오프셋은 `--payload-com X Y Z`(EE 로컬 좌표, m), 구형 관성 반경은 `--payload-radius`로 설정한다. 다른 관절은 고정되고 접촉은 비활성화된 단일 팔 실험이다.

Start마다 `payload_logs/YYYYMMDD_HHMMSS_microseconds/` 아래 저장한다.

- `samples.csv`: Start 기준 시간(첫 샘플 0초), 각 joint의 실제 torque(N·m), 요청 torque, velocity(rad/s), 관절 위치, EE 위치·방향, 포화 여부, 구간 번호 및 phase 등.
- `metadata.json`: payload, gain, 토크 제한, 순서·이동 시간·유지 시간, 샘플 수, 완료/중단 상태.
- `commands.jsonl`: 실제 구간 시작 시간과 목표 자세.
- `poses.json`: 해당 실험에 사용한 저장 자세 전체의 스냅샷.

측정 중 토크·속도를 터미널에도 출력한다. 실제 torque는 actuator 제한 적용 후 값이며 요청 torque와 구분한다. 저장 형식은 기존 plot 코드와 호환된다.

```bash
python scripts/plot_arm_payload.py payload_logs/실험폴더
```

기존 torque-time, velocity-time, torque-velocity 그래프와 요약 표를 생성한다. 정격/피크 토크는 `configs/torque_ratings.json`의 실제 joint-output 사양을 입력해야 표시할 수 있다.

기존 `--mode slide`, `--mode ik` 명령은 호환용으로 남아 있다. 위의 버튼 기반 두 모드는 기본 실행(`--mode residual`)에 적용된다.

## 저장한 EE frame 파일 불러오기

편집 모드의 `EE frame JSON 경로`에 `payload_logs/ee_poses.json` 또는 이전 실험의 `poses.json` 경로를 입력하고 `파일에서 EE frames 불러오기`를 누른다. A/B/C/ordinary pos가 갱신된다. 원하는 이름을 선택하고 `선택한 EE frame 적용`을 누르면 로봇과 편집 frame이 함께 이동한다. 다른 팔의 관절 이름이나 범위를 벗어나는 값은 거부한다. 파일 불러오기는 원본 파일을 변경하지 않는다.

## 20kg wrist 링크 모델

```bash
python scripts/arm_payload.py --payload-kg 20
```

- 로봇 XML: `prototype_v2.1.1/mjcf/prototype_v2.1.1_wrist20kg.xml`
- scene XML: `prototype_v2.1.1/mjcf/scene_wrist20kg.xml`
- URDF: `prototype_v2.1.1/urdf/prototype_v2.1.1_wrist20kg.urdf`

원본 0.2kg 링크를 총질량 20kg으로 바꾼다. 원래 COM 및 관성축 방향을 유지하고, 같은 형상·질량 분포를 가정하여 관성을 100배로 조정한다. XML의 diagonal inertia는 `[0.0562349, 0.0499452, 0.0130589]` kg·m²이다. URDF는 비대각 성분을 포함한 전체 관성 텐서를 100배로 조정한다. 원본 모델과 모터 토크 제한은 그대로 보존한다.

20kg 모델에는 `--payload-com` 오프셋을 적용하지 않는다. `--payload-radius`도 사용하지 않는다. 시작 시 실제 선택 XML과 질량 구성을 터미널에 표시하고 metadata에 `xml`, `requested_xml`, `added_payload_kg=0`, `payload_definition`을 기록한다. 기존 로그는 과거의 EE 추가 질량 실험이므로 새 모델 측정과 구분해야 한다. 저장한 EE 자세는 링크 좌표계가 같아 계속 사용할 수 있다.

## 10kg / 15kg 링크 모델

`python scripts/arm_payload.py --payload-kg 10` 또는 `--payload-kg 15`로 실행한다. 20kg와 마찬가지로 wrist pitch 링크 자체의 총질량이며 별도 추는 추가하지 않는다. 원래 COM·형상을 유지하고 관성은 원본 0.2kg 대비 각각 50배, 75배이다.

각각 `prototype_v2.1.1/urdf/prototype_v2.1.1_wrist10kg.urdf`, `prototype_v2.1.1/mjcf/prototype_v2.1.1_wrist10kg.xml`, `prototype_v2.1.1/mjcf/scene_wrist10kg.xml`을 사용한다. 15kg 파일은 동일한 이름의 10kg 부분을 15kg로 바꾼 것이다. 실행 시 선택된 XML 경로를 터미널에서 확인할 수 있다.

## 오른쪽 wrist pitch COM 표시

3D 화면의 자홍색 구와 `Right wrist COM (질량 kg)` 라벨이 해당 링크 자체의 무게중심을 표시한다. 모델의 body_ipos / xipos를 사용하므로 EE site 또는 별도 payload의 COM과 다르다. 편집·측정 중 움직임을 따라 갱신된다. 라벨은 로봇 표면에 가려지지 않게 표시한다. `Right wrist pitch COM` 패널에서 표시를 켜고 끄거나 질량, 링크 기준 및 world 기준 COM 좌표(m)를 확인할 수 있다.

## 링크 COM y=-0.0315m 변경

10/15/20kg 파생 URDF·XML의 right_wrist_pitch_link COM y좌표를 링크 기준 -0.063/2 = -0.0315m로 변경했다. x/z, 질량, COM 기준 관성 텐서는 유지한다. XML COM은 `[0.0701643, -0.0315, 0.0124248]` m이다. 원본 0.2kg 모델은 유지한다. 재실행 시 시각화와 동역학에 반영되며, 기존 로그는 변경 전 결과이다.
