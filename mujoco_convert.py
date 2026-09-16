import mujoco

urdf_path = "/home/jiho/robot_keyframe_kit/prototype_v2.1.1/urdf/prototype_v2.1.1.urdf"
xml_path = "/home/jiho/robot_keyframe_kit/prototype_v2.1.1/mjcf/robot.xml"

# URDF 불러오기
model = mujoco.MjModel.from_xml_path(urdf_path)

# MJCF XML로 저장
mujoco.mj_saveLastXML(xml_path, model)

print("saved:", xml_path)