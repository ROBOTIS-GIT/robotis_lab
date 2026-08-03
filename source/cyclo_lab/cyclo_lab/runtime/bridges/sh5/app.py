"""FFW SH5 Isaac Sim runtime assembly."""

from __future__ import annotations

import os
import time
from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from cyclo_lab.assets.environments.simple_warehouse import (
    make_card_boxes_graspable,
    make_simple_warehouse_environment_cfg,
)
from cyclo_lab.assets.robots import (
    FFW_SH5_CFG,
)
from cyclo_lab.robot_specs.ffw.mobile_base.swerve_drive import SwerveModule
from cyclo_lab.runtime.bridges.sh5.bridge import SH5ZenohRos2Bridge
from cyclo_lab.robot_specs.ffw import sh5 as sh5_cfg
from cyclo_lab.robot_specs.ffw.sh5 import (
    FFW_SH5_ACTION_TOPICS,
    SH5_SWERVE_MODULE_ANGLE_OFFSETS,
    SH5_SWERVE_MODULE_X_OFFSETS,
    SH5_SWERVE_MODULE_Y_OFFSETS,
    SH5_SWERVE_STEERING_JOINTS,
    SH5_SWERVE_WHEEL_JOINTS,
    SH5_SWERVE_WHEEL_RADIUS,
)
from cyclo_lab.runtime.config import (
    FFW_SH5_CMD_VEL_TIMEOUT,
    FFW_SH5_OVERVIEW_CAMERA_EYE,
    FFW_SH5_OVERVIEW_CAMERA_TARGET,
    FFW_SH5_PUBLISH_HZ,
    FFW_SH5_RENDER_INTERVAL,
    FFW_SH5_ROBOT_POS,
    FFW_SH5_STEP_HZ,
)
from cyclo_lab.runtime.transport.ros2_zenoh import best_effort_qos, ros_domain_id


SH5_CAMERA_VIEW_WINDOWS = []


@configclass
class SH5BringupSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    environment: AssetBaseCfg = None
    robot: ArticulationCfg = None


def _make_robot_cfg(usd_path: str) -> ArticulationCfg:
    robot_cfg = deepcopy(FFW_SH5_CFG)
    robot_cfg.spawn.usd_path = usd_path
    robot_cfg.spawn.rigid_props.disable_gravity = False
    robot_cfg.init_state.pos = FFW_SH5_ROBOT_POS
    return robot_cfg


def _trajectory_qos():
    return best_effort_qos(10)


def _enabled_topics(args_cli) -> dict[str, str]:
    topics = {
        "right_arm": FFW_SH5_ACTION_TOPICS["right_arm"],
        "right_hand": FFW_SH5_ACTION_TOPICS["right_hand"],
        "left_arm": FFW_SH5_ACTION_TOPICS["left_arm"],
        "left_hand": FFW_SH5_ACTION_TOPICS["left_hand"],
    }
    if not args_cli.disable_head:
        topics["head"] = FFW_SH5_ACTION_TOPICS["head"]
    if not args_cli.disable_lift:
        topics["lift"] = FFW_SH5_ACTION_TOPICS["lift"]
    return topics


def _swerve_modules() -> list[SwerveModule]:
    return [
        SwerveModule(
            steering_joint=steering_joint,
            wheel_joint=wheel_joint,
            x_offset=SH5_SWERVE_MODULE_X_OFFSETS[index],
            y_offset=SH5_SWERVE_MODULE_Y_OFFSETS[index],
            angle_offset=SH5_SWERVE_MODULE_ANGLE_OFFSETS[index],
            steering_limit_lower=sh5_cfg.FFW_SH5_SWERVE_STEERING_LIMIT_LOWER,
            steering_limit_upper=sh5_cfg.FFW_SH5_SWERVE_STEERING_LIMIT_UPPER,
            wheel_speed_limit_lower=sh5_cfg.FFW_SH5_SWERVE_WHEEL_SPEED_LIMIT_LOWER,
            wheel_speed_limit_upper=sh5_cfg.FFW_SH5_SWERVE_WHEEL_SPEED_LIMIT_UPPER,
        )
        for index, (steering_joint, wheel_joint) in enumerate(
            zip(SH5_SWERVE_STEERING_JOINTS, SH5_SWERVE_WHEEL_JOINTS)
        )
    ]


def _write_default_joint_state(robot):
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.set_joint_position_target(default_joint_pos)
    robot.set_joint_velocity_target(default_joint_vel)


def _find_camera_prim_by_name(stage, prim_name: str):
    for prim in stage.Traverse():
        if prim.GetName() == prim_name and prim.GetTypeName() == "Camera":
            return prim
    return None


def _ensure_camera_viewport_attrs(camera_prim):
    from pxr import Gf, Sdf

    coi_attr = camera_prim.GetProperty("omni:kit:centerOfInterest")
    if not coi_attr or not coi_attr.IsValid():
        coi_attr = camera_prim.CreateAttribute(
            "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
        )
    if coi_attr.Get() is None:
        coi_attr.Set(Gf.Vec3d(0.0, 0.0, -10.0))


def _position_window(window, width: int, height: int, x: int | None = None, y: int | None = None):
    for attr_name, value in (("width", width), ("height", height), ("position_x", x), ("position_y", y)):
        if value is None:
            continue
        try:
            setattr(window, attr_name, value)
        except Exception:
            pass
        try:
            frame = getattr(window, "frame", None)
            if frame is not None:
                setattr(frame, attr_name, value)
        except Exception:
            pass


def _set_viewport_camera(
    window_name: str,
    camera_path: str,
    width: int = 640,
    height: int = 480,
    x: int | None = None,
    y: int | None = None,
):
    try:
        from omni.kit.viewport.utility import create_viewport_window, get_viewport_from_window_name
        from pxr import Sdf

        viewport = get_viewport_from_window_name(window_name)
        if viewport is None:
            window = create_viewport_window(
                window_name,
                width=width,
                height=height,
                position_x=0 if x is None else x,
                position_y=0 if y is None else y,
                camera_path=Sdf.Path(camera_path),
            )
            SH5_CAMERA_VIEW_WINDOWS.append(window)
            _position_window(window, width, height, x, y)
            viewport = get_viewport_from_window_name(window_name)
        if viewport is not None:
            viewport.set_active_camera(camera_path)
            return True
    except Exception as exc:
        print(f"[WARN] Could not create viewport '{window_name}': {exc}")
    return False


def _setup_camera_views():
    from isaacsim.core.utils.stage import get_current_stage

    stage = get_current_stage()

    camera_specs = (
        ("Center Camera", sh5_cfg.AI_WORKER_CAMERA_CENTER_NAME, 780, 490, 50, 22),
        ("Left Camera", sh5_cfg.AI_WORKER_CAMERA_LEFT_NAME, 387, 280, 50, 517),
        ("Right Camera", sh5_cfg.AI_WORKER_CAMERA_RIGHT_NAME, 387, 280, 441, 517),
    )
    camera_paths: dict[str, str] = {}
    missing_camera_names: list[str] = []

    for window_name, camera_name, width, height, x, y in camera_specs:
        camera_prim = _find_camera_prim_by_name(stage, camera_name)
        if camera_prim is None:
            missing_camera_names.append(camera_name)
            continue
        _ensure_camera_viewport_attrs(camera_prim)
        camera_path = str(camera_prim.GetPath())
        camera_paths[camera_name] = camera_path
        _set_viewport_camera(window_name, camera_path, width=width, height=height, x=x, y=y)

    print("[INFO] Main Isaac Sim viewport left unchanged for overview/manual view.")
    for camera_name, camera_path in camera_paths.items():
        print(f"[INFO] {camera_name}: {camera_path}")
    if missing_camera_names:
        available_cameras = [
            str(prim.GetPath()) for prim in stage.Traverse() if prim.GetTypeName() == "Camera"
        ]
        print(f"[WARN] Missing requested camera prims: {missing_camera_names}")
        print(f"[WARN] Available cameras: {available_cameras}")


def run_simulator(simulation_app, args_cli, sim: sim_utils.SimulationContext, scene: InteractiveScene, bridge: SH5ZenohRos2Bridge):
    sim_dt = sim.get_physics_dt()
    step_period = 1.0 / FFW_SH5_STEP_HZ if FFW_SH5_STEP_HZ > 0 else 0.0
    publish_period = 1.0 / FFW_SH5_PUBLISH_HZ if FFW_SH5_PUBLISH_HZ > 0 else 0.0
    last_publish = 0.0
    last_step = time.time()
    start_time = time.time()

    while simulation_app.is_running():
        if args_cli.max_runtime > 0.0 and time.time() - start_time >= args_cli.max_runtime:
            print(f"[INFO] max_runtime reached: {args_cli.max_runtime:.3f}s")
            break

        bridge.apply_latest_targets()
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        bridge.update_odometry(sim_dt)

        now = time.time()
        if publish_period == 0.0 or now - last_publish >= publish_period:
            bridge.publish_joint_states()
            bridge.publish_odometry()
            bridge.publish_tf()
            last_publish = now

        if step_period > 0.0:
            next_step = last_step + step_period
            sleep_time = next_step - time.time()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            last_step = next_step if sleep_time > 0.0 else time.time()


def main(args_cli, simulation_app):
    usd_path = FFW_SH5_CFG.spawn.usd_path
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"SH5 USD not found: {usd_path}")

    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=1.0 / FFW_SH5_STEP_HZ,
        render_interval=FFW_SH5_RENDER_INTERVAL,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(FFW_SH5_OVERVIEW_CAMERA_EYE, FFW_SH5_OVERVIEW_CAMERA_TARGET)

    scene_cfg = SH5BringupSceneCfg(num_envs=1, env_spacing=2.0)
    if args_cli.enable_environment:
        scene_cfg.environment = make_simple_warehouse_environment_cfg()
    scene_cfg.robot = _make_robot_cfg(usd_path).replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)
    if args_cli.enable_environment:
        make_card_boxes_graspable()

    sim.reset()
    scene.reset()
    scene.update(sim.get_physics_dt())

    robot = scene["robot"]
    _write_default_joint_state(robot)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
    if args_cli.enable_camera_views:
        _setup_camera_views()

    domain_id = ros_domain_id()
    bridge = SH5ZenohRos2Bridge(
        robot=robot,
        topic_names=_enabled_topics(args_cli),
        joint_states_topic=sh5_cfg.JOINT_STATES_TOPIC,
        odom_topic=sh5_cfg.ODOM_TOPIC,
        tf_topic=sh5_cfg.TF_TOPIC,
        base_frame=sh5_cfg.BASE_FRAME,
        odom_frame=sh5_cfg.ODOM_FRAME,
        trajectory_qos=_trajectory_qos(),
        cmd_vel_topic=None if args_cli.disable_cmd_vel else sh5_cfg.CMD_VEL_TOPIC,
        swerve_modules=[] if args_cli.disable_cmd_vel else _swerve_modules(),
        wheel_radius=SH5_SWERVE_WHEEL_RADIUS,
        cmd_vel_timeout=FFW_SH5_CMD_VEL_TIMEOUT,
    )

    print(f"[INFO] FFW SH5 Zenoh ROS2 runtime ready. ROS_DOMAIN_ID={domain_id}")
    if args_cli.enable_environment:
        print("[INFO] Environment: Simple Warehouse")
    print("[Zenoh ROS2] JointTrajectory subscriber reliability: best_effort")
    print(f"[Zenoh ROS2] Publishing joint states: {sh5_cfg.JOINT_STATES_TOPIC}")
    print(f"[Zenoh ROS2] Publishing odometry: {sh5_cfg.ODOM_TOPIC} ({sh5_cfg.ODOM_FRAME} -> {sh5_cfg.BASE_FRAME})")
    print(f"[Zenoh ROS2] Publishing TF: {sh5_cfg.TF_TOPIC} ({sh5_cfg.BASE_FRAME} -> robot links)")
    if not args_cli.disable_cmd_vel:
        print(f"[Zenoh ROS2] Applying swerve cmd_vel: {sh5_cfg.CMD_VEL_TOPIC}")

    try:
        run_simulator(simulation_app, args_cli, sim, scene, bridge)
    finally:
        bridge.shutdown()
