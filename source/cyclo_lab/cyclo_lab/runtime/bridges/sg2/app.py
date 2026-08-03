"""FFW SG2 Isaac Sim runtime assembly."""

from __future__ import annotations

import contextlib
import os
import time
from collections import defaultdict
from copy import deepcopy

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from cyclo_lab.assets.environments.galileo_locomanip import (
    GALILEO_LOCOMANIP_ENVIRONMENT_USD_PATH,
    make_galileo_locomanip_environment_cfg,
)
from cyclo_lab.assets.environments.robotis_showroom import (
    ROBOTIS_SHOWROOM_USD_PATH,
    make_robotis_showroom_environment_cfg,
)
from cyclo_lab.assets.robots import (
    FFW_SG2_PHYSICS_CFG,
)
from cyclo_lab.assets.sensors.ffw_sg2_cameras import (
    FFW_SG2_HEAD_CAMERA_NAME,
    FFW_SG2_WRIST_LEFT_CAMERA_NAME,
    FFW_SG2_WRIST_RIGHT_CAMERA_NAME,
    camera_publish_period,
    make_ffw_sg2_head_camera_cfg,
    make_ffw_sg2_wrist_camera_cfg,
)
from cyclo_lab.robot_specs.ffw.mobile_base.swerve_drive import SwerveModule
from cyclo_lab.runtime.bridges.reset_handler import ResetRequestHandler
from cyclo_lab.runtime.bridges.sg2.bridge import SG2ZenohRos2Bridge
from cyclo_lab.robot_specs.ffw import sg2 as sg2_cfg
from cyclo_lab.robot_specs.ffw.sg2 import (
    FFW_SG2_ACTION_TOPICS,
    FFW_SG2_CAMERA_TOPICS,
    FFW_SG2_HEAD_JOINT_NAMES,
    FFW_SG2_LEFT_ARM_JOINT_NAMES,
    FFW_SG2_LEFT_GRIPPER_JOINT_NAMES,
    FFW_SG2_LIFT_JOINT_NAMES,
    FFW_SG2_RIGHT_ARM_JOINT_NAMES,
    FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES,
    FFW_SG2_SWERVE_DRIVE_SPEED_SCALE,
    FFW_SG2_SWERVE_STEERING_LIMIT_LOWER,
    FFW_SG2_SWERVE_STEERING_LIMIT_UPPER,
    FFW_SG2_SWERVE_WHEEL_SPEED_LIMIT_LOWER,
    FFW_SG2_SWERVE_WHEEL_SPEED_LIMIT_UPPER,
    SG2_SWERVE_MODULE_ANGLE_OFFSETS,
    SG2_SWERVE_MODULE_X_OFFSETS,
    SG2_SWERVE_MODULE_Y_OFFSETS,
    SG2_SWERVE_STEERING_JOINTS,
    SG2_SWERVE_WHEEL_JOINTS,
    SG2_SWERVE_WHEEL_RADIUS,
)
from cyclo_lab.runtime.config import (
    FFW_SG2_CMD_VEL_TIMEOUT,
    FFW_SG2_ENVIRONMENT_GROUND_Z,
    FFW_SG2_OVERVIEW_CAMERA_EYE,
    FFW_SG2_OVERVIEW_CAMERA_TARGET,
    FFW_SG2_PUBLISH_HZ,
    FFW_SG2_RENDER_INTERVAL,
    FFW_SG2_ROBOT_POS,
    FFW_SG2_ROBOT_ROT,
    FFW_SG2_STEP_HZ,
)
from cyclo_lab.runtime.transport.ros2_zenoh import best_effort_qos, ros_domain_id


_SG2_RUNTIME_LEFT_READY_JOINT_POSITIONS = (0.0659, 0.3421, 0.5123, -2.4973, 0.612, 0.8882, -0.6281, 0.0)
_SG2_RUNTIME_RIGHT_READY_JOINT_POSITIONS = (0.0659, -0.3421, -0.5123, -2.4973, -0.612, 0.8882, 0.6281, 0.0)
_SG2_RUNTIME_READY_JOINT_POSITIONS = {
    **dict(
        zip(
            (*FFW_SG2_LEFT_ARM_JOINT_NAMES, *FFW_SG2_LEFT_GRIPPER_JOINT_NAMES),
            _SG2_RUNTIME_LEFT_READY_JOINT_POSITIONS,
        )
    ),
    **dict(
        zip(
            (*FFW_SG2_RIGHT_ARM_JOINT_NAMES, *FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES),
            _SG2_RUNTIME_RIGHT_READY_JOINT_POSITIONS,
        )
    ),
}
_GALILEO_LOCOMANIP_SG2_POS = (0.0, 0.18, 0.0)
_GALILEO_LOCOMANIP_SG2_ROT = (1.0, 0.0, 0.0, 0.0)


class RuntimeLoopProfiler:
    """Small timing profiler for the SG2 runtime loop."""

    def __init__(self, enabled: bool, interval: int, cuda_sync: bool = False):
        self.enabled = enabled
        self.interval = max(1, int(interval))
        self.cuda_sync = bool(cuda_sync)
        self.loop_count = 0
        self._window_start = time.perf_counter()
        self._stats = defaultdict(lambda: {"count": 0, "total": 0.0, "max": 0.0})

    def _sync(self):
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextlib.contextmanager
    def time(self, name: str):
        if not self.enabled:
            yield
            return
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - start
            stat = self._stats[name]
            stat["count"] += 1
            stat["total"] += elapsed
            stat["max"] = max(stat["max"], elapsed)

    def tick(self):
        if not self.enabled:
            return
        self.loop_count += 1
        if self.loop_count % self.interval != 0:
            return

        now = time.perf_counter()
        window_elapsed = max(now - self._window_start, 1e-9)
        measured_hz = self.interval / window_elapsed
        loop_total = self._stats.get("loop_total", {}).get("total", 0.0)

        print(f"\n[PROFILE] last {self.interval} loops: wall_hz={measured_hz:.2f}")
        for name, stat in sorted(self._stats.items(), key=lambda item: item[1]["total"], reverse=True):
            if stat["count"] == 0:
                continue
            percent = (stat["total"] / loop_total * 100.0) if loop_total > 0.0 and name != "loop_total" else 0.0
            print(
                f"[PROFILE] {name:32s} "
                f"mean={stat['total'] / stat['count'] * 1000.0:8.2f}ms "
                f"max={stat['max'] * 1000.0:8.2f}ms "
                f"total={stat['total']:7.3f}s "
                f"n={stat['count']:5d} "
                f"{percent:5.1f}%"
            )

        self._stats.clear()
        self._window_start = now


@configclass
class SG2BringupSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    robot: ArticulationCfg = None
    environment: AssetBaseCfg = None
    cam_head: CameraCfg = None
    cam_wrist_left: CameraCfg = None
    cam_wrist_right: CameraCfg = None


def _make_robot_cfg(
    robot_pos: tuple[float, float, float] = FFW_SG2_ROBOT_POS,
    robot_rot: tuple[float, float, float, float] = FFW_SG2_ROBOT_ROT,
) -> ArticulationCfg:
    robot_cfg = deepcopy(FFW_SG2_PHYSICS_CFG)
    robot_cfg.spawn.rigid_props.disable_gravity = False
    robot_cfg.init_state.pos = robot_pos
    robot_cfg.init_state.rot = robot_rot
    robot_cfg.init_state.joint_pos.update(_SG2_RUNTIME_READY_JOINT_POSITIONS)
    base_drive_actuator = robot_cfg.actuators.get("base_drive")
    if base_drive_actuator is not None:
        base_drive_actuator.velocity_limit_sim *= FFW_SG2_SWERVE_DRIVE_SPEED_SCALE
    return robot_cfg


def _trajectory_qos():
    return best_effort_qos(10)


def _enabled_topics(args_cli) -> dict[str, tuple[str, tuple[str, ...]]]:
    topics = {}
    if not args_cli.disable_left_arm:
        topics["left_arm"] = (
            FFW_SG2_ACTION_TOPICS["left_arm"],
            (*FFW_SG2_LEFT_ARM_JOINT_NAMES, *FFW_SG2_LEFT_GRIPPER_JOINT_NAMES),
        )
    if not args_cli.disable_right_arm:
        topics["right_arm"] = (
            FFW_SG2_ACTION_TOPICS["right_arm"],
            (*FFW_SG2_RIGHT_ARM_JOINT_NAMES, *FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES),
        )
    if not args_cli.disable_head:
        topics["head"] = (FFW_SG2_ACTION_TOPICS["head"], FFW_SG2_HEAD_JOINT_NAMES)
    if not args_cli.disable_lift:
        topics["lift"] = (FFW_SG2_ACTION_TOPICS["lift"], FFW_SG2_LIFT_JOINT_NAMES)
    return topics


def _mobile_base_enabled(args_cli) -> bool:
    return not args_cli.disable_cmd_vel


def _swerve_modules() -> list[SwerveModule]:
    return [
        SwerveModule(
            steering_joint=steering_joint,
            wheel_joint=wheel_joint,
            x_offset=SG2_SWERVE_MODULE_X_OFFSETS[index],
            y_offset=SG2_SWERVE_MODULE_Y_OFFSETS[index],
            angle_offset=SG2_SWERVE_MODULE_ANGLE_OFFSETS[index],
            steering_limit_lower=FFW_SG2_SWERVE_STEERING_LIMIT_LOWER,
            steering_limit_upper=FFW_SG2_SWERVE_STEERING_LIMIT_UPPER,
            wheel_speed_limit_lower=FFW_SG2_SWERVE_WHEEL_SPEED_LIMIT_LOWER * FFW_SG2_SWERVE_DRIVE_SPEED_SCALE,
            wheel_speed_limit_upper=FFW_SG2_SWERVE_WHEEL_SPEED_LIMIT_UPPER * FFW_SG2_SWERVE_DRIVE_SPEED_SCALE,
        )
        for index, (steering_joint, wheel_joint) in enumerate(
            zip(SG2_SWERVE_STEERING_JOINTS, SG2_SWERVE_WHEEL_JOINTS)
        )
    ]


def _write_default_joint_state(robot):
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.set_joint_position_target(default_joint_pos)
    robot.set_joint_velocity_target(default_joint_vel)


def _print_robot_info(robot):
    print("[INFO] SG2 joint names:")
    for joint_name in robot.data.joint_names:
        print(f"  - {joint_name}")
    print("[INFO] SG2 body names:")
    for body_name in robot.data.body_names:
        print(f"  - {body_name}")


def _reset_runtime_scene(sim: sim_utils.SimulationContext, scene: InteractiveScene, bridge: SG2ZenohRos2Bridge):
    print("[INFO] Resetting SG2 runtime scene.")
    bridge.reset_runtime_state()
    sim.reset()
    scene.reset()
    scene.update(sim.get_physics_dt())

    robot = scene["robot"]
    _write_default_joint_state(robot)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
    print("[INFO] SG2 runtime scene reset complete.")


def run_simulator(
    simulation_app,
    args_cli,
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    bridge: SG2ZenohRos2Bridge,
    reset_handler: ResetRequestHandler | None = None,
):
    sim_dt = sim.get_physics_dt()
    step_period = 1.0 / FFW_SG2_STEP_HZ if FFW_SG2_STEP_HZ > 0 else 0.0
    publish_period = 1.0 / FFW_SG2_PUBLISH_HZ if FFW_SG2_PUBLISH_HZ > 0 else 0.0
    image_publish_period = camera_publish_period(args_cli.camera_publish_hz)
    last_publish = 0.0
    last_camera_publish = 0.0
    last_step = time.time()
    start_time = time.time()
    profiler = RuntimeLoopProfiler(
        enabled=bool(getattr(args_cli, "profile", False)),
        interval=int(getattr(args_cli, "profile_interval", 120)),
        cuda_sync=bool(getattr(args_cli, "profile_cuda_sync", False)),
    )

    while simulation_app.is_running():
        if args_cli.max_runtime > 0.0 and time.time() - start_time >= args_cli.max_runtime:
            print(f"[INFO] max_runtime reached: {args_cli.max_runtime:.3f}s")
            break
        if reset_handler is not None and reset_handler.consume_reset_request():
            _reset_runtime_scene(sim, scene, bridge)
            last_publish = 0.0
            last_camera_publish = 0.0
            last_step = time.time()
            continue

        with profiler.time("loop_total"):
            with profiler.time("apply_targets"):
                bridge.apply_latest_targets()
                bridge.update_odometry(sim_dt)
            with profiler.time("scene_write"):
                scene.write_data_to_sim()
            if profiler.enabled:
                with profiler.time("sim_step_no_render"):
                    sim.step(render=False)
                with profiler.time("sim_render"):
                    sim.render()
            else:
                with profiler.time("sim_step"):
                    sim.step()
            with profiler.time("scene_update"):
                scene.update(sim_dt)

            now = time.time()
            if publish_period == 0.0 or now - last_publish >= publish_period:
                with profiler.time("publish_state"):
                    bridge.publish_joint_states()
                    bridge.publish_odometry()
                    bridge.publish_tf()
                last_publish = now
            if image_publish_period == 0.0 or now - last_camera_publish >= image_publish_period:
                with profiler.time("publish_cameras"):
                    bridge.publish_cameras()
                last_camera_publish = now

            if step_period > 0.0:
                with profiler.time("rate_sleep"):
                    next_step = last_step + step_period
                    sleep_time = next_step - time.time()
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)
                    last_step = next_step if sleep_time > 0.0 else time.time()
        profiler.tick()


def main(args_cli, simulation_app):
    camera_enabled = args_cli.enable_camera
    usd_path = FFW_SG2_PHYSICS_CFG.spawn.usd_path
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"SG2 USD not found: {usd_path}")

    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=1.0 / FFW_SG2_STEP_HZ,
        render_interval=FFW_SG2_RENDER_INTERVAL,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(FFW_SG2_OVERVIEW_CAMERA_EYE, FFW_SG2_OVERVIEW_CAMERA_TARGET)

    scene_cfg = SG2BringupSceneCfg(num_envs=1, env_spacing=2.0)
    environment_defaults = {
        "robotis_showroom": ROBOTIS_SHOWROOM_USD_PATH,
        "galileo_locomanip": GALILEO_LOCOMANIP_ENVIRONMENT_USD_PATH,
    }
    environment_usd_path = args_cli.environment_usd or environment_defaults[args_cli.environment]
    if args_cli.enable_environment and args_cli.environment == "galileo_locomanip":
        robot_pos = _GALILEO_LOCOMANIP_SG2_POS
        robot_rot = _GALILEO_LOCOMANIP_SG2_ROT
    else:
        robot_pos = FFW_SG2_ROBOT_POS
        robot_rot = FFW_SG2_ROBOT_ROT
    scene_cfg.robot = _make_robot_cfg(robot_pos, robot_rot).replace(prim_path="{ENV_REGEX_NS}/Robot")
    if args_cli.enable_environment:
        if "://" not in environment_usd_path and not os.path.exists(environment_usd_path):
            raise FileNotFoundError(f"Environment USD not found: {environment_usd_path}")
        if args_cli.environment == "galileo_locomanip":
            # Galileo supplies its own collision floor at world z=0.
            scene_cfg.ground = None
            scene_cfg.environment = make_galileo_locomanip_environment_cfg(environment_usd_path)
        else:
            scene_cfg.ground.init_state.pos = (0.0, 0.0, FFW_SG2_ENVIRONMENT_GROUND_Z)
            scene_cfg.ground.spawn.visible = True
            scene_cfg.ground.spawn.color = None
            scene_cfg.environment = make_robotis_showroom_environment_cfg(environment_usd_path)
    if camera_enabled:
        image_update_period = camera_publish_period(args_cli.camera_publish_hz)
        scene_cfg.cam_head = make_ffw_sg2_head_camera_cfg(update_period=image_update_period)
        scene_cfg.cam_wrist_left = make_ffw_sg2_wrist_camera_cfg("left", update_period=image_update_period)
        scene_cfg.cam_wrist_right = make_ffw_sg2_wrist_camera_cfg("right", update_period=image_update_period)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()
    scene.update(sim.get_physics_dt())

    robot = scene["robot"]
    _write_default_joint_state(robot)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

    if args_cli.print_robot_info:
        _print_robot_info(robot)
    cameras = {}
    if camera_enabled:
        cameras = {
            FFW_SG2_HEAD_CAMERA_NAME: scene[FFW_SG2_HEAD_CAMERA_NAME],
            FFW_SG2_WRIST_LEFT_CAMERA_NAME: scene[FFW_SG2_WRIST_LEFT_CAMERA_NAME],
            FFW_SG2_WRIST_RIGHT_CAMERA_NAME: scene[FFW_SG2_WRIST_RIGHT_CAMERA_NAME],
        }

    domain_id = ros_domain_id()
    bridge = SG2ZenohRos2Bridge(
        robot=robot,
        topic_names=_enabled_topics(args_cli),
        joint_states_topic=sg2_cfg.JOINT_STATES_TOPIC,
        odom_topic=sg2_cfg.ODOM_TOPIC,
        tf_topic=None if args_cli.disable_tf else sg2_cfg.TF_TOPIC,
        base_frame=args_cli.base_frame,
        base_body=args_cli.base_body,
        odom_frame=sg2_cfg.ODOM_FRAME,
        trajectory_qos=_trajectory_qos(),
        cmd_vel_topic=sg2_cfg.CMD_VEL_TOPIC if _mobile_base_enabled(args_cli) else None,
        swerve_modules=_swerve_modules() if _mobile_base_enabled(args_cli) else [],
        wheel_radius=SG2_SWERVE_WHEEL_RADIUS,
        cmd_vel_timeout=FFW_SG2_CMD_VEL_TIMEOUT,
        publish_lift_target_state=not args_cli.publish_measured_lift_state,
        cameras=cameras,
        camera_topics=FFW_SG2_CAMERA_TOPICS if camera_enabled else None,
    )

    print(f"[INFO] FFW SG2 Zenoh ROS2 runtime ready. ROS_DOMAIN_ID={domain_id}")
    print(f"[INFO] SG2 USD: {usd_path}")
    if args_cli.enable_environment:
        print(f"[INFO] Environment: {args_cli.environment} ({environment_usd_path})")
    print("[Zenoh ROS2] JointTrajectory subscriber reliability: best_effort")
    print(f"[Zenoh ROS2] Publishing joint states: {sg2_cfg.JOINT_STATES_TOPIC}")
    if _mobile_base_enabled(args_cli):
        print(f"[Zenoh ROS2] Publishing odometry: {sg2_cfg.ODOM_TOPIC} ({sg2_cfg.ODOM_FRAME} -> {bridge.base_frame})")
        print(f"[Zenoh ROS2] Applying swerve cmd_vel: {sg2_cfg.CMD_VEL_TOPIC}")
        print("[INFO] SG2 base integration: wheel-contact physics")
    if bridge.publish_lift_target_state:
        print("[Zenoh ROS2] Publishing lift_joint target in joint_states to keep incremental lift commands stable.")
    if not args_cli.disable_tf:
        print(f"[Zenoh ROS2] Publishing TF: {sg2_cfg.TF_TOPIC} ({bridge.base_frame} -> robot links)")
    if camera_enabled:
        for camera_name, camera_topic in FFW_SG2_CAMERA_TOPICS.items():
            print(f"[Zenoh ROS2] Publishing camera {camera_name}: {camera_topic}")

    is_headless = bool(getattr(args_cli, "headless", False))
    reset_handler = ResetRequestHandler(enable_gui=not is_headless, enable_stdin=is_headless)
    try:
        run_simulator(simulation_app, args_cli, sim, scene, bridge, reset_handler)
    finally:
        reset_handler.close()
        bridge.shutdown()


def close_simulation_app(simulation_app, *, skip_cleanup: bool = False):
    try:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=skip_cleanup)
    except TypeError:
        simulation_app.close()
