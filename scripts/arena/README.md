# Cyclo Arena launcher

## Standard workflow

Run inference from the Cyclo Lab host checkout with one command:

```bash
./scripts/arena/run.sh
```

The no-argument command selects the `ffw_sg2_gr00t` profile. It starts the Cyclo Lab container,
validates the checkpoint, builds or starts the matching GR00T server when necessary, and launches
Isaac Sim. Entering the container or running `start-groot` first is not required.

Normal setup is intentionally limited to two editable YAML locations:

- `source/cyclo_arena/configs/profiles/ffw_sg2_gr00t.yaml` selects the robot, scene, checkpoint,
  language instruction, and runtime options. It also lists every available scene and every supported
  field.
- `source/cyclo_arena/configs/robots/ffw_sg2/poses/showroom.yaml` defines the joint pose selected by
  `robot.initial_pose: showroom`. Keep it aligned with the checkpoint's training-time initial state.

The default checkpoint directory is `docker/workspace/model/showroom_groot` on the host. Docker
mounts `docker/workspace/model` at `/workspace/model`; `${CYCLO_ARENA_MODEL_ROOT}` therefore keeps
the profile portable between host and container. Selecting another compatible checkpoint only
requires changing `model.checkpoint`. No Python catalog entry is needed.

## Named profiles and overrides

Profiles are selected by ID, without exposing their filesystem path:

```bash
./scripts/arena/run.sh infer ffw_sg2_gr00t
./scripts/arena/run.sh show profile ffw_sg2_gr00t
./scripts/arena/run.sh plan ffw_sg2_gr00t
./scripts/arena/run.sh validate ffw_sg2_gr00t
```

`plan` prints the immutable `ResolvedManifest` passed across the host/container boundary.
`validate` checks robot, scene, task, checkpoint, embodiment, and adapter compatibility without
starting Isaac Sim. Command-line values can temporarily override profile values:

```bash
./scripts/arena/run.sh --scene kitchen --num-steps 100 --headless
./scripts/arena/run.sh --dry-run
```

Inspect registered and discovered choices with:

```bash
./scripts/arena/run.sh list robots
./scripts/arena/run.sh list scenes
./scripts/arena/run.sh list models
./scripts/arena/run.sh list profiles
./scripts/arena/run.sh list workflows
./scripts/arena/run.sh --list-poses
./scripts/arena/run.sh --list-model-adapters
```

## Advanced use

The standard command prepares the model server automatically. To prewarm it without launching the
simulator, or to force a rebuild, use:

```bash
./docker/container.sh start-groot
./docker/container.sh start-groot --rebuild
```

An external run configuration remains supported for automation and one-off experiments:

```bash
./scripts/arena/run.sh --config /path/to/experiment.yaml
```

N1.7 FFW-SG2 checkpoints use the same launcher. The adapter reads temporal observation offsets and
the action horizon from checkpoint metadata. The simulator client, action handshake, remote
transport, server loop, and chunk scheduler come directly from the checked-out Isaac Lab Arena
submodule. A generic NVIDIA base checkpoint is not automatically an FFW-SG2 checkpoint; its
processor must contain a supported FFW-SG2 `new_embodiment` schema. The showroom variant uses head
and both wrist cameras, 16 arm/gripper joints, and a three-axis mobile-base command. Metadata
selects its 22D mobile embodiment automatically; joint-only checkpoints use the 19D embodiment.

`run.sh` only establishes portable paths and enters the tested Python router. Robot, scene,
model-adapter, policy, profile, manifest, and workflow logic belongs in `source/cyclo_arena`;
checkpoint instances remain external files.
