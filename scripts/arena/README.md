# Cyclo Arena launcher

Choose a robot, scene, downloaded checkpoint, and robot initial pose in
`source/cyclo_arena/configs/run.yaml`:

```yaml
robot:
  name: ffw_sg2
  initial_pose: recycling

scene: robotis_showroom

model:
  checkpoint: ${CYCLO_ARENA_MODEL_ROOT}/GR00T-N1.6_SeparateRecycling_model
  adapter: auto
```

Prepare the selected GR00T server on the host, then run inference inside Cyclo Lab:

```bash
./docker/container.sh start-groot
./docker/container.sh enter

# Inside Cyclo Lab
./scripts/arena/run.sh
```

The host model workspace is `docker/workspace/model`. Docker Compose mounts it at
`/workspace/model` and sets `CYCLO_ARENA_MODEL_ROOT` automatically, so the same YAML path works in
both environments. `start-groot` reads the checkpoint type, builds either `cyclo-gr00t:n1.6` or
`cyclo-gr00t:n1.7`, starts the checkpoint server, and writes its endpoint below the shared model
workspace. Use `start-groot --rebuild` to rebuild the selected version.

The launcher reads the checkpoint metadata and verifies its observation/action schema. The prepared
server endpoint is read from the shared model workspace before the selected scene starts.
Downloading another compatible checkpoint only requires changing `model.checkpoint`, running
`start-groot` again, and starting inference; no Python registration is needed.

N1.6 and N1.7 FFW-SG2 checkpoints use the same launcher. The adapter reads temporal observation
offsets and the action horizon from the live model server. A generic NVIDIA base checkpoint is not
an FFW-SG2 checkpoint; its processor must contain the `new_embodiment` tag with the
`cam_left_head`, `arm_left`, and `arm_right` schema.

Inspect valid and discovered selections without starting Isaac Sim:

```bash
./scripts/arena/run.sh --list-robots
./scripts/arena/run.sh --list-scenes
./scripts/arena/run.sh --list-models
./scripts/arena/run.sh --list-poses
./scripts/arena/run.sh --list-model-adapters
```

`--list-models` scans `CYCLO_ARENA_MODEL_ROOT` and reports each
checkpoint's compatible adapter. CLI values override runtime values without editing the config:

```bash
./scripts/arena/run.sh --num-steps 100 --headless
./scripts/arena/run.sh --dry-run
```

Scripts in this directory remain thin entry points. Robot, scene, model-adapter, policy, and
workflow logic belongs in `source/cyclo_arena`; checkpoint instances remain external files.
