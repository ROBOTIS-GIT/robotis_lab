# Cyclo Arena

`cyclo_arena` is the ROBOTIS extension package for Isaac Lab Arena. It owns Arena-specific
environment composition, embodiments, tasks, policies, workflow configuration, and the
`cyclo-arena` command. Robot models, sensors, USDs, and joint contracts remain in `cyclo_lab` and
are reused without copying.

The package keeps simulator-independent registrations under `cyclo_arena.core`. A typed registry
resolves a robot, scene, and task into a composition plan; runtime adapters then create the Isaac
Lab objects only after Isaac Sim starts.

For a one-command inference run, select `robot`, `scene`, and a downloaded checkpoint in
`configs/run.yaml`, prepare its model server on the host, then launch inside Cyclo Lab:

```bash
./docker/container.sh start-groot
./docker/container.sh enter
./scripts/arena/run.sh
```

The host command accepts `Gr00tN1d7`, builds the pinned GR00T N1.7 runtime when needed, and
prepares a checkpoint-specific server. Both sides use Arena's native remote-policy
stack: `ActionChunkingClientSidePolicy`, `ActionProtocol`, `PolicyClient`, `PolicyServer`, and
`remote_policy_server_runner`. Cyclo only supplies the robot-specific observation/action adapter
needed to connect an FFW-SG2 checkpoint to that generic Arena contract.

This boundary is deliberate. Updating the Arena submodule immediately updates the policy runner,
remote transport, chunk scheduler, environment builder, evaluation, teleoperation, recording,
Mimic, and replay entry points used by Cyclo. Robot-specific code stays below
`cyclo_arena/policies/adapters/`; adding BG2, SH5, or OMY does not require another RPC client or a
fork of Arena's inference code.

Update the configured Arena compatibility branch and restart the selected server with:

```bash
git submodule update --remote third_party/IsaacLab-Arena
./docker/container.sh start-groot
```

The launcher records the mounted Arena revision and recreates the disposable model-server
container when that revision changes. Checkpoint data remains in `docker/workspace/model`.

Use the component catalogs to inspect the available composition surface:

```bash
cyclo-arena list robots
cyclo-arena list scenes
cyclo-arena list tasks
cyclo-arena list models
```
