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

The host command detects `Gr00tN1d6` or `Gr00tN1d7`, builds the pinned matching GR00T runtime when
needed, and prepares a checkpoint-specific server. The container command validates the FFW-SG2
processor schema, reads its temporal and action horizons, and launches simulation.

Use the component catalogs to inspect the available composition surface:

```bash
cyclo-arena list robots
cyclo-arena list scenes
cyclo-arena list tasks
cyclo-arena list models
```
