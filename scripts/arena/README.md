# Cyclo Arena launcher

Run Cyclo Arena inference from the **Cyclo Lab host checkout**.

## Quick start

bash
cd ~/cyclo_lab
./scripts/arena/run.sh



The default command uses the `ffw_sg2_gr00t` profile and automatically prepares
the Cyclo Lab container, matching GR00T server, and Isaac Sim. Do not enter the
container or run `start-groot` first. The default opens the simulator window;
add `--headless` to run without it.

## Configuration

For normal inference, edit only these two files:

| Purpose | File |
| --- | --- |
| Robot, scene, model, instruction, and runtime | `source/cyclo_arena/configs/profiles/ffw_sg2_gr00t.yaml` |
| FFW-SG2 initial joint pose | `source/cyclo_arena/configs/robots/ffw_sg2/poses/showroom.yaml` |

Store checkpoints under `docker/workspace/model/<checkpoint-name>`. This directory
is mounted at `/workspace/model` in Docker and its contents are ignored by Git.
Reference a checkpoint portably in the profile:

```yaml
model:
  checkpoint: ${CYCLO_ARENA_MODEL_ROOT}/showroom_groot
  adapter: auto
```

Adding a compatible checkpoint requires no Python catalog entry.

## Common commands

| Command | Purpose |
| --- | --- |
| `./scripts/arena/run.sh` | Run the default profile |
| `./scripts/arena/run.sh infer ffw_sg2_gr00t` | Run a named profile |
| `./scripts/arena/run.sh --scene kitchen --headless` | Temporarily override profile values |
| `./scripts/arena/run.sh --num-steps 100` | Limit the rollout length |
| `./scripts/arena/run.sh --dry-run` | Preview the generated Arena command |
| `./scripts/arena/run.sh show profile ffw_sg2_gr00t` | Print the profile and source path |
| `./scripts/arena/run.sh validate ffw_sg2_gr00t` | Validate the profile and checkpoint |
| `./scripts/arena/run.sh plan ffw_sg2_gr00t` | Print the resolved execution settings |

Robot-pose overrides follow the same pattern:

```bash
./scripts/arena/run.sh \
  --robot-position-xyz 0 0 0 \
  --robot-yaw -1.5708 \
  --head-position 0.5 0 \
  --lift-position -0.2
```

CLI overrides affect only the current run. The default `showroom_groot` model was
trained with `robotis_showroom_training`; behavior in other scenes is out of
distribution and may not succeed.

List available resources with `list`:

```bash
./scripts/arena/run.sh list scenes
```

Valid categories are `robots`, `scenes`, `models`, `poses`, `profiles`,
`model-adapters`, and `workflows`.

## Advanced commands

```bash
# Prepare the default GR00T server without launching Isaac Sim
./docker/container.sh start-groot

# Force the pinned GR00T image to rebuild
./docker/container.sh start-groot --rebuild

# Run an external configuration
./scripts/arena/run.sh --config /path/to/experiment.yaml

# Show all CLI options
./scripts/arena/run.sh --help```