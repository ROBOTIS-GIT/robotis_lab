# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Seongwoo Kim

"""Share one prepared GR00T server endpoint between host and simulator."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from cyclo_arena.core.model_resolver import ResolvedModel

SERVER_STATE_ENVIRONMENT = "CYCLO_ARENA_SERVER_STATE"


def _state_path() -> Path | None:
    value = os.environ.get(SERVER_STATE_ENVIRONMENT)
    return Path(value).expanduser().resolve() if value else None


def _checkpoint_key(checkpoint: Path) -> str:
    model_root_value = os.environ.get("CYCLO_ARENA_MODEL_ROOT")
    assert model_root_value, "CYCLO_ARENA_MODEL_ROOT is required for server state"
    model_root = Path(model_root_value).expanduser().resolve()
    try:
        return checkpoint.resolve().relative_to(model_root).as_posix()
    except ValueError as exc:
        raise AssertionError(f"Checkpoint {checkpoint} is outside model root {model_root}") from exc


def write_server_state(
    model: ResolvedModel,
    port: int,
    container_name: str,
) -> None:
    """Persist a prepared model server endpoint when state sharing is enabled."""
    state_path = _state_path()
    if state_path is None:
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "schema_version": 1,
        "checkpoint": _checkpoint_key(model.checkpoint),
        "adapter": model.adapter.name,
        "host": "127.0.0.1",
        "port": port,
        "container": container_name,
    }
    temporary_path = state_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(state_path)


def load_server_port(model: ResolvedModel) -> int | None:
    """Return the prepared endpoint port and reject stale model state."""
    state_path = _state_path()
    if state_path is None:
        return None
    assert state_path.is_file(), (
        "GR00T server is not prepared. Run ./scripts/arena/run.sh on the host, "
        "or prewarm it with ./docker/container.sh start-groot."
    )
    values: Any = json.loads(state_path.read_text(encoding="utf-8"))
    assert isinstance(values, Mapping), f"Invalid GR00T server state: {state_path}"
    expected_checkpoint = _checkpoint_key(model.checkpoint)
    assert values.get("checkpoint") == expected_checkpoint, (
        f"Prepared GR00T server uses {values.get('checkpoint')!r}, but the selected profile or config "
        f"uses {expected_checkpoint!r}. Run ./scripts/arena/run.sh on the host again."
    )
    assert values.get("adapter") == model.adapter.name, (
        f"Prepared GR00T adapter {values.get('adapter')!r} does not match "
        f"{model.adapter.name!r}. Run ./scripts/arena/run.sh on the host again."
    )
    port = int(values["port"])
    assert 0 < port < 65536, f"Invalid GR00T server port in {state_path}: {port}"
    return port
