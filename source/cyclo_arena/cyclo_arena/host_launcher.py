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

"""Host-side orchestration for one-command Cyclo Arena inference."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from cyclo_arena.catalog import REGISTRY
from cyclo_arena.core.config import load_run_config
from cyclo_arena.core.model_resolver import ResolvedModel, model_search_root
from cyclo_arena.core.server_state import write_server_state

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONTAINER_REPOSITORY_ROOT = Path("/workspace/cyclo_lab")
SERVER_CHECKPOINT_PATH = "/models/checkpoint"


def _run(
    command: Sequence[str],
    *,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one host command with consistent text output."""
    return subprocess.run(
        list(command),
        check=check,
        capture_output=capture_output,
        text=True,
    )


def _container_status(container_name: str) -> str | None:
    result = _run(
        [
            "docker",
            "inspect",
            "--format",
            "{{.State.Status}}",
            container_name,
        ],
        check=False,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _container_label(container_name: str, label: str) -> str | None:
    result = _run(
        [
            "docker",
            "inspect",
            "--format",
            f'{{{{ index .Config.Labels "{label}" }}}}',
            container_name,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value if value and value != "<no value>" else None


def _ensure_cyclo_container(container_name: str) -> None:
    """Start the Cyclo Lab container when necessary."""
    if _container_status(container_name) == "running":
        return
    print(f"[INFO] Starting Cyclo Lab container: {container_name}", flush=True)
    _run([str(REPOSITORY_ROOT / "docker" / "container.sh"), "start"])
    assert _container_status(container_name) == "running", (
        f"Cyclo Lab container did not start: {container_name}"
    )


def _server_container_name(model: ResolvedModel) -> str:
    """Return a deterministic Docker name for one checkpoint path."""
    slug = re.sub(r"[^a-z0-9]+", "-", model.name.lower()).strip("-")
    digest = hashlib.sha256(str(model.checkpoint).encode()).hexdigest()[:10]
    return f"cyclo-gr00t-{slug[:38]}-{digest}"


def _available_port() -> int:
    """Ask the host kernel for one currently available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return int(server_socket.getsockname()[1])


def _stop_inactive_model_servers(active_container: str) -> None:
    """Stop other managed model servers so version switches do not exhaust VRAM."""
    result = _run(
        [
            "docker",
            "ps",
            "--filter",
            "label=cyclo_arena.managed=true",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
    )
    for container_name in result.stdout.splitlines():
        if container_name and container_name != active_container:
            print(f"[INFO] Stopping inactive GR00T server: {container_name}")
            _run(["docker", "stop", container_name])


def _huggingface_root(checkpoint: Path) -> Path:
    """Return a cache root to expose alongside the exact checkpoint mount."""
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    search_root = model_search_root()
    if search_root.name == "checkpoints":
        return search_root.parent
    return search_root


def _create_server_container(
    model: ResolvedModel,
    container_name: str,
    port: int,
) -> None:
    """Create a GR00T server directly from a resolved checkpoint."""
    adapter = model.adapter
    image = _run(
        ["docker", "image", "inspect", adapter.server_image],
        check=False,
        capture_output=True,
    )
    assert image.returncode == 0, (
        f"GR00T server image {adapter.server_image!r} is unavailable. Build the "
        "checkpoint-selected GR00T runtime with "
        "./docker/container.sh start-groot."
    )
    server_program = (
        ".venv/bin/python",
        "gr00t/eval/run_gr00t_server.py",
        "--model-path",
        SERVER_CHECKPOINT_PATH,
        "--embodiment-tag",
        adapter.server_embodiment_tag,
        "--device",
        adapter.server_device,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    )
    shell_command = (
        f"cd {shlex.quote(adapter.server_workdir)} "
        f"&& exec {shlex.join(server_program)}"
    )
    huggingface_root = _huggingface_root(model.checkpoint)
    print(f"[INFO] Creating GR00T server: {container_name}", flush=True)
    _run(
        [
            "docker",
            "run",
            "--detach",
            "--name",
            container_name,
            "--gpus",
            "all",
            "--ipc",
            "host",
            "--network",
            "host",
            "--security-opt",
            "label=disable",
            "--label",
            "cyclo_arena.managed=true",
            "--label",
            f"cyclo_arena.checkpoint={model.checkpoint}",
            "--label",
            f"cyclo_arena.adapter={adapter.name}",
            "--label",
            f"cyclo_arena.server_port={port}",
            "-e",
            "HF_HOME=/models/huggingface",
            "-e",
            "HUGGINGFACE_HUB_CACHE=/models/huggingface/hub",
            "-v",
            f"{model.checkpoint}:{SERVER_CHECKPOINT_PATH}:ro",
            "-v",
            f"{huggingface_root}:/models/huggingface",
            adapter.server_image,
            "bash",
            "-lc",
            shell_command,
        ]
    )


def _ping_server(container_name: str, host: str, port: int) -> bool:
    result = _run(
        [
            "docker",
            "exec",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            container_name,
            "/isaac-sim/python.sh",
            "-m",
            "cyclo_arena.policies.gr00t_rpc",
            "--host",
            host,
            "--port",
            str(port),
            "--timeout-ms",
            "1000",
        ],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _ensure_model_server(
    cyclo_container: str,
    model: ResolvedModel,
) -> int:
    """Start a checkpoint-specific server and return its persistent port."""
    container_name = _server_container_name(model)
    _stop_inactive_model_servers(container_name)
    status = _container_status(container_name)
    if status is None:
        port = _available_port()
        _create_server_container(model, container_name, port)
    else:
        port_value = _container_label(container_name, "cyclo_arena.server_port")
        assert port_value is not None, (
            f"Existing container {container_name!r} is not a Cyclo-managed model server"
        )
        port = int(port_value)
        checkpoint = _container_label(container_name, "cyclo_arena.checkpoint")
        assert checkpoint == str(model.checkpoint), (
            f"Model server {container_name!r} points to {checkpoint!r}, not "
            f"{str(model.checkpoint)!r}"
        )
        adapter_name = _container_label(container_name, "cyclo_arena.adapter")
        assert adapter_name == model.adapter.name, (
            f"Model server {container_name!r} uses adapter {adapter_name!r}, not "
            f"{model.adapter.name!r}. Remove the stale server container and run "
            "./docker/container.sh start-groot again."
        )
        if status != "running":
            print(f"[INFO] Starting GR00T server: {container_name}", flush=True)
            _run(["docker", "start", container_name])

    print(
        f"[INFO] Waiting for {model.name} at 127.0.0.1:{port}",
        flush=True,
    )
    deadline = time.monotonic() + model.adapter.startup_timeout_seconds
    next_update = time.monotonic()
    while time.monotonic() < deadline:
        if _ping_server(cyclo_container, "127.0.0.1", port):
            print(f"[INFO] GR00T model is ready: {model.name}", flush=True)
            return port
        status = _container_status(container_name)
        if status != "running":
            logs = _run(
                ["docker", "logs", "--tail", "80", container_name],
                check=False,
                capture_output=True,
            )
            raise RuntimeError(
                f"GR00T server stopped while loading ({status}).\n"
                f"{logs.stdout}{logs.stderr}"
            )
        if time.monotonic() >= next_update:
            remaining = max(0, int(deadline - time.monotonic()))
            print(f"[INFO] GR00T model is loading ({remaining}s remaining)", flush=True)
            next_update = time.monotonic() + 10.0
        time.sleep(1.0)
    raise TimeoutError(
        f"GR00T server 127.0.0.1:{port} did not become ready within "
        f"{model.adapter.startup_timeout_seconds}s"
    )


def _launch_in_container(
    container_name: str,
    config_path: Path,
    forwarded_args: Sequence[str],
    model_adapter: str | None = None,
    remote_port: int | None = None,
) -> int:
    """Run the simulator through the repository launcher inside Cyclo Lab."""
    relative_config = config_path.resolve().relative_to(REPOSITORY_ROOT)
    container_config = CONTAINER_REPOSITORY_ROOT / relative_config
    command = ["docker", "exec"]
    if sys.stdin.isatty() and sys.stdout.isatty():
        command.append("-it")
    if os.environ.get("DISPLAY"):
        command += ["-e", f"DISPLAY={os.environ['DISPLAY']}"]
    run_args = [*forwarded_args]
    if model_adapter is not None:
        run_args += ["--resolved-model-adapter", model_adapter]
    if remote_port is not None:
        run_args += ["--remote-port", str(remote_port)]
    command += [
        "-w",
        str(CONTAINER_REPOSITORY_ROOT),
        container_name,
        str(CONTAINER_REPOSITORY_ROOT / "scripts" / "arena" / "run.sh"),
        "--inside",
        "--config",
        str(container_config),
        *run_args,
    ]
    return _run(command, check=False).returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve a checkpoint, prepare its server, and launch Cyclo Arena."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--print-server-runtime", action="store_true")
    parser.add_argument("forwarded_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    forwarded_args = list(args.forwarded_args)
    if forwarded_args[:1] == ["--"]:
        forwarded_args.pop(0)

    config_path = args.config.expanduser().resolve()
    config_path.relative_to(REPOSITORY_ROOT)
    config = load_run_config(config_path)
    model = config.resolve_model(REGISTRY)
    if args.print_server_runtime:
        assert model is not None, "A GR00T model must be selected in run.yaml"
        print(
            f"{model.adapter.server_image}\t"
            f"{model.adapter.server_source_revision}"
        )
        return 0

    assert shutil.which("docker"), "Docker CLI is required on the host"
    container_name = f"cyclo_lab{os.environ.get('DOCKER_NAME_SUFFIX', '')}"
    _ensure_cyclo_container(container_name)
    remote_port = None
    if model is not None and "--dry-run" not in forwarded_args:
        remote_port = _ensure_model_server(container_name, model)
        write_server_state(
            model,
            remote_port,
            _server_container_name(model),
        )
    if args.prepare_only:
        assert model is not None, "start-groot requires a model in run.yaml"
        assert remote_port is not None, "GR00T server was not started"
        print(
            f"[OK] GR00T server prepared at 127.0.0.1:{remote_port}",
            flush=True,
        )
        return 0
    return _launch_in_container(
        container_name,
        config_path,
        forwarded_args,
        model_adapter=model.adapter.name if model is not None else None,
        remote_port=remote_port,
    )


if __name__ == "__main__":
    raise SystemExit(main())
