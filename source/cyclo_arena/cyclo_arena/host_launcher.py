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
import tempfile
import time
from pathlib import Path
from typing import Mapping, Sequence

from cyclo_arena.catalog import REGISTRY
from cyclo_arena.core.config import load_run_config
from cyclo_arena.core.manifest import ResolvedManifest
from cyclo_arena.core.model_resolver import (
    MODEL_ROOT_ENVIRONMENT,
    ResolvedModel,
    model_search_root,
)
from cyclo_arena.core.profile_store import DEFAULT_PROFILE_ID, ProfileStore
from cyclo_arena.core.server_state import write_server_state

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONTAINER_REPOSITORY_ROOT = Path("/workspace/cyclo_lab")
CONTAINER_MODEL_ROOT = Path("/workspace/model")
SERVER_CHECKPOINT_PATH = "/models/checkpoint"
SERVER_CYCLO_ARENA_ROOT = "/opt/cyclo_arena"
SERVER_ISAACLAB_ARENA_ROOT = "/opt/isaaclab_arena"
SERVER_PROTOCOL_VERSION = "arena-remote-policy-v1"
GROOT_REPOSITORY_URL = "https://github.com/NVIDIA/Isaac-GR00T.git"


def _run(
    command: Sequence[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one host command with consistent text output."""
    return subprocess.run(
        list(command),
        check=check,
        capture_output=capture_output,
        env=dict(env) if env is not None else None,
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


def _arena_revision() -> str:
    """Return the Arena checkout revision mounted into model servers."""
    arena_root = REPOSITORY_ROOT / "third_party" / "IsaacLab-Arena"
    result = _run(
        ["git", "-C", str(arena_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
    )
    assert result.returncode == 0, f"Isaac Lab Arena submodule is unavailable: {arena_root}"
    return result.stdout.strip()


def _cyclo_arena_fingerprint() -> str:
    """Return a digest of the Cyclo Arena Python mounted into model servers."""
    source_root = REPOSITORY_ROOT / "source" / "cyclo_arena" / "cyclo_arena"
    digest = hashlib.sha256()
    for source_path in sorted(source_root.rglob("*.py")):
        digest.update(source_path.relative_to(source_root).as_posix().encode())
        digest.update(source_path.read_bytes())
    return digest.hexdigest()


def _ensure_server_image(model: ResolvedModel, *, force_rebuild: bool = False) -> bool:
    """Build the checkpoint-selected GR00T image when it is unavailable or stale."""
    adapter = model.adapter
    revision = adapter.server_source_revision
    assert revision, f"Model adapter {adapter.name!r} has no GR00T source revision"
    image = _run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            '{{ index .Config.Labels "cyclo_arena.gr00t_revision" }}',
            adapter.server_image,
        ],
        check=False,
        capture_output=True,
    )
    if image.returncode == 0 and image.stdout.strip() == revision and not force_rebuild:
        return False

    if image.returncode == 0 and not force_rebuild:
        print(f"[INFO] Rebuilding stale GR00T image: {adapter.server_image}", flush=True)
    with tempfile.TemporaryDirectory(prefix="cyclo-groot-build-") as build_directory:
        build_root = Path(build_directory)
        source_checkout = build_root / "source"
        _run(["git", "init", "--quiet", str(source_checkout)])
        _run(["git", "-C", str(source_checkout), "remote", "add", "origin", GROOT_REPOSITORY_URL])
        print(f"[INFO] Fetching Isaac-GR00T revision: {revision}", flush=True)
        _run([
            "git",
            "-C",
            str(source_checkout),
            "fetch",
            "--quiet",
            "--depth",
            "1",
            "origin",
            revision,
        ])
        checkout_environment = os.environ.copy()
        checkout_environment["GIT_LFS_SKIP_SMUDGE"] = "1"
        _run(
            ["git", "-C", str(source_checkout), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
            env=checkout_environment,
        )

        source_dockerfile = source_checkout / "docker" / "Dockerfile"
        assert source_dockerfile.is_file(), f"Isaac-GR00T Dockerfile is unavailable at {revision}"
        if "COPY src/gr00t" in source_dockerfile.read_text(encoding="utf-8"):
            build_context = build_root / "context"
            build_context.mkdir()
            dockerfile = build_context / "Dockerfile"
            shutil.copy2(source_dockerfile, dockerfile)
            shutil.copytree(
                source_checkout,
                build_context / "src" / "gr00t",
                ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", "logs"),
            )
        else:
            build_context = source_checkout
            dockerfile = source_dockerfile

        buildx = _run(["docker", "buildx", "version"], check=False, capture_output=True)
        if buildx.returncode == 0:
            command = ["docker", "buildx", "build", "--load"]
        else:
            command = ["docker", "build"]
        command += [
            "--platform",
            "linux/amd64",
            "--network",
            "host",
            "--label",
            f"cyclo_arena.gr00t_revision={revision}",
            "-f",
            str(dockerfile),
            "-t",
            adapter.server_image,
            str(build_context),
        ]
        print(f"[INFO] Building isolated GR00T image: {adapter.server_image}", flush=True)
        _run(command)
    return True


def _ensure_cyclo_container(container_name: str) -> None:
    """Start the Cyclo Lab container when necessary."""
    if _container_status(container_name) == "running":
        return
    print(f"[INFO] Starting Cyclo Lab container: {container_name}", flush=True)
    _run([str(REPOSITORY_ROOT / "docker" / "container.sh"), "start"])
    assert _container_status(container_name) == "running", f"Cyclo Lab container did not start: {container_name}"


def _server_container_name(model: ResolvedModel) -> str:
    """Return a deterministic Docker name for one checkpoint path."""
    slug = re.sub(r"[^a-z0-9]+", "-", model.name.lower()).strip("-")
    identity = f"{model.checkpoint}|{SERVER_PROTOCOL_VERSION}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:10]
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
    default_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    if (default_cache / "token").is_file():
        return default_cache
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
    server_program = (
        ".venv/bin/python",
        "-m",
        "isaaclab_arena.remote_policy.remote_policy_server_runner",
        "--policy_type",
        "cyclo_arena.policies.gr00t_server.CycloGr00tServerSidePolicy",
        "--model_path",
        SERVER_CHECKPOINT_PATH,
        "--robot_adapter",
        adapter.server_robot_adapter,
        "--embodiment_tag",
        adapter.server_embodiment_tag,
        "--device",
        adapter.server_device,
        "--action_repeat",
        str(adapter.action_repeat),
        "--action_chunk_length",
        str(adapter.action_chunk_length),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    )
    shell_command = f"cd {shlex.quote(adapter.server_workdir)} && exec {shlex.join(server_program)}"
    huggingface_root = _huggingface_root(model.checkpoint)
    print(f"[INFO] Creating GR00T server: {container_name}", flush=True)
    _run([
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
        "--label",
        f"cyclo_arena.server_protocol={SERVER_PROTOCOL_VERSION}",
        "--label",
        f"cyclo_arena.arena_revision={_arena_revision()}",
        "--label",
        f"cyclo_arena.source_fingerprint={_cyclo_arena_fingerprint()}",
        "-e",
        "HF_HOME=/models/huggingface",
        "-e",
        "HUGGINGFACE_HUB_CACHE=/models/huggingface/hub",
        "-e",
        f"PYTHONPATH={SERVER_CYCLO_ARENA_ROOT}:{SERVER_ISAACLAB_ARENA_ROOT}",
        "-v",
        f"{model.checkpoint}:{SERVER_CHECKPOINT_PATH}:ro",
        "-v",
        f"{huggingface_root}:/models/huggingface",
        "-v",
        f"{REPOSITORY_ROOT / 'source' / 'cyclo_arena'}:{SERVER_CYCLO_ARENA_ROOT}:ro",
        "-v",
        f"{REPOSITORY_ROOT / 'third_party' / 'IsaacLab-Arena'}:{SERVER_ISAACLAB_ARENA_ROOT}:ro",
        adapter.server_image,
        "bash",
        "-lc",
        shell_command,
    ])


def _ping_server(container_name: str, host: str, port: int) -> bool:
    probe = (
        "from isaaclab_arena.remote_policy.policy_client import PolicyClient;"
        "from isaaclab_arena.remote_policy.remote_policy_config import "
        "RemotePolicyConfig;"
        f"c=PolicyClient(RemotePolicyConfig(host={host!r},port={port},"
        "timeout_ms=1000));"
        "raise SystemExit(0 if c.ping() else 1)"
    )
    result = _run(
        [
            "docker",
            "exec",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            container_name,
            "/isaac-sim/python.sh",
            "-c",
            probe,
        ],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _ensure_model_server(
    cyclo_container: str,
    model: ResolvedModel,
    *,
    rebuild_image: bool = False,
) -> int:
    """Start a checkpoint-specific server and return its persistent port."""
    image_rebuilt = _ensure_server_image(model, force_rebuild=rebuild_image)
    container_name = _server_container_name(model)
    _stop_inactive_model_servers(container_name)
    status = _container_status(container_name)
    if image_rebuilt and status is not None:
        print(f"[INFO] Recreating GR00T server for rebuilt image: {container_name}", flush=True)
        _run(["docker", "rm", "--force", container_name])
        status = None
    if status is not None:
        server_arena_revision = _container_label(container_name, "cyclo_arena.arena_revision")
        if server_arena_revision != _arena_revision():
            print(
                f"[INFO] Recreating GR00T server for the updated Arena checkout: {container_name}",
                flush=True,
            )
            _run(["docker", "rm", "--force", container_name])
            status = None
        elif _container_label(container_name, "cyclo_arena.source_fingerprint") != _cyclo_arena_fingerprint():
            print(
                f"[INFO] Recreating GR00T server for updated Cyclo Arena code: {container_name}",
                flush=True,
            )
            _run(["docker", "rm", "--force", container_name])
            status = None
    if status is None:
        port = _available_port()
        _create_server_container(model, container_name, port)
    else:
        port_value = _container_label(container_name, "cyclo_arena.server_port")
        assert port_value is not None, f"Existing container {container_name!r} is not a Cyclo-managed model server"
        port = int(port_value)
        checkpoint = _container_label(container_name, "cyclo_arena.checkpoint")
        assert checkpoint == str(
            model.checkpoint
        ), f"Model server {container_name!r} points to {checkpoint!r}, not {str(model.checkpoint)!r}"
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
            raise RuntimeError(f"GR00T server stopped while loading ({status}).\n{logs.stdout}{logs.stderr}")
        if time.monotonic() >= next_update:
            remaining = max(0, int(deadline - time.monotonic()))
            print(f"[INFO] GR00T model is loading ({remaining}s remaining)", flush=True)
            next_update = time.monotonic() + 10.0
        time.sleep(1.0)
    raise TimeoutError(
        f"GR00T server 127.0.0.1:{port} did not become ready within {model.adapter.startup_timeout_seconds}s"
    )


def _model_workspace_root() -> Path:
    """Return the host directory mounted at /workspace/model."""
    configured = os.environ.get(MODEL_ROOT_ENVIRONMENT)
    root = Path(configured) if configured else REPOSITORY_ROOT / "docker" / "workspace" / "model"
    return root.expanduser().resolve()


def _persist_manifest(manifest: ResolvedManifest) -> tuple[Path, Path]:
    """Persist one process-boundary manifest and return host/container paths."""
    host_root = _model_workspace_root()
    relative_path = Path(".cyclo_arena") / "manifests" / f"{manifest.fingerprint}.json"
    host_path = manifest.write(host_root / relative_path)
    return host_path, CONTAINER_MODEL_ROOT / relative_path


def _launch_in_container(
    container_name: str,
    manifest: ResolvedManifest,
    forwarded_args: Sequence[str],
) -> int:
    """Run one resolved manifest through the repository launcher inside Cyclo Lab."""
    _, container_manifest = _persist_manifest(manifest)
    run_args = list(forwarded_args)
    visualization_overridden = any(argument in {"--headless", "--windowed"} for argument in run_args)
    if manifest.run_values.get("headless") and not visualization_overridden:
        run_args.insert(0, "--headless")
    command = ["docker", "exec"]
    if sys.stdin.isatty() and sys.stdout.isatty():
        command.append("-it")
    if os.environ.get("DISPLAY"):
        command += ["-e", f"DISPLAY={os.environ['DISPLAY']}"]
    command += [
        "-w",
        str(CONTAINER_REPOSITORY_ROOT),
        container_name,
        str(CONTAINER_REPOSITORY_ROOT / "scripts" / "arena" / "run.sh"),
        "--inside",
        "--manifest",
        str(container_manifest),
        *run_args,
    ]
    return _run(command, check=False).returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve a checkpoint, prepare its server, and launch Cyclo Arena."""
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path)
    source.add_argument("--profile", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--print-server-runtime", action="store_true")
    parser.add_argument("--rebuild-server-image", action="store_true")
    parser.add_argument("forwarded_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    forwarded_args = list(args.forwarded_args)
    if forwarded_args[:1] == ["--"]:
        forwarded_args.pop(0)

    if args.config is not None:
        manifest = ResolvedManifest.from_run_config(load_run_config(args.config), REGISTRY)
    else:
        manifest = ProfileStore().resolve(args.profile or DEFAULT_PROFILE_ID, REGISTRY)
    model = manifest.model.to_resolved_model(REGISTRY) if manifest.model is not None else None
    if args.print_server_runtime:
        assert model is not None, "The selected profile or config must contain a GR00T model"
        print(f"{model.adapter.server_image}\t{model.adapter.server_source_revision}")
        return 0

    assert not args.rebuild_server_image or model is not None, "A GR00T model is required to rebuild its image"
    assert shutil.which("docker"), "Docker CLI is required on the host"
    container_name = f"cyclo_lab{os.environ.get('DOCKER_NAME_SUFFIX', '')}"
    _ensure_cyclo_container(container_name)
    remote_port = None
    if model is not None and "--dry-run" not in forwarded_args:
        remote_port = _ensure_model_server(
            container_name,
            model,
            rebuild_image=args.rebuild_server_image,
        )
        write_server_state(
            model,
            remote_port,
            _server_container_name(model),
        )
    if args.prepare_only:
        assert model is not None, "start-groot requires a model in the selected profile or config"
        assert remote_port is not None, "GR00T server was not started"
        print(
            f"[OK] GR00T server prepared at 127.0.0.1:{remote_port}",
            flush=True,
        )
        return 0
    if remote_port is not None:
        manifest = manifest.with_run_overrides(remote_host="127.0.0.1", remote_port=remote_port)
    return _launch_in_container(
        container_name,
        manifest,
        forwarded_args,
    )


if __name__ == "__main__":
    raise SystemExit(main())
