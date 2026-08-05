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

"""Route the thin Arena shell script to host orchestration or the container CLI."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from cyclo_arena import cli, host_launcher
from cyclo_arena.core.profile_store import DEFAULT_PROFILE_ID

ISAAC_SIM_PYTHON = Path("/isaac-sim/python.sh")
STATIC_COMMANDS = frozenset({"list", "show", "plan", "validate"})
LEGACY_LIST_COMMANDS = {
    "--list-robots": "robots",
    "--list-scenes": "scenes",
    "--list-models": "models",
    "--list-model-adapters": "model-adapters",
    "--list-poses": "poses",
}

USAGE = """Usage:
  ./scripts/arena/run.sh [inference overrides]
  ./scripts/arena/run.sh infer [profile] [overrides]
  ./scripts/arena/run.sh --config /path/to/experiment.yaml [overrides]
  ./scripts/arena/run.sh list profiles|workflows|robots|scenes|models
  ./scripts/arena/run.sh show profile [profile]
  ./scripts/arena/run.sh plan [profile]
  ./scripts/arena/run.sh validate [profile]

Examples:
  ./scripts/arena/run.sh
  ./scripts/arena/run.sh infer ffw_sg2_gr00t
  ./scripts/arena/run.sh list profiles
  ./scripts/arena/run.sh --num-steps 10 --headless
  ./scripts/arena/run.sh --scene kitchen --dry-run

The no-argument command uses the default named profile. Profiles hide config
paths and are available through `infer <profile>`.
Models are stored under docker/workspace/model on the host and mounted at
/workspace/model in the Cyclo Lab container.
"""


@dataclass(frozen=True)
class LaunchRequest:
    """Describe one parsed invocation of the user-facing Arena script."""

    inside: bool
    source_kind: str
    source: str
    forwarded_args: tuple[str, ...] = ()
    static_args: tuple[str, ...] = ()


def _set_source(current_kind: str, explicit: bool, new_kind: str) -> None:
    """Reject ambiguous source selections while allowing the default to be replaced."""
    assert not explicit or current_kind == new_kind, "Select only one profile, --config, or --manifest"


def parse_launch_request(argv: Sequence[str]) -> LaunchRequest:
    """Parse wrapper options while leaving inference overrides untouched."""
    arguments = list(argv)
    inside = False
    source_kind = "profile"
    source = DEFAULT_PROFILE_ID
    source_explicit = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--inside":
            inside = True
            index += 1
            continue
        if argument in {"--config", "--profile", "--manifest"}:
            assert index + 1 < len(arguments), f"{argument} requires a value"
            new_kind = argument.removeprefix("--")
            _set_source(source_kind, source_explicit, new_kind)
            source_kind = new_kind
            source = arguments[index + 1]
            source_explicit = True
            index += 2
            continue
        if argument in {"infer", "run"}:
            index += 1
            if index < len(arguments) and not arguments[index].startswith("-"):
                _set_source(source_kind, source_explicit, "profile")
                source_kind = "profile"
                source = arguments[index]
                source_explicit = True
                index += 1
            continue
        if argument in STATIC_COMMANDS:
            return LaunchRequest(
                inside=inside,
                source_kind=source_kind,
                source=source,
                static_args=tuple(arguments[index:]),
            )
        if argument in LEGACY_LIST_COMMANDS:
            return LaunchRequest(
                inside=inside,
                source_kind=source_kind,
                source=source,
                static_args=("list", LEGACY_LIST_COMMANDS[argument]),
            )
        if argument in {"-h", "--help"}:
            return LaunchRequest(
                inside=inside,
                source_kind=source_kind,
                source=source,
                static_args=("help",),
            )
        break
    return LaunchRequest(
        inside=inside,
        source_kind=source_kind,
        source=source,
        forwarded_args=tuple(arguments[index:]),
    )


def _source_args(request: LaunchRequest, *, container_cli: bool) -> list[str]:
    """Return source arguments for the host launcher or composed container CLI."""
    if request.source_kind == "profile":
        return [request.source] if container_cli else ["--profile", request.source]
    return [f"--{request.source_kind}", request.source]


def _check_display_access(forwarded_args: Sequence[str]) -> None:
    """Fail early when a windowed container run has no usable X11 display."""
    requires_display = True
    for argument in forwarded_args:
        if argument in {"--headless", "--dry-run"}:
            requires_display = False
        elif argument == "--windowed":
            requires_display = True
    if not requires_display:
        return
    assert os.environ.get("DISPLAY"), "DISPLAY is not set; re-enter with ./docker/container.sh enter or use --headless"
    xauthority = os.environ.get("XAUTHORITY")
    assert not xauthority or (
        Path(xauthority).is_file() and os.access(xauthority, os.R_OK)
    ), f"XAUTHORITY does not exist or is unreadable: {xauthority}. Recreate and re-enter the Cyclo Lab container."


def main(argv: Sequence[str] | None = None) -> int:
    """Run the host/container routing for scripts/arena/run.sh."""
    request = parse_launch_request(sys.argv[1:] if argv is None else argv)
    if request.static_args:
        if request.static_args == ("help",):
            print(USAGE, end="")
            return 0
        return cli.main(request.static_args)

    if request.source_kind in {"config", "manifest"}:
        source_path = Path(request.source).expanduser()
        assert source_path.is_file(), f"Cyclo Arena {request.source_kind} does not exist: {source_path}"

    inside = request.inside or ISAAC_SIM_PYTHON.is_file()
    if not inside:
        assert request.source_kind != "manifest", "--manifest is an internal container option"
        return host_launcher.main([
            *_source_args(request, container_cli=False),
            "--",
            *request.forwarded_args,
        ])

    _check_display_access(request.forwarded_args)
    return cli.main([
        "infer",
        *_source_args(request, container_cli=True),
        *request.forwarded_args,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
