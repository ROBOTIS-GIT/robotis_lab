#!/usr/bin/env bash

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

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

CYCLO_ARENA_SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CYCLO_ARENA_REPOSITORY_ROOT="$(cd -- "${CYCLO_ARENA_SCRIPT_ROOT}/../.." && pwd)"
CYCLO_ARENA_DEFAULT_CONFIG="${CYCLO_ARENA_REPOSITORY_ROOT}/source/cyclo_arena/configs/run.yaml"
CYCLO_ARENA_PYTHON_ROOT="${CYCLO_ARENA_REPOSITORY_ROOT}/source/cyclo_arena"

if [[ -f /isaac-sim/python.sh ]]; then
    export CYCLO_ARENA_MODEL_ROOT="${CYCLO_ARENA_MODEL_ROOT:-/workspace/model}"
else
    export CYCLO_ARENA_MODEL_ROOT="${CYCLO_ARENA_MODEL_ROOT:-${CYCLO_ARENA_REPOSITORY_ROOT}/docker/workspace/model}"
fi
export CYCLO_ARENA_SERVER_STATE="${CYCLO_ARENA_SERVER_STATE:-${CYCLO_ARENA_MODEL_ROOT}/.cyclo_arena/server.json}"

print_usage() {
    cat <<'EOF'
Usage:
  ./scripts/arena/run.sh [cyclo-arena run overrides]
  ./scripts/arena/run.sh --config <config.yaml> [overrides]
  ./scripts/arena/run.sh --list-robots|--list-scenes|--list-models
  ./scripts/arena/run.sh --list-poses|--list-model-adapters

Examples:
  ./scripts/arena/run.sh
  ./scripts/arena/run.sh --num-steps 10 --headless
  ./scripts/arena/run.sh --scene kitchen --dry-run

The no-argument command reads source/cyclo_arena/configs/run.yaml. Prepare its
GR00T server once on the host with ./docker/container.sh start-groot, then run
this script inside the Cyclo Lab container.
Models are stored under docker/workspace/model on the host and mounted at
/workspace/model in the Cyclo Lab container.
EOF
}

list_catalog() {
    local category="$1"
    PYTHONPATH="${CYCLO_ARENA_PYTHON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        python3 -m cyclo_arena.cli list "${category}"
}

check_display_access() {
    local argument
    local requires_display=true

    for argument in "$@"; do
        case "${argument}" in
            --headless|--dry-run)
                requires_display=false
                ;;
            --windowed)
                requires_display=true
                ;;
        esac
    done

    if [[ "${requires_display}" != true ]]; then
        return 0
    fi
    if [[ -z "${DISPLAY:-}" ]]; then
        echo "[ERROR] DISPLAY is not set; the Isaac Sim window cannot be created." >&2
        echo "[INFO] Re-enter with: ./docker/container.sh enter" >&2
        exit 1
    fi
    if [[ -n "${XAUTHORITY:-}" && ! -r "${XAUTHORITY}" ]]; then
        echo "[ERROR] XAUTHORITY does not exist or is unreadable: ${XAUTHORITY}" >&2
        echo "[INFO] On the host, recreate the container to refresh X11 authentication:" >&2
        echo "       ./docker/container.sh recreate" >&2
        echo "       ./docker/container.sh enter" >&2
        exit 1
    fi
}

CYCLO_ARENA_CONFIG_PATH="${CYCLO_ARENA_DEFAULT_CONFIG}"
CYCLO_ARENA_INSIDE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --inside)
            CYCLO_ARENA_INSIDE=true
            shift
            ;;
        --config)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --config requires a YAML path." >&2
                exit 2
            fi
            CYCLO_ARENA_CONFIG_PATH="$2"
            shift 2
            ;;
        --list-robots)
            list_catalog robots
            exit 0
            ;;
        --list-scenes)
            list_catalog scenes
            exit 0
            ;;
        --list-models)
            list_catalog models
            exit 0
            ;;
        --list-model-adapters)
            list_catalog model-adapters
            exit 0
            ;;
        --list-poses)
            list_catalog poses
            exit 0
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

if [[ ! -f "${CYCLO_ARENA_CONFIG_PATH}" ]]; then
    echo "[ERROR] Cyclo Arena config does not exist: ${CYCLO_ARENA_CONFIG_PATH}" >&2
    exit 2
fi

if [[
    "${CYCLO_ARENA_INSIDE}" != true
    && !("${CYCLOLAB_PATH:-}" == "${CYCLO_ARENA_REPOSITORY_ROOT}" && -f /isaac-sim/python.sh)
]]; then
    cd -- "${CYCLO_ARENA_REPOSITORY_ROOT}"
    export PYTHONPATH="${CYCLO_ARENA_PYTHON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    exec python3 -m cyclo_arena.host_launcher \
        --config "${CYCLO_ARENA_CONFIG_PATH}" \
        -- "$@"
fi

if ! command -v cyclo-arena >/dev/null 2>&1; then
    echo "[ERROR] cyclo-arena is not installed in this shell." >&2
    echo "[INFO] Enter the Cyclo Lab container first: ./docker/container.sh enter" >&2
    exit 127
fi

check_display_access "$@"

cd -- "${CYCLO_ARENA_REPOSITORY_ROOT}"
exec cyclo-arena run --config "${CYCLO_ARENA_CONFIG_PATH}" "$@"
