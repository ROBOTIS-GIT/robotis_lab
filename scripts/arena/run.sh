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
CYCLO_ARENA_PYTHON_ROOT="${CYCLO_ARENA_REPOSITORY_ROOT}/source/cyclo_arena"

if [[ -f /isaac-sim/python.sh ]]; then
    export CYCLO_ARENA_MODEL_ROOT="${CYCLO_ARENA_MODEL_ROOT:-/workspace/model}"
else
    export CYCLO_ARENA_MODEL_ROOT="${CYCLO_ARENA_MODEL_ROOT:-${CYCLO_ARENA_REPOSITORY_ROOT}/docker/workspace/model}"
fi
export CYCLO_ARENA_SERVER_STATE="${CYCLO_ARENA_SERVER_STATE:-${CYCLO_ARENA_MODEL_ROOT}/.cyclo_arena/server.json}"
cd -- "${CYCLO_ARENA_REPOSITORY_ROOT}"
export PYTHONPATH="${CYCLO_ARENA_PYTHON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 -m cyclo_arena.entrypoint "$@"
