#!/bin/bash

# Apply Cyclo Lab's local Isaac Lab fixes without committing inside the submodule.

set -euo pipefail

patch_dir="${CYCLOLAB_PATH}/docker/patches"

for patch_file in "${patch_dir}"/isaaclab-*.patch; do
    if [ ! -e "${patch_file}" ]; then
        continue
    fi

    if git -C "${ISAACLAB_PATH}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
        echo "[INFO] Isaac Lab patch already applied: $(basename "${patch_file}")"
        continue
    fi

    if ! git -C "${ISAACLAB_PATH}" apply --check "${patch_file}"; then
        echo "[ERROR] Isaac Lab patch does not apply cleanly: ${patch_file}" >&2
        exit 1
    fi

    git -C "${ISAACLAB_PATH}" apply "${patch_file}"
    echo "[INFO] Applied Isaac Lab patch: $(basename "${patch_file}")"
done
