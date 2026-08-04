#!/usr/bin/env bash

# Copyright (c) 2024, Cyclo Lab Project Developers.
# All rights reserved.
#
# Author: Seongwoo Kim
#
# Based on Isaac Lab container management script

#==
# Configurations
#==

# Exits if error occurs
set -e

# Set tab-spaces when the terminal supports it.
tabs 4 2>/dev/null || true

# get source directory
export CYCLOLAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
export DOCKER_DIR="${CYCLOLAB_PATH}/docker"
GROOT_REPOSITORY_URL="https://github.com/NVIDIA/Isaac-GR00T.git"
CYCLO_ARENA_RUN_CONFIG="${CYCLOLAB_PATH}/source/cyclo_arena/configs/run.yaml"

#==
# Helper functions
#==

# print the usage description
print_help() {
    echo -e "\nusage: $(basename "$0") [-h] <command> [<args>]"
    echo -e "\nCyclo Lab Docker Container Management Script"
    echo -e "\noptional arguments:"
    echo -e "  -h, --help           Display this help message."
    echo ""
    echo -e "commands:"
    echo -e "  build                Build the docker image for Cyclo Lab"
    echo -e "  start                Start the docker container"
    echo -e "  start-groot          Auto-select, build, and start the run.yaml GR00T runtime"
    echo -e "                       Supports N1.7 checkpoint metadata"
    echo -e "                       Add --rebuild to rebuild the selected version"
    echo -e "  recreate             Recreate the container from the current image"
    echo -e "  enter                Enter the running docker container"
    echo -e "  stop                 Stop the docker container"
    echo -e "  clean                Remove the docker container and image"
    echo -e "  logs                 Show logs from the container"
    echo ""
}

# Load environment variables
load_env() {
    if [ -f "${DOCKER_DIR}/.env.base" ]; then
        set -a
        source "${DOCKER_DIR}/.env.base"
        set +a
        echo "[INFO] Loaded environment from .env.base"
    else
        echo "[ERROR] .env.base file not found in ${DOCKER_DIR}"
        exit 1
    fi
}

# Create host directories required by bind mounts.
prepare_workspace() {
    mkdir -p "${DOCKER_DIR}/workspace/model"
}

# Print the image and pinned source revision selected by run.yaml.
resolve_groot_runtime() {
    CYCLO_ARENA_MODEL_ROOT="${DOCKER_DIR}/workspace/model" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${CYCLOLAB_PATH}/source/cyclo_arena${PYTHONPATH:+:${PYTHONPATH}}" \
        python3 -m cyclo_arena.host_launcher \
        --config "${CYCLO_ARENA_RUN_CONFIG}" \
        --print-server-runtime
}

# Build one isolated GR00T version without changing the checked-out submodules.
build_groot_image() {
    local image="$1"
    local source_revision="$2"
    local force_rebuild="${3:-false}"
    local build_root
    local source_checkout
    local build_context
    local dockerfile
    local build_status
    local -a build_command

    if [ "${force_rebuild}" != "true" ] && docker image inspect "${image}" &> /dev/null; then
        echo "[INFO] GR00T image already exists: ${image}"
        return 0
    fi

    build_root="$(mktemp -d /tmp/cyclo-groot-build.XXXXXX)"
    source_checkout="${build_root}/source"
    git init --quiet "${source_checkout}"
    git -C "${source_checkout}" remote add origin "${GROOT_REPOSITORY_URL}"
    echo "[INFO] Fetching Isaac-GR00T revision: ${source_revision}"
    git -C "${source_checkout}" fetch --quiet --depth 1 origin "${source_revision}"
    # The Docker build needs source code only. Avoid downloading large Git LFS
    # media assets while checking out the pinned Isaac-GR00T revision.
    GIT_LFS_SKIP_SMUDGE=1 \
        git -C "${source_checkout}" checkout --quiet --detach FETCH_HEAD

    if [ ! -f "${source_checkout}/docker/Dockerfile" ]; then
        echo "[ERROR] Isaac-GR00T Dockerfile is unavailable at ${source_revision}" >&2
        rm -rf "${build_root}"
        exit 1
    fi

    if grep -q 'COPY src/gr00t' "${source_checkout}/docker/Dockerfile"; then
        build_context="${build_root}/context"
        dockerfile="${build_context}/Dockerfile"
        mkdir -p "${build_context}/src/gr00t"
        cp "${source_checkout}/docker/Dockerfile" "${dockerfile}"
        tar \
            --exclude='./.git' \
            --exclude='./.venv' \
            --exclude='*/__pycache__' \
            --exclude='./logs' \
            -C "${source_checkout}" -cf - . \
            | tar -C "${build_context}/src/gr00t" -xf -
    else
        build_context="${source_checkout}"
        dockerfile="${source_checkout}/docker/Dockerfile"
    fi

    echo "[INFO] Building isolated GR00T image: ${image}"
    if docker buildx version &> /dev/null; then
        build_command=(
            docker buildx build
            --load
            --platform linux/amd64
            --network host
            --label "cyclo_arena.gr00t_revision=${source_revision}"
            -f "${dockerfile}"
            -t "${image}"
            "${build_context}"
        )
    else
        build_command=(
            docker build
            --platform linux/amd64
            --network host
            --label "cyclo_arena.gr00t_revision=${source_revision}"
            -f "${dockerfile}"
            -t "${image}"
            "${build_context}"
        )
    fi
    set +e
    "${build_command[@]}"
    build_status=$?
    set -e
    rm -rf "${build_root}"
    if [ ${build_status} -ne 0 ]; then
        echo "[ERROR] Failed to build GR00T image: ${image}" >&2
        exit ${build_status}
    fi
}

# Prepare the model server selected by source/cyclo_arena/configs/run.yaml.
start_groot() {
    local force_rebuild=false
    local runtime_info
    local groot_image
    local groot_source_revision
    if [ "${1:-}" = "--rebuild" ]; then
        force_rebuild=true
    elif [ -n "${1:-}" ]; then
        echo "[ERROR] Unknown start-groot option: $1" >&2
        echo "[INFO] Supported option: --rebuild" >&2
        exit 2
    fi

    prepare_workspace
    runtime_info="$(resolve_groot_runtime)"
    IFS=$'\t' read -r groot_image groot_source_revision <<< "${runtime_info}"
    if [ -z "${groot_image}" ] || [ -z "${groot_source_revision}" ]; then
        echo "[ERROR] Failed to resolve the GR00T runtime from run.yaml" >&2
        exit 1
    fi
    echo "[INFO] Selected GR00T runtime: ${groot_image}"
    build_groot_image \
        "${groot_image}" \
        "${groot_source_revision}" \
        "${force_rebuild}"
    start_container

    echo "[INFO] Preparing the GR00T model selected in configs/run.yaml..."
    CYCLO_ARENA_MODEL_ROOT="${DOCKER_DIR}/workspace/model" \
    CYCLO_ARENA_SERVER_STATE="${DOCKER_DIR}/workspace/model/.cyclo_arena/server.json" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${CYCLOLAB_PATH}/source/cyclo_arena${PYTHONPATH:+:${PYTHONPATH}}" \
        python3 -m cyclo_arena.host_launcher \
        --config "${CYCLO_ARENA_RUN_CONFIG}" \
        --prepare-only

    echo "[INFO] GR00T is ready. Enter Cyclo Lab and run ./scripts/arena/run.sh"
}

# Configure X11 forwarding
setup_x11() {
    # Check if xauth is installed
    if ! command -v xauth &> /dev/null; then
        echo "[WARN] xauth is not installed. X11 forwarding will not work."
        echo "[WARN] Install with: sudo apt install xauth"
        return 1
    fi

    # Check if DISPLAY is set
    if [ -z "$DISPLAY" ]; then
        echo "[WARN] DISPLAY variable is not set. X11 forwarding will not work."
        return 1
    fi

    # Create temporary directory for xauth
    export __CYCLOLAB_TMP_DIR=$(mktemp -d)
    export __CYCLOLAB_TMP_XAUTH="${__CYCLOLAB_TMP_DIR}/.xauth"

    # Create xauth file
    touch "${__CYCLOLAB_TMP_XAUTH}"
    xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "${__CYCLOLAB_TMP_XAUTH}" nmerge -
    
    echo "[INFO] X11 forwarding configured"
    echo "[INFO] XAUTH file: ${__CYCLOLAB_TMP_XAUTH}"

    return 0
}

# Check if X11 is available
check_x11() {
    if [ -n "$DISPLAY" ] && command -v xauth &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Build docker image
build_image() {
    echo "[INFO] Building Cyclo Lab docker image..."
    cd "${DOCKER_DIR}"
    docker compose build cyclo_lab
    echo "[INFO] Build complete!"
}

# Start docker container
start_container() {
    echo "[INFO] Starting Cyclo Lab docker container..."
    prepare_workspace

    # Check and initialize git submodules
    echo "[INFO] Checking git submodules..."
    cd "${CYCLOLAB_PATH}"
    if [ -d ".git" ]; then
        if git submodule status | grep -q '^-'; then
            echo "[INFO] Initializing git submodules..."
            # Initialize direct dependencies only. Isaac Lab Arena's nested Isaac Lab checkout
            # must not replace Cyclo Lab's pinned Isaac Lab 2.3 runtime.
            git submodule update --init
            echo "[INFO] Git submodules initialized"
        else
            echo "[INFO] Git submodules already initialized"
        fi
    else
        echo "[WARN] Not a git repository, skipping submodule initialization"
    fi

    cd "${DOCKER_DIR}"

    # Setup X11 forwarding
    X11_COMPOSE_FILE=""
    if check_x11; then
        if setup_x11; then
            X11_COMPOSE_FILE="-f x11.yaml"
            echo "[INFO] X11 forwarding enabled"
        fi
    else
        echo "[INFO] X11 forwarding not available (no DISPLAY or xauth)"
    fi

    # Check if container is already running
    if [ -n "$(docker ps -q --filter "name=^cyclo_lab${DOCKER_NAME_SUFFIX}$")" ]; then
        echo "[INFO] Container is already running"
        return 0
    fi

    # Check if container exists but is stopped
    if [ -n "$(docker ps -aq --filter "name=^cyclo_lab${DOCKER_NAME_SUFFIX}$")" ]; then
        echo "[INFO] Starting existing container..."
        docker start cyclo_lab${DOCKER_NAME_SUFFIX}
    else
        echo "[INFO] Creating and starting new container..."
        docker compose -f docker-compose.yaml ${X11_COMPOSE_FILE} up -d cyclo_lab
    fi

    echo "[INFO] Container started successfully!"
    echo "[INFO] Use './docker/container.sh enter' to access the container"
}

# Recreate the container from the current image
recreate_container() {
    echo "[INFO] Recreating Cyclo Lab docker container..."
    prepare_workspace
    cd "${DOCKER_DIR}"

    X11_COMPOSE_FILE=""
    if check_x11 && setup_x11; then
        X11_COMPOSE_FILE="-f x11.yaml"
        echo "[INFO] X11 forwarding enabled"
    fi

    docker compose -f docker-compose.yaml ${X11_COMPOSE_FILE} up -d --force-recreate cyclo_lab
    echo "[INFO] Container recreated successfully!"
}

# Enter running container
enter_container() {
    echo "[INFO] Entering Cyclo Lab docker container..."

    # Check if container is running
    if [ -z "$(docker ps -q --filter "name=^cyclo_lab${DOCKER_NAME_SUFFIX}$")" ]; then
        echo "[ERROR] Container is not running. Start it first with './docker/container.sh start'"
        exit 1
    fi

    # Pass DISPLAY environment variable to the container
    docker exec -it -e DISPLAY="${DISPLAY}" cyclo_lab${DOCKER_NAME_SUFFIX} /bin/bash
}

# Stop container
stop_container() {
    echo "[INFO] Stopping Cyclo Lab docker container..."
    cd "${DOCKER_DIR}"
    docker compose stop cyclo_lab
    echo "[INFO] Container stopped"
}

# Clean up container and image
clean_docker() {
    echo "[INFO] Cleaning up Cyclo Lab docker resources..."
    cd "${DOCKER_DIR}"

    read -p "This will remove the container and image. Continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker compose down cyclo_lab
        docker rmi robotis/cyclo-lab${DOCKER_NAME_SUFFIX}:latest || true
        echo "[INFO] Cleanup complete"
    else
        echo "[INFO] Cleanup cancelled"
    fi
}

# Show container logs
show_logs() {
    echo "[INFO] Showing Cyclo Lab container logs..."
    cd "${DOCKER_DIR}"
    docker compose logs -f cyclo_lab
}

#==
# Main
#==

# check argument provided
if [ -z "$*" ]; then
    echo "[ERROR] No arguments provided." >&2
    print_help
    exit 1
fi

# Load environment variables
load_env

# pass the arguments
case "$1" in
    build)
        build_image
        ;;
    start)
        start_container
        ;;
    start-groot)
        start_groot "${2:-}"
        ;;
    recreate)
        recreate_container
        ;;
    enter)
        enter_container
        ;;
    stop)
        stop_container
        ;;
    clean)
        clean_docker
        ;;
    logs)
        show_logs
        ;;
    -h|--help)
        print_help
        exit 0
        ;;
    *)
        echo "[ERROR] Invalid command: $1"
        print_help
        exit 1
        ;;
esac

echo ""
echo "[INFO] Command completed successfully!"
