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

"""Bridge Isaac Sim 5.1's USD crate check for resolved asset paths."""

from __future__ import annotations

import functools
import importlib
from typing import Any

_PATCH_MARKER = "__cyclo_arena_resolved_path_compat__"


def install_isaac_sim_51_compat() -> tuple[str, ...]:
    """Normalize resolved paths throughout Isaac Sim 5.1's crate-version check."""
    version = importlib.import_module("isaacsim.core.version")
    version_info = version.get_version()
    if tuple(version_info[2:4]) != ("5", "1"):
        return ()

    omni_usd = importlib.import_module("omni.usd")
    usd_utils = importlib.import_module("omni.usd._impl.utils")
    ar = importlib.import_module("pxr.Ar")
    sdf = importlib.import_module("pxr.Sdf")
    usd = importlib.import_module("pxr.Usd")
    tf = importlib.import_module("pxr.Tf")
    carb = importlib.import_module("carb")
    original_version_check = omni_usd.is_usd_crate_file_version_supported
    if getattr(original_version_check, _PATCH_MARKER, False):
        return ()

    probe = ar.ResolvedPath(__file__)
    try:
        sdf.FileFormat.GetFileExtension(probe)
    except Exception as exc:  # noqa: BLE001 - match the Boost.Python binding error by signature
        message = str(exc)
        if "did not match C++ signature" not in message or "GetFileExtension(ResolvedPath)" not in message:
            raise
    else:
        return ()

    @functools.wraps(original_version_check)
    def is_usd_crate_file_version_supported(
        filepath: str,
        stage: Any = None,
        usd_context_name: str = "",
    ) -> bool:
        if not stage:
            usd_context = omni_usd.get_context(usd_context_name)
            stage = usd_context.get_stage() if usd_context else None

        resolver = ar.GetResolver()
        if stage:
            with ar.ResolverContextBinder(stage.GetPathResolverContext()):
                resolved_path = resolver.Resolve(resolver.CreateIdentifier(filepath))
        else:
            resolved_path = resolver.Resolve(resolver.CreateIdentifier(filepath))

        if isinstance(resolved_path, ar.ResolvedPath):
            resolved_path = resolved_path.GetPathString()
        if not resolved_path:
            carb.log_warn(f"Failed to resolve asset path {filepath}; checking the original path instead.")
            resolved_path = filepath
        if not usd_utils.is_usd_crate_file(resolved_path):
            return True
        try:
            usd.CrateInfo.Open(resolved_path)
        except tf.ErrorException as exc:
            carb.log_error(f"Failed to open crate file {resolved_path}: {exc}")
            return False
        return True

    setattr(is_usd_crate_file_version_supported, _PATCH_MARKER, True)
    usd_utils.is_usd_crate_file_version_supported = is_usd_crate_file_version_supported
    omni_usd.is_usd_crate_file_version_supported = is_usd_crate_file_version_supported
    return ("omni.usd.is_usd_crate_file_version_supported(Ar.ResolvedPath)",)
