"""Manager-based task implementations for Cyclo Lab."""

_OPTIONAL_ISAAC_MODULES = {"carb", "isaaclab", "isaaclab_tasks", "isaacsim", "omni"}


def _is_optional_isaac_module(module_name: str | None) -> bool:
    if module_name is None:
        return False
    return any(module_name == name or module_name.startswith(f"{name}.") for name in _OPTIONAL_ISAAC_MODULES)


_BLACKLIST_PKGS = ["utils"]

try:
    from isaaclab_tasks.utils import import_packages

    import_packages(__name__, _BLACKLIST_PKGS)
except ModuleNotFoundError as exc:
    if not _is_optional_isaac_module(exc.name):
        raise
