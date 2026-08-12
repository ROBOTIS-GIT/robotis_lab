"""Shared Newton physics configuration for Cyclo Lab environments."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg


def make_newton_cfg(
    *,
    njmax: int = 300,
    nconmax: int = 200,
    num_substeps: int = 1,
) -> NewtonCfg:
    """Create the default MuJoCo-Warp-backed Newton configuration.

    ``njmax`` and ``nconmax`` are allocated per replicated environment.  The
    defaults intentionally leave headroom for Cyclo Lab's manipulation scenes;
    individual tasks can lower them later after profiling.
    """

    return NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=njmax,
            nconmax=nconmax,
            cone="pyramidal",
            integrator="implicitfast",
            impratio=1.0,
            ls_iterations=50,
            ls_parallel=True,
        ),
        num_substeps=num_substeps,
        debug_mode=False,
    )
