"""cartpole_race: single shared dynamics spine for an n-link cart-pole."""

# Pin native math libraries (BLAS/OpenMP) to a single thread BEFORE numpy /
# casadi / scipy are imported anywhere in the process. This package is the
# first import in every spawned mapper worker, so setting the variables here
# guarantees they take effect before those runtimes initialize their thread
# pools. With 20 spawn workers on a 20-core box, the default per-core thread
# arenas multiply each process's committed memory and spike the Windows commit
# charge past the lazily grown page file during the simultaneous-spawn burst,
# which previously crashed the gate with "OSError WinError 1455 — The paging
# file is too small for this operation to complete." One thread per worker
# also prevents CPU oversubscription (20 procs * N threads).
import os as _os  # noqa: E402

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    _os.environ.setdefault(_v, "1")

from cartpole_race.dynamics import NLinkCartPole  # noqa: E402
from cartpole_race.env_spec import CartPoleSpec, load_spec  # noqa: E402

__all__ = ["CartPoleSpec", "load_spec", "NLinkCartPole"]
