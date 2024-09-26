import os
from functools import wraps
from typing import Any, Callable, Optional


def _get_rank() -> int:
    # Borrowed from pytorch_lightning

    # https://pytorch-lightning.readthedocs.io/en/1.7.7/_modules/pytorch_lightning/utilities/rank_zero.html#rank_zero_warn
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def rank_zero_only(fn: Callable) -> Callable:
    # Borrowed from pytorch_lightning

    """Function that can be used as a decorator to enable a function/method being called only on global rank 0."""

    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Optional[Any]:
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


rank_zero_only.rank = getattr(rank_zero_only, "rank", _get_rank())
