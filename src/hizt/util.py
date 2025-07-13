from __future__ import annotations

import hist
import numpy as np

__all__ = [
    "_categorical_axes",
    "_get_chunks",
    "_get_slice",
    "_storage2dtype",
    "_to_var_str_dtype",
]


def _mk_sdtype(*names: str, dtype=np.float64) -> np.dtype:
    return np.dtype([(n, dtype) for n in names])


def _to_var_str_dtype(array: np.ndarray) -> np.ndarray:
    dt = array.dtype
    if dt.kind == "U":
        return np.asarray(array, dtype=np.dtypes.StringDType())
    return array


_storage2dtype = {
    hist.storage.Double: np.float64,
    hist.storage.Int64: np.int64,
    hist.storage.AtomicInt64: np.int64,
    hist.storage.Weight: _mk_sdtype(
        "value",
        "variance",
    ),
    hist.storage.Mean: _mk_sdtype(
        "count",
        "value",
        "variance",
    ),
    hist.storage.WeightedMean: _mk_sdtype(
        "sum_of_weights",
        "sum_of_weights_squared",
        "value",
        "_sum_of_weighted_deltas_squared",
    ),
}

_categorical_axes = (
    hist.axis.StrCategory,
    hist.axis.IntCategory,
)


def _get_chunks(ax):
    if ax.traits.growth:
        ret = getattr(ax, "chunks", None)
        if ret is None:
            ret = 1 if isinstance(ax, _categorical_axes) else 100
        elif ret is max:
            ret = ax.extent
        return ret
    return ax.extent


def _get_slice(refaxes: hist.axis.NamedAxesTuple, hist: hist.Hist) -> tuple[int, ...]:
    idx = []
    assert len(refaxes) == len(hist.axes), (
        "Reference axes and histogram axes must have the same length"
    )
    for ref, ax in zip(refaxes, hist.axes, strict=True):
        assert type(ref) == type(ax), (
            "Reference axes and histogram axes must be of the same type"
        )
        if isinstance(ax, _categorical_axes):
            (i,) = ref.index([*ax])
            idx.append(i)
        else:
            idx.append(slice(None))
    return tuple(idx)
