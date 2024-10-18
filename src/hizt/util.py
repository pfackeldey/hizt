from functools import lru_cache

import hist
import numpy as np


def _mk_sdtype(*names: str, dtype=np.float64) -> np.dtype:
    return np.dtype([(n, dtype) for n in names])


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


def _regrow_axis(values, ax):
    assert ax.traits.growth is True
    ret = hist.axis.StrCategory(values, growth=True)
    ret._ax.metadata = ax._ax.metadata
    return ret


def _get_chunks(ax):
    if ax.traits.growth:
        ret = getattr(ax, "chunks", None)
        if ret is None:
            ret = 1 if isinstance(ax, hist.axis.StrCategory) else 100
        elif ret is max:
            ret = ax.extent
        return ret
    return ax.extent


def view_sdtype_flat(array, type=np.ndarray):
    """
    Returns a view of the input, with its structured dtype flattend.
    The output ndarray subclass is changed to *type*, if not *None*.
    """
    dt = _flatten_sdtype(array.dtype)
    return array.view(dt) if type is None else array.view(dt, type)


@lru_cache(32)
def _flatten_sdtype(dtype: np.dtype) -> np.dtype:
    if dtype.fields is None:
        msg = "not a structured dtype"
        raise ValueError(msg)
    n = len(dtype.fields)
    for i, (t, o) in enumerate(dtype.fields.values()):
        if i == 0:
            dt = t
        else:
            assert dt == t, "inconsistent field dtype"
        assert dt.itemsize * i == o, "inconsistent field offests"
    assert dt.itemsize * n == dtype.itemsize, "inconsistent total size"
    return np.dtype((dt, (n,)))


def _siadd(a, b):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    if a.size:
        av = view_sdtype_flat(np.asarray(a))
        bv = view_sdtype_flat(np.asarray(b))
        np.add(av, bv, out=av)
    return a


def _merge_axis(ax0, ax1):
    if ax0 == ax1:
        return ax0
    if not any(isinstance(a, type(b)) for a, b in [(ax0, ax1), (ax1, ax0)]):
        msg = f"incompatible axes types for merge: {type(ax0)} & {type(ax1)}"
        raise ValueError(msg)
    if not ax0.traits.growth:
        msg = "original axis can't grow"
        raise ValueError(msg)
    if isinstance(ax0, hist.axis.StrCategory):
        return _regrow_axis(list(ax0) + [c for c in ax1 if c not in ax0], ax0)
    msg = f"axis merge for {type(ax0)} not supported"
    raise NotImplementedError(msg)


def _merge_slices(ax0, ax1):
    if ax0 == ax1:
        return [(slice(None), slice(None))]
    if isinstance(ax0, hist.axis.StrCategory):
        if len(ax1) == 1:
            idx = ax0.index(ax1[0])
            return [(idx, 0)] if idx < len(ax0) else []
        keys = sorted(set(ax0) & set(ax1), key=ax0.index)
        if not keys:
            return []
        return [
            tuple(run[0])
            if len(run) == 1
            else tuple(slice(s, e + 1) for s, e in zip(run[0], run[-1], strict=False))
            for run in _runs(np.c_[ax0.index(keys), ax1.index(keys)])
        ]
    msg = f"axis merge for {type(ax0)} not supported"
    raise NotImplementedError(msg)


def _runs(iterable, cmp=lambda a, b: np.all(a + 1 == b)):
    ret = []  # type: ignore[var-annotated]
    for item in iterable:
        if ret and cmp(ret[-1][-1], item):
            ret[-1].append(item)
        else:
            ret.append([item])
    return ret
