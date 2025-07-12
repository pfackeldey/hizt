"""
This is a re-implementation of https://git.rwth-aachen.de/3pia/cms_analyses/common/-/blob/master/utils/bh5.py

In addition, it supports different array backends via the ArrayFactoryProtocol.
"""

import typing as tp
from itertools import product, starmap

import boost_histogram as bh
import hist
import numpy as np

from hizt.array_factory import ArrayFactoryProtocol, default_array_factory
from hizt.util import (
    _get_chunks,
    _merge_axis,
    _merge_slices,
    _regrow_axis,
    _siadd,
    _storage2dtype,
)


class Histogram:
    def __init__(
        self,
        *axes: bh.axis.Axis,
        storage: hist.storage.Storage | None = None,
        metadata: tp.Any = None,
        array_factory: ArrayFactoryProtocol = default_array_factory,
    ) -> None:
        self.axes = hist.axis.NamedAxesTuple(axes)
        if storage is None:
            storage = hist.storage.Weight()
        self.storage = storage
        self.metadata = metadata

        # create underlying zarr array
        self.dtype = _storage2dtype.get(self.storage_type())
        if self.dtype is None:
            msg = f"Unsupported storage type: {self.storage_type()}"
            raise TypeError(msg)

        self._hist = array_factory(
            shape=self.axes.extent,
            chunks=tuple(map(_get_chunks, self.axes)),
            dtype=self.dtype,
        )

    def __repr__(self) -> str:
        # yoink the repr from boost-histogram
        # https://github.com/scikit-hep/boost-histogram/blob/develop/src/boost_histogram/_internal/hist.py#L653-L662
        newline = "\n  "
        first_newline = newline if len(self.axes) > 1 else ""
        storage_newline = (
            newline if len(self.axes) > 1 else " " if len(self.axes) > 0 else ""
        )
        sep = "," if len(self.axes) > 0 else ""
        ret = f"{self.__class__.__name__}({first_newline}"
        ret += f",{newline}".join([str(ax) for ax in self.axes])
        ret += f"{sep}{storage_newline}storage={self.storage_type()},"  # pylint: disable=not-callable
        ret += "\n)"
        if hasattr(self._hist, "info"):
            header = "Underlying histogram info:"
            ret += f"\n\n{header}"
            ret += f"\n{len(header) * '-'}"
            ret += f"\n{self._hist.info}"
        return ret

    def storage_type(self) -> type[hist.storage.Storage]:
        return type(self.storage)

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @property
    def size(self) -> int:
        return self._hist.size

    @property
    def shape(self) -> tuple[int, ...]:
        return self.axes.size

    @property
    def has_flow(self):
        return self.axes.size != self.axes.extent

    @classmethod
    def from_hist(cls, hist, **kwargs):
        ret = cls(
            *hist.axes, storage=hist.storage_type(), metadata=hist.metadata, **kwargs
        )
        ret[:] = hist.view(flow=True)
        return ret

    def to_hist(self, cls=hist.Hist):
        if not issubclass(cls, bh.Histogram):
            msg = f"Unsupported histogram type: {cls}"
            raise TypeError(msg)
        return self._copy(cls)

    def view(self, flow=False):
        if flow or not self.has_flow:
            return self._hist
        raise NotImplementedError()

    def copy(self):
        return self._copy(None, empty=False)

    def reset(self):
        return self._copy(None, empty=True)

    def regrow(self, categories=(), cls=None, copy=True):
        if categories is min:
            categories = {
                i: []
                for i, ax in enumerate(self.axes)
                if isinstance(ax, hist.axis.StrCategory) and ax.traits.growth
            }
        axes = self._mod_axes(categories)
        if cls is None:
            cls = type(self)

        ret = cls(*axes, storage=self.storage, metadata=self.metadata)
        if copy:
            ret._hiadd(self)

        return ret

    def __setitem__(self, key, value) -> None:
        assert isinstance(value, np.ndarray)
        indices = self._expand_key(key)
        self._hist[tuple(indices)] = value

    def __iadd__(self, other):
        self._adopt(other.axes)
        return self._hiadd(other)

    def fill(self, *args, **kwargs) -> None:
        self += self._copy(cls=hist.Hist, empty="ungrow").fill(*args, **kwargs)

    # utility functions, all start with an underscore
    def _adopt(self, axes):
        assert len(self.axes) == len(axes)
        self.axes = hist.axis.NamedAxesTuple(
            starmap(_merge_axis, zip(self.axes, axes, strict=False))
        )
        if self._hist.shape != self.axes.extent:
            self._hist.resize(self.axes.extent)

    def _hiadd(self, other, set=False):
        dst = self.view(flow=True)
        src = other.view(flow=True)
        for il, ir in starmap(
            zip,
            product(*starmap(_merge_slices, zip(self.axes, other.axes, strict=False))),
        ):
            dst[il] = src[ir] if set else _siadd(dst[il], src[ir])
        return self

    def _copy(self, cls=None, empty: bool | str | None = None):
        kwargs = {
            "storage": self.storage,
            "metadata": self.metadata,
            "array_factory": type(self._hist),
        }
        if cls is None:
            cls = type(self)
        if empty == "ungrow":
            axes = tuple(
                _regrow_axis([], ax)
                if isinstance(ax, hist.axis.StrCategory) and ax.traits.growth
                else ax
                for ax in self.axes
            )
        else:
            axes = self.axes
        if issubclass(cls, bh.Histogram):
            kwargs.pop("array_factory", None)
            ret = cls(*axes, **kwargs)
        else:
            ret = cls(*axes, **kwargs)
        if empty is not None:
            ret.view(flow=True)[...] = self.view(flow=True)[...]
        return ret

    def _mod_axes(self, categories):
        if not isinstance(categories, dict):
            categories = dict(enumerate(categories))

        axes = list(self.axes)
        for key, cats in categories.items():
            if cats is None:
                continue
            idx = self._expand_k(key)
            axes[idx] = _regrow_axis(cats, axes[idx])

        return axes

    def _expand_key(self, key):
        # dictify
        if not isinstance(key, dict):
            if not isinstance(key, tuple):
                key = (key,)
            assert key.count(...) <= 1
            if ... in key:
                rev = key[::-1]
                rev = key[: rev.index(...)]
                key = dict(enumerate(key[: key.index(...)]))
                key.update((-i, v) for i, v in enumerate(rev, start=1))
            else:
                key = dict(enumerate(key))

        # build indices
        indices = [slice(None)] * self.ndim
        for k, v in key.items():
            indices[Histogram._expand_k(self, k)] = v

        for i, idx in enumerate(indices):
            if callable(idx):
                indices[i] = idx(self.axes[i])
            elif isinstance(idx, str):  # type: ignore[unreachable]
                indices[i] = self.axes[i].index(idx)  # type: ignore[unreachable]

        return indices

    def _expand_k(self, k):
        if isinstance(k, bh.axis.Axis):
            if k in self.axes:
                return tuple(self.axes).index(k)
            msg = f"axis {k!r} foreign to {self!r}"
            raise ValueError(msg)
        if isinstance(k, str):
            if k in self.axes.name:
                return self.axes.name.index(k)
            msg = f"unknown key {k!r} among {self.axes!r}"
            raise ValueError(msg)
        if isinstance(k, int):
            return k
        msg = f"key {k!r} not understood"
        raise ValueError(msg)
