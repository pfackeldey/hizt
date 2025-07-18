from __future__ import annotations

import typing as tp
from functools import reduce
from operator import mul

import boost_histogram as bh
import hist
import icechunk as ic
import numpy as np
import zarr
from hist._compat.typing import ArrayLike

from hizt.util import (
    _categorical_axes,
    _get_chunks,
    _get_slice,
    _storage2dtype,
    _to_var_str_dtype,
)


class IcechunkHist:
    def __init__(
        self,
        *axes: bh.axis.Axis,
        storage: hist.storage.Storage,
        repo: ic.Repository,
        __skip_create__: bool = False,
    ) -> None:
        self.axes = hist.axis.NamedAxesTuple(axes)
        self.storage = storage
        self.repo = repo

        # create underlying zarr array
        self.dtype = _storage2dtype.get(self.storage_type())
        if self.dtype is None:
            msg = f"Unsupported storage type: {self.storage_type()}"
            raise TypeError(msg)

        if not __skip_create__:
            """Initialize the histogram by creating the underlying zarr array."""
            session = self.repo.writable_session(branch="main")
            _ = zarr.create_array(
                session.store,
                fill_value=0,
                shape=self.axes.extent,
                chunks=self.chunks,
                dtype=self.dtype,
            )
            session.commit(message="Initialize histogram")

    @classmethod
    def from_icechunk_histogram(cls, other: IcechunkHist) -> IcechunkHist:
        """Create a new IcechunkHist from an existing one."""
        # TODO: we should also add a `from_icechunk_repository` method
        # but this one needs first to store the axes and storage in the repository as well
        return cls(
            *other.axes,
            storage=other.storage,
            repo=other.repo,
            __skip_create__=True,
        )

    @property
    def chunks(self) -> tuple[int, ...]:
        """Return the chunks of the histogram."""
        return tuple(map(_get_chunks, self.axes))

    @property
    def readonly(self) -> zarr.Array:
        """Return a readonly version of the underlying zarr array."""
        session = self.repo.readonly_session(branch="main")
        return self.get_zarr_array(session.store)

    def get_zarr_array(self, store) -> zarr.Array:
        """Access the underlying zarr array."""
        return zarr.open_array(
            store,
            shape=self.axes.extent,
            chunks=self.chunks,
            dtype=self.dtype,
        )

    def get_icechunk_session(self) -> ic.Session:
        return self.repo.writable_session(branch="main")

    def storage_type(self) -> type[hist.storage.Storage]:
        return type(self.storage)

    def history(self) -> tp.Iterator:
        yield from self.repo.ancestry(branch="main")

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @property
    def size(self) -> int:
        return reduce(mul, self.axes.extent)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.axes.size

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
        header = "Zarr store info:"
        ret += f"\n\n{header}"
        ret += f"\n{len(header) * '-'}"
        ret += f"\n{self.readonly.info_complete()}"
        return ret

    def fill(
        self,
        weight: ArrayLike | None = None,
        sample: ArrayLike | None = None,
        threads: int | None = None,
        **kwargs: ArrayLike,  # allow only keyword arguments for axes
    ) -> None:
        tmp_axes = []
        for ax in self.axes:
            if not (ax_name := ax.name):
                raise ValueError(f"Axis {ax} must have a name for filling")
            if ax_name not in kwargs:
                raise ValueError(f"Missing keyword argument for axis {ax_name}")
            if isinstance(ax, _categorical_axes):
                cats: list[str | int] = kwargs[ax_name]  # type: ignore
                if len(cats) != 1:
                    raise ValueError(
                        f"Currently, categorical axis {ax_name} must have exactly one category, got {cats}"
                    )
                tmp_axes.append(
                    # is there a better way to do this?
                    # see: https://github.com/scikit-hep/hist/issues/616
                    type(ax)(
                        cats,
                        name=ax_name,
                        label=ax.label,
                        growth=ax.traits.growth,
                        flow=False,
                    )
                )
            else:
                tmp_axes.append(ax)

        broadcasted = np.broadcast_arrays(*kwargs.values())
        bkwargs = {
            k: _to_var_str_dtype(v)
            for k, v in zip(kwargs.keys(), broadcasted, strict=False)
        }

        # create a small temporary histogram
        tmp_hist = hist.Hist(
            *tmp_axes,
            storage=self.storage,
        )

        tmp_hist.fill(
            weight=weight,
            sample=sample,
            threads=threads,
            **bkwargs,
        )

        self += tmp_hist

    def __iadd__(self, other: hist.Hist) -> IcechunkHist:
        # prepare metadata and axes
        idx = _get_slice(self.axes, other)

        work = []
        for ax in other.axes:
            if isinstance(ax, _categorical_axes):
                work += [*ax]  # type: ignore

        def do(session: ic.Session) -> None:
            ondisk_hist = self.get_zarr_array(session.store)
            # write the histogram
            ondisk_hist[idx] += np.squeeze(np.asarray(other.view(True)))

        # resolve merge conflicts
        self.resolve_merge_conflict(
            do, message="Fill histogram on disk", metadata={"finished": work}
        )
        return self

    def resolve_merge_conflict(
        self, do: tp.Callable, message: str, metadata: dict | None = None
    ) -> None:
        if metadata is None:
            metadata = {}
        metadata["tries"] = 1
        # resolve merge conflicts:
        # the strategy is to read the values corresponding to the HEAD of main branch,
        # then, add the new values, and try to write them back
        # if there is a conflict, we retry until we succeed (i.e. we're at the head of the branch)
        while True:
            try:
                session = self.get_icechunk_session()
                _ = do(session)
                session.commit(message=message, metadata=metadata)
                break
            except ic.ConflictError:
                metadata["tries"] += 1
