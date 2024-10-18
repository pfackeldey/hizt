import typing as tp

import numcodecs
import zarr


@tp.runtime_checkable
class ArrayFactoryProtocol(tp.Protocol):
    def __call__(
        self, shape: tuple[int, ...], chunks: tuple[int, ...] | bool, dtype: tp.Any
    ) -> zarr.Array: ...


class ZarrArray(ArrayFactoryProtocol):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def __call__(
        self, shape: tuple[int, ...], chunks: tuple[int, ...] | bool, dtype: tp.Any
    ) -> zarr.Array:
        return zarr.create(**self.kwargs, shape=shape, chunks=chunks, dtype=dtype)


default_array_factory = ZarrArray(
    store=zarr.storage.MemoryStore(),
    fill_value=0,
    compressor=numcodecs.Blosc(cname="lz4", clevel=9, shuffle=0),
)
