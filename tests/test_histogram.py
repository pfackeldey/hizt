import shutil

import hist
import numcodecs
import zarr

import hizt


def test_histogram_in_memory():
    array_factory = hizt.array_factory.ZarrArray(
        store=zarr.storage.MemoryStore(),
        fill_value=0,
        compressor=numcodecs.Blosc(cname="lz4", clevel=9, shuffle=0),
    )

    h = hizt.Histogram(
        hist.axis.StrCategory(["DY", "QCD", "Higgs"]),
        hist.axis.StrCategory(["1e", "1mu", "2e", "2mu"]),
        hist.axis.Regular(10, 0, 1),
        array_factory=array_factory,
    )
    print("before", repr(h))
    h.fill("DY", "1e", 0.1)
    print("after", repr(h))


def test_histogram_on_disk():
    array_factory = hizt.array_factory.ZarrArray(
        store=zarr.DirectoryStore("hist.zarr"),
        fill_value=0,
        compressor=numcodecs.Blosc(cname="lz4", clevel=9, shuffle=0),
        overwrite=True,
    )

    h = hizt.Histogram(
        hist.axis.StrCategory(["DY", "QCD", "Higgs"]),
        hist.axis.StrCategory(["1e", "1mu", "2e", "2mu"]),
        hist.axis.Regular(10, 0, 1),
        array_factory=array_factory,
    )
    print("before", repr(h))
    h.fill("DY", "1e", 0.1)
    print("after", repr(h))

    shutil.rmtree("hist.zarr")
