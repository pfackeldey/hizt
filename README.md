# hizt
packed (compressed) boost-histograms/hists


## Installation

```bash
git clone https://github.com/pfackeldey/hizt.git

pip install -e .
```


## Usage

The following example is a minimal example of how to use `hizt` to create a histogram and fill it with some data.

Here we use a Zarr Array on disk as the underlying storage for the histogram. The array is compressed with Blosc.
Zarr supports multiple concurrent writers, so multiple `.fill` calls can be made in parallel, e.g. with Dask.

A `fill` operation will fill a temporary _small_ boost-histogram in memory, and then write the data at the correct position to the underlying zarr array.

```python
import hizt
import hist
import numcodecs
import zarr


# build the underlying histogram array, here: Blosc compressed Zarr Array on disk
array_factory = hizt.array_factory.ZarrArray(
    store=zarr.DirectoryStore('hist.zarr'),
    fill_value=0,
    compressor=numcodecs.Blosc(cname="lz4", clevel=9, shuffle=0),
    overwrite=True,
)

# high-dimensional histogram that would not fit in memory.
# the example below would be 7.6GB in memory (decompressed),
# but only 558 bytes stored on disk (compressed, zero-initialised).
h = hizt.Histogram(
    hist.axis.StrCategory(["DY", "QCD", "Higgs"]),
    hist.axis.StrCategory(["1e", "1mu", "2e", "2mu"]),
    hist.axis.StrCategory([str(i) for i in range(0, 500)]),
    hist.axis.StrCategory([str(i) for i in range(500, 1000)]),
    hist.axis.Regular(100, 0, 1),
    array_factory=array_factory,
)
print(h)
```
