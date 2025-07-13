# hizt

Ondisk histograms using `zarr` and `icechunk`.


## Installation

```bash
git clone https://github.com/pfackeldey/hizt.git

pip install -e .
```


## Usage

The following example is a minimal example of how to use `hizt` to create a histogram and fill it with some data.

`hizt` uses a Zarr Array on disk as the underlying storage for the histogram. The array is compressed with Blosc.
Zarr supports multiple concurrent writers, so multiple `.fill` calls can be made in parallel, e.g. with Dask.
We're locking automatically conflicting writes by using `icechunk`'s transactional properties.

A `fill` operation will fill a temporary _small_ boost-histogram in memory, and then write the data at the correct position to the underlying zarr array.

The following snippet shows the usage, including the commit history of the `icechunk` repo, and how we can figure out what was dataseted so we can (in principle) resume histogramming at a later stage.

```python
import hizt
import hist
import icechunk as ic
import itertools as it
from datetime import datetime
import numpy as np
from typing import NamedTuple

storage = ic.local_filesystem_storage(f"./.hist/{str(datetime.now()).split()[1]}/")
repo = ic.Repository.create(storage)

hist = hizt.IcechunkHist(
    hist.axis.StrCategory(["DY", "QCD", "Higgs"], name="dataset"),
    hist.axis.StrCategory(["1e", "1mu", "2e", "2mu"], name="category"),
    hist.axis.Regular(100, 0, 100, name="pt"),
    storage=hist.storage.Double(),
    repo=repo,
)


# The following is user code, e.g. in the coffea.datasetor.dataset(...):
# ---
pt_data = np.array([10., 20., 30., 40., 50.])

for dataset in hist.axes[0]:
    # skip "QCD" for demonstration, so we have "work left to do"
    if dataset == "QCD":
        continue
    for cat in hist.axes[1]:
        data = dict(
            dataset=np.array([dataset]),
            category=np.array([cat]),
            pt=pt_data,
        )
        # fill will automatically keep track on what has been finished processing in `hist.readonly.attrs["finished"]`
        hist.fill(**data)
# ---


# let's investigate what we did in the user code:
for commit in hist.history():
    print(commit)

# SnapshotInfo(id="5D77VXYR55AYWFNVGPG0", parent_id=VHMERVEMTTW0E51V748G, written_at=datetime.datetime(2025,7,13,19,33,6,700261, tzinfo=datetime.timezone.utc), message="Fill histo...")
# SnapshotInfo(id="VHMERVEMTTW0E51V748G", parent_id=4RW5N8TYCQFF3ZDZWP60, written_at=datetime.datetime(2025,7,13,19,33,6,697316, tzinfo=datetime.timezone.utc), message="Fill histo...")
# SnapshotInfo(id="4RW5N8TYCQFF3ZDZWP60", parent_id=TJTNR9XQ38SSMEJ8KZQG, written_at=datetime.datetime(2025,7,13,19,33,6,694565, tzinfo=datetime.timezone.utc), message="Fill histo...")
# SnapshotInfo(id="TJTNR9XQ38SSMEJ8KZQG", parent_id=VFFP523WH3TDW3Z53X30, written_at=datetime.datetime(2025,7,13,19,33,6,691542, tzinfo=datetime.timezone.utc), message="Fill histo...")
# SnapshotInfo(id="VFFP523WH3TDW3Z53X30", parent_id=5WWXTSPVKM4PPRRQ9BSG, written_at=datetime.datetime(2025,7,13,19,33,6,688505, tzinfo=datetime.timezone.utc), message="Fill histo...")
# SnapshotInfo(id="5WWXTSPVKM4PPRRQ9BSG", parent_id=2CPDBG9H29HG3BQ89XYG, written_at=datetime.datetime(2025,7,13,19,33,6,685523, tzinfo=datetime.timezone.utc), message="Fill histo...")
# SnapshotInfo(id="2CPDBG9H29HG3BQ89XYG", parent_id=3D9RQZV26MTH7P5SMJEG, written_at=datetime.datetime(2025,7,13,19,33,6,682503, tzinfo=datetime.timezone.utc), message="Fill histo...")
# SnapshotInfo(id="3D9RQZV26MTH7P5SMJEG", parent_id=2D9RDECY98AC4TZ6QBJG, written_at=datetime.datetime(2025,7,13,19,33,6,679314, tzinfo=datetime.timezone.utc), message="Fill histo...")
# SnapshotInfo(id="2D9RDECY98AC4TZ6QBJG", parent_id=1CECHNKREP0F1RSTCMT0, written_at=datetime.datetime(2025,7,13,19,33,6,675053, tzinfo=datetime.timezone.utc), message="Initialize...")
# SnapshotInfo(id="1CECHNKREP0F1RSTCMT0", parent_id=None, written_at=datetime.datetime(2025,7,13,19,33,6,672469, tzinfo=datetime.timezone.utc), message="Repository...")

# get information of what was dataseted up to the last commit
all_work = set(it.product(hist.axes[0], hist.axes[1]))
finished_work = set(map(tuple, hist.readonly.attrs["finished"]))
remaining_work = all_work - finished_work
print("Still need to process:", remaining_work) # we skipped processing "QCD" earlier with `continue`
# Still need to process: {('QCD', '1e'), ('QCD', '2e'), ('QCD', '2mu'), ('QCD', '1mu')}
```
