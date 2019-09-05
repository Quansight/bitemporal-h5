# bitemporal-h5
A generic bitemporal model built on HDF5 (h5py)

## Model
The basic model for a bitemporal is an HDF5 dataset that is extensible
along a single dimension with named columns and different dtypes for
each column. In-memory, this will be represented by a numpy structured array.
We will call this structure a `Table`, for purposes here.

Note that HDF5 has its own Table data structure in the high-level
interface (hdf5-hl). We will not be using the high-level table here for
a couple of reasons. The first is that `h5py` does not support HDF5's
high-level constructs. The second is that we plan on eventually swapping out
the value column with a deduplicated cache. Relying on low-level HDF5 constructs
grants us this flexibility in the future.

The columns present in the table are as follows:

* `transaction_id (uint64)`: This is a monotonic integer that represents the
  precise write action that caused this row to be written. Multiple rows may
  be written at the same time, so this value is not unique among rows, though
  presumably all rows with a given transaction id are contiguous in the table.
  This value is zero-indexed. The current largest transaction id should be
  written to the table's attributes as `max_transaction_id` (also uint64).
  Write operations should bump the `max_transaction_id` by one.
* `transaction_time (datetime64)`: This is a timestamp (sec since epoch). Any metadata
  about the timezones should be stored as a string attribute of the dataset as
  `transaction_time_zone`. This represents the time at which the data was
  recorded by the write operation. All rows with the same `transaction_id` should
  have the same value here.
* `valid_time (datetime64)`: This is a timestamp (sec since epoch). Any metadata
  about the timetzones should be stored as a string attribute of the dataset as
  `valid_time_zone`. This is the primary axis of the time series. It represents
  the data stored in the `value` column.
* `value ((I,J,K,...)<scalar-type>|)`: This column represents the actual values
  of a time series. This may be an N-dimensional array of any valid dtype.
  It is likely sufficient to restrict ourselves to floats and ints, but the model
  should be general enough to accept any scalar dtypes. Additionally, the typical
  usecase will be for this column to be a scalar float value.

Therefor an example numpy dtype with float values and a shape of `(1, 2, 3)` is:

```python
np.dtype([
    ('transaction_id', '<uint64'),
    ('transaction_time', '<M8'),
    ('valid_time', '<M8'),
    ('value', '<f8', (1, 2, 3))
])
```

## Quickstart API
The interface for writing to the bitemporal HDF5 storage is as follows:

```python
import bth5

with bth5.open("/path/to/file.h5", "/path/to/group", "a+") as ds:
    # all writes are staged into a single transaction and then
    # written when the context exits. Transaction times and IDs
    # are automatically applied. The first write call determines
    # the dtype & shape of value, if the data set does not already
    # exist.
    ds.write(t1, p1)
    ds.write(valid_time=t2, value=p2)
    ds.write(valid_time=[t3, t4, t5], value=[p3, p4, p5])
    ds.write((t6, p6))
    ds.write([(t7, p7), (t8, p8)])
```

Reading from the data set should follow the normal numpy indexing:

```python
# opened in read-only mode
>>> ds = bth5.open("/path/to/file.h5", "/path/to/group")
>>> in_mem = ds[:]
>>> in_mem.dtype
np.dtype([
    ('valid_time', '<M8'),
    ('value', '<f8', (1, 2, 3))
])
```

This again should return the latest valid time and value which is present in the
reduced dataset. Furthermore, there is an escape valve to reach the actual
dataset as represented on-disk. This is the `raw` attribute, that returns
a reference to the h5py dataset.

```python
>>> ds = bth5.open("/path/to/file.h5", "/path/to/group")
>>> ds.raw
```

