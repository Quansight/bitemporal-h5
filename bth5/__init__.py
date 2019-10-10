"""
A generic bitemporal model built on HDF5 (h5py)

Model
-----

The basic model for a bitemporal is an HDF5 dataset that is extensible
along a single dimension with named columns and different dtypes for
each column. In-memory, this will be represented by a numpy structured array.
We will call this structure a *Table*, for purposes here.

Note that HDF5 has its own Table data structure in the high-level
interface (hdf5-hl). We will not be using the high-level table here for
a couple of reasons. The first is that ``h5py`` does not support HDF5's
high-level constructs. The second is that we plan on eventually swapping out
the value column with a deduplicated cache. Relying on low-level HDF5 constructs
grants us this flexibility in the future.

The columns present in the table are as follows:

* ``transaction_id (uint64)``: This is a monotonic integer that represents the
  precise write action that caused this row to be written. Multiple rows may
  be written at the same time, so this value is not unique among rows, though
  presumably all rows with a given transaction id are contiguous in the table.
  This value is zero-indexed. The current largest transaction id should be
  written to the table's attributes as ``max_transaction_id`` (also uint64).
  Write operations should bump the ``max_transaction_id`` by one.
* ``transaction_time (datetime64)``: This is a timestamp (sec since epoch). Any metadata
  about the timezones should be stored as a string attribute of the dataset as
  ``transaction_time_zone``. This represents the time at which the data was
  recorded by the write operation. All rows with the same ``transaction_id`` should
  have the same value here.
* ``valid_time (datetime64)``: This is a timestamp (sec since epoch). Any metadata
  about the timetzones should be stored as a string attribute of the dataset as
  ``valid_time_zone``. This is the primary axis of the time series. It represents
  the data stored in the ``value`` column.
* ``value ((I,J,K,...)<scalar-type>|)``: This column represents the actual values
  of a time series. This may be an N-dimensional array of any valid dtype.
  It is likely sufficient to restrict ourselves to floats and ints, but the model
  should be general enough to accept any scalar dtypes. Additionally, the typical
  usecase will be for this column to be a scalar float value.

Therefor an example numpy dtype with float values and a shape of ``(1, 2, 3)`` is:

.. code:: python

    np.dtype([
        ('transaction_id', '<uint64'),
        ('transaction_time', '<M8'),
        ('valid_time', '<M8'),
        ('value', '<f8', (1, 2, 3))
    ])

Quickstart API
--------------

The interface for writing to the bitemporal HDF5 storage is as follows:

>>> with bth5.open(temp_h5, '/', mode='w', value_dtype=np.int64) as ds:
...     ds.write(np.datetime64("2018-06-21 12:26:47"), 1.0)
...     ds.write(np.datetime64("2018-06-21 12:26:49"), 2.0)
...     ds.write([
...         np.datetime64("2018-06-21 12:26:51"),
...         np.datetime64("2018-06-21 12:26:53"),
...     ], [3.0, 4.0])
>>> with bth5.open(temp_h5, '/', mode='r', value_dtype=np.int64) as ds:
...     ds.records[:]
array([(0, '2018-06-21T12:26:47.000000', 1),
       (0, '2018-06-21T12:26:49.000000', 2),
       (0, '2018-06-21T12:26:51.000000', 3),
       (0, '2018-06-21T12:26:53.000000', 4)],
      dtype=[('transaction_id', '<u8'), ('valid_time', '<M8[us]'), ('value', '<i8')])
>>> with bth5.open(temp_h5, '/', mode='r', value_dtype=np.int64) as ds:
...     ds.transactions[:]  # doctest: +SKIP
array([('2019-09-30T15:35:31.009517', '2018-06-21T12:26:47.000000', '2018-06-21T12:26:53.000000', 0, 4)],
      dtype=[('transaction_time', '<M8[us]'), ('start_valid_time', '<M8[us]'), ('end_valid_time', '<M8[us]'), ('start_idx', '<u8'), ('end_idx', '<u8')])
"""

from .dataset import Dataset, open

__version__ = "0.0.1"
