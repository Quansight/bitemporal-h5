"""The main bitemporal data set interface"""
import datetime
import posixpath
import numbers
import functools

import h5py
import numpy as np


def _deduplicate(ids, dates):
    a = {}
    b = []

    j = np.intp(0)
    for i in range(len(ids)):
        tid, date = ids[i], dates[i]

        if date in a:
            old_idx = a[date]
            b[old_idx] = i
            a[date] = j
        else:
            b.append(i)
            a[date] = j
            j += 1

    ret = np.array(b, dtype=np.intp)
    np.sort(ret, kind="mergesort")
    return ret


def _wrap_deduplicate(f):
    @functools.wraps(f)
    def wrapped(*a, **kw):
        ret = f(*a, **kw)

        if ret.ndim > 0:
            dedup_ids = _deduplicate(ret["transaction_id"], ret["valid_time"])
            ret = ret[dedup_ids]
        return ret

    return wrapped


NAT = np.datetime64("nat")
TIME_DTYPE = np.dtype("<M8[us]")
h5py.register_dtype(TIME_DTYPE)


def _ensure_groups(handle, path):
    """Makes sure a path exists, returning the final group object."""
    # this assumes the path is an abspath
    group = handle["/"]  # the file handle is the root group.
    heirarchy = path[1:].split("/")
    for name in heirarchy:
        if not name:
            # handle double slashes, //
            continue
        elif name in group:
            group = group[name]
        else:
            group = group.create_group(name)
    return group


def _check_index_dtype(k):
    if not isinstance(k, slice):
        return np.asarray(k).dtype
    arr = [v for v in (k.start, k.stop, k.step) if v is not None]
    return _check_index_dtype(arr)


class _Indexer:
    def __init__(self, reader):
        self._reader = reader

    def __get__(self, obj, otype=None):
        if obj is not None:
            reader = self._reader.__get__(obj, otype)
            return type(self)(reader)

        return self

    def __getitem__(self, k):
        return self._reader(k)


class Dataset:
    """Represents a bitemporal dataset as a memory-mapped structure
    stored in HDF5.
    """

    def __init__(self, filename, path, value_dtype=None):
        """
        Parameters
        ----------
        filename : str
            The path to the h5 file, on disk.
        path : str
            The path to the dataset within the HDF5 file.
        """
        if not posixpath.isabs(path):
            raise ValueError(
                path + "must be a posix absolute path, i.e. "
                "start with a leading '/'."
            )
        self.filename = filename
        self.path = path
        self.closed = True
        self._mode = self._handle = None
        self._staged_data = None
        if value_dtype is not None:
            self._dtype = np.dtype(
                [
                    ("transaction_id", "<u8"),
                    ("transaction_time", TIME_DTYPE),
                    ("valid_time", TIME_DTYPE),
                    ("value", value_dtype),
                ]
            )
        else:
            self._dtype = None

    def _dtype_from_file(self):
        if self._dataset_name not in self._group:
            return None
        ds = self._group[self._dataset_name]
        return ds.dtype

    @property
    def dtype(self):
        # first, see if we have already computed a dtype
        if self._dtype is not None:
            return self._dtype
        # next, try to get it from the HDF5 file
        if self.closed:
            with self:
                dtype = self._dtype_from_file()
        else:
            dtype = self._dtype_from_file()
        # next compute from data
        if dtype is None:
            raise RuntimeError("Must specify dtype on first transaction.")
        self._dtype = dtype
        return dtype

    @dtype.setter
    def dtype(self, value):
        # TODO: add verification that this has the proper format
        self._dtype = value

    def open(self, mode="r", **kwargs):
        """Opens the file for various operations"""
        # check that we should open the dataset
        if not self.closed:
            if self._mode == mode:
                return  # already open in the same mode!
            else:
                raise IOError("attempted to reopen dataset in new mode")
        # open the dataset and return
        self._handle = h5py.File(self.filename, mode)
        self.closed = False
        self._mode = mode
        self._group_name, self._dataset_name = posixpath.split(self.path)
        self._group = _ensure_groups(self._handle, self._group_name)
        if "w" in mode or "a" in mode:
            self._staged_data = []

    @property
    def _dataset(self):
        if not self._dataset_name in self._group:
            self._group.create_dataset(
                self._dataset_name, dtype=self.dtype, maxshape=(None,), shape=(0,)
            )

        return self._group[self._dataset_name]

    def close(self):
        """Close the current file handle."""
        # write the staged data
        if self._staged_data:
            ds = self._dataset
            n = len(self._staged_data)
            data = np.empty(n, dtype=self.dtype)
            data[:] = self._staged_data
            # 1. Mergesort is stable
            # 2. Faster on almost sorted data
            sorted_idx = np.argsort(data["valid_time"], kind="mergesort")
            # set transaction id
            tid = ds.attrs.get("transaction_id", -1) + 1
            data["transaction_id"][:] = tid
            # set transaction time
            now = np.datetime64(datetime.datetime.utcnow())
            data["transaction_time"][:] = now
            # write dataset
            m = ds.len()
            ds.resize((m + n,))
            ds[m:] = data[sorted_idx]
            ds.attrs.modify("transaction_id", tid)
        # now close the file
        self._handle.close()
        self._handle = None
        self.closed = True
        # self._mode does not need to be reset, so that the file can be reopened
        self._group_name = self._group = None
        self._dataset_name = None
        self._staged_data = None

    def __enter__(self):
        if self.closed:
            mode = self._mode or "r"
            self.open(mode=mode)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, valid_time, value):
        """Appends data to a dataset."""
        if self.closed or self._mode not in ("w", "a"):
            raise RuntimeError("dataset must be open to write data to it.")

        data = (-1, NAT, valid_time, value)
        self._staged_data.append(data)

    def interpolate_values(self, valid_times):
        """Interpolates the values at the given valid times."""
        valid_times = self._dataset["valid_time"]
        valid_times = np.asarray(valid_times).astype(TIME_DTYPE)
        min_time, max_time = np.min(valid_times), np.max(valid_times)
        min_idx, max_idx = (
            np.searchsorted(valid_times, min_time, side="right") - 1,
            np.searchsorted(valid_times, max_time, side="left") + 1,
        )
        considered_records = self._dataset[min_idx:max_idx]

        x = considered_records["valid_time"].view(np.int64)
        y = considered_records["value"]

        return np.interp(valid_times.view(np.int64), x, y)

    @_wrap_deduplicate
    def _index_by(self, field, k, multi=False):
        sort_field = self._dataset[field]

        if multi and not isinstance(k, slice):
            k = slice(k, k, None)

        if isinstance(k, slice):
            if k.step is not None:
                raise ValueError(
                    "Stepping is not supported with indexing, use interpolate_values."
                )

            start_idx = (
                np.searchsorted(sort_field, k.start) if k.start is not None else None
            )
            end_idx = (
                np.searchsorted(sort_field, k.stop, side="right")
                if k.stop is not None
                else None
            )

            return self._dataset[start_idx:end_idx]
        else:
            possible_idx = np.searchsorted(sort_field, k)
            if sort_field[possible_idx] == k:
                return self._dataset[possible_idx]
            else:
                raise ValueError(
                    "The specified date was not found in the dataset, use interpolate_value."
                )

    def _construct_indexer(key, multi=False):
        def reader(self, k):
            return self._index_by(key, k, multi=multi)

        return _Indexer(reader)

    valid_time_idx = _construct_indexer("valid_time")
    valid_time_idx.__doc__ = """Indexes into the dataset by valid time."""
    transaction_time_idx = _construct_indexer("transaction_time", multi=True)
    transaction_time_idx.__doc__ = """Indexes into the dataset by transaction time."""
    transaction_id_idx = _construct_indexer("transaction_id", multi=True)
    transaction_id_idx.__doc__ = """Indexes into the dataset by transaction ID."""

    def _record_idx(self, k):
        return self._dataset[k]

    record_idx = _Indexer(_record_idx)


def open(filename, path, mode="r", value_dtype=None, **kwargs):
    """Opens a bitemporal HDF5 dataset."""
    ds = Dataset(filename, path, value_dtype=value_dtype)
    ds.open(mode, **kwargs)
    return ds
