"""The main bitemporal data set interface"""
import datetime
import posixpath

import h5py
import numpy as np


NAT = np.datetime64("nat")
TIME_DTYPE = np.dtype("<M8[us]")
h5py.register_dtype(TIME_DTYPE)


def _ensure_groups(handle, path):
    """Makes sure a path exists, returning the final group object."""
    # this assumes the path is an abspath
    group = handle  # the file handle is the root group.
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


class Dataset:
    """Represents a bitemporal dataset as a memory-mapped structure
    stored in HDF5.
    """

    def __init__(self, filename, path):
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
            if self._staged_data:
                first_value = np.asarray(self._staged_data[0][3])
                dtype = np.dtype(
                    [
                        ("transaction_id", '<u8'),
                        ("transaction_time", TIME_DTYPE),
                        ("valid_time", TIME_DTYPE),
                        ("value", first_value.dtype, first_value.shape),
                    ]
                )
            else:
                raise RuntimeError("not enough information to compute dtype")
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

    def close(self):
        """Close the current file handle."""
        # write the staged data
        if self._staged_data:
            ds = self._group.require_dataset(
                self._dataset_name, dtype=self.dtype, maxshape=(None,),
                shape=(0,),
            )
            n = len(self._staged_data)
            data = np.empty(n, dtype=self.dtype)
            data[:] = self._staged_data
            # set transaction id
            tid = ds.attrs.get("transaction_id", -1) + 1
            data["transaction_id"][:] = tid
            # set transaction time
            now = np.datetime64(datetime.datetime.utcnow())
            data["transaction_time"][:] = now
            # write dataset
            m = ds.len()
            ds.resize((m + n,))
            ds[m:] = data
            ds.attrs.modify("transaction_id", tid)
        # now close the file
        self._handle.close()
        self._handle = None
        self.closed = True
        # self._mode does not need to be reset, so that the file can be reopened
        self._group_name = self._group = None
        self._dataset_name = self._dataset = None
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
        
        self._staged_data.append((-1, NAT, valid_time, value))

    def __getitem__(self, k):
        if self.closed or self._mode != "r":
            raise RuntimeError("dataset must be open in read mode to read data from it.")

        return self._handle[self._group_name][self._dataset_name][k]
    


def open(filename, path, mode="r", **kwargs):
    """Opens a bitemporal HDF5 dataset."""
    ds = Dataset(filename, path)
    ds.open(mode, **kwargs)
    return ds
