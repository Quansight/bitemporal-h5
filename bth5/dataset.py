"""The main bitemporal data set interface"""
import posixpath

import h5py
import numpy as np


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
            raise ValueError(path + "must be a posix absolute path, i.e. "
                             "start with a leading '/'.")
        self.filename = filename
        self.path = path
        self.closed = True
        self._mode = self._handle = None

    def open(self, mode="r", **kwargs):
        """Opens the file for various operations"""
        # check that we should open the dataset
        if not self.closed:
            if self._mode == mode
                return  # already open in the same mode!
            else:
                raise IOError("attempted to reopen dataset in new mode")
        # open the dataset and return
        self._handle = h5py.File(self.filename, mode)
        self.closed = False
        self._mode = mode
        self._group_name, self._dataset_name = posixpath.split(self.path)
        self._group = _ensure_groups(self._handle, self._group_name)

    def close(self):
        """Close the current file handle."""
        # write the staged data
        ds = self._group.require_dataset(self._dataset_name,
                                                             dtype=self._data.dtype
                                                             maxshape=(None,))
        # now close the file
        self._handle.close()
        self._handle = None
        self.closed = True
        # self._mode does not need to be reset, so that the file can be reopened
        self._group_name = self._group = None
        self._dataset_name = self._dataset = None

    def __enter__(self):
        if self.closed:
            mode = self._mode or "r"
            self.open(mode=mode)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, valid_time, value=None):
        """Appends data to a dataset."""
        if self.closed:
            raise RuntimeError("dataset must be open to write data to it.")


def open(filename, path, mode="r", **kwargs):
    """Opens a bitemporal HDF5 dataset."""
    ds = Dataset(filename, path)
    ds.open(mode, **kwargs)
    return ds
