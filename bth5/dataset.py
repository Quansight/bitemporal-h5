"""The main bitemporal data set interface"""
import h5py
import numpy as np


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
        self.filename = filename
        self.path = filename
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

    def close(self):
        """Close the current file handle."""
        self._handle.close()
        self._handle = None
        self.closed = True
        # self._mode does not need to be reset, so that the file can be reopened

    def __enter__(self):
        if self.closed:
            mode = self._mode or "r"
            self.open(mode=mode)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def open(filename, path, mode="r", **kwargs):
    """Opens a bitemporal HDF5 dataset."""
    ds = Dataset(filename, path)
    ds.open(mode, **kwargs)
    return ds
