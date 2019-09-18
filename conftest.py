import os
import tempfile
import numpy
import h5py
import pytest


@pytest.fixture(autouse=True)
def temp_h5(tmpdir):
    fname = tmpdir / "temp.h5"
    return fname


@pytest.fixture(autouse=True)
def add_doctest_vars(doctest_namespace, temp_h5):
    import bth5

    doctest_namespace["np"] = numpy
    doctest_namespace["h5py"] = h5py
    doctest_namespace["bth5"] = bth5
    doctest_namespace["temp_h5"] = temp_h5
