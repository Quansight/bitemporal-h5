"""tests basic dataset properties"""
import numpy as np
import bth5


def test_write(tmp_path):
    with bth5.open(tmp_path / 'example.h5', "/example", "w") as ds:
        ds.write(np.datetime64('2018-06-21 12:26:47'), 2.0)
        ds.write(np.datetime64('2018-06-21 12:26:48'), 2.0)
