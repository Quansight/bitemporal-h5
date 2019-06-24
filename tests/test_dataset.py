"""tests basic dataset properties"""
import numpy as np
import bth5


def test_write(tmp_path):
    with bth5.open(tmp_path / 'example.h5', "/example", "w") as ds:
        ds.write(np.datetime64('2018-06-21 12:26:47'), 2.0)
        ds.write(np.datetime64('2018-06-21 12:26:48'), 1.0)

    with bth5.open(tmp_path / 'example.h5', "/example", "r") as ds:
        assert_recordvalidequal(ds[0], np.datetime64('2018-06-21 12:26:47'), 2.0)
        assert_recordvalidequal(ds[1], np.datetime64('2018-06-21 12:26:48'), 1.0)

def assert_recordvalidequal(record, valid_time, value):
    assert record["valid_time"] == valid_time
    assert record["value"] == value
    assert not np.isnat(record["transaction_time"])
