"""tests basic dataset properties"""
import bth5


def test_write(temp_h5):
    with bth5.open(temp_h5, "/example", "w") as ds:
        ds.write(1, 2.0)
        ds.write(2, 3.0)
