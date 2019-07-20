"""tests basic dataset properties"""
import numpy as np
import bth5
import pytest


def test_write(tmp_path):
    with bth5.open(
        tmp_path / "example.h5", "/example", "w", value_dtype=np.float64
    ) as ds:
        ds.write(np.datetime64("2018-06-21 12:26:47"), 2.0)
        ds.write(np.datetime64("2018-06-21 12:26:48"), 1.0)

    with bth5.open(tmp_path / "example.h5", "/example", "r") as ds:
        assert_recordvalidequal(
            ds.record_idx[0], np.datetime64("2018-06-21 12:26:47"), 2.0
        )
        assert_recordvalidequal(
            ds.record_idx[1], np.datetime64("2018-06-21 12:26:48"), 1.0
        )
        assert_recordvalidequal(
            ds.valid_time_idx[np.datetime64("2018-06-21 12:26:47")],
            np.datetime64("2018-06-21 12:26:47"),
            2.0,
        )
        assert_recordvalidequal(
            ds.valid_time_idx[np.datetime64("2018-06-21 12:26:48")],
            np.datetime64("2018-06-21 12:26:48"),
            1.0,
        )

        records = ds.valid_time_idx[
            np.datetime64("2018-06-21 12:26:47") : np.datetime64("2018-06-21 12:26:49")
        ]
        assert_recordvalidequal(records[0], np.datetime64("2018-06-21 12:26:47"), 2.0)
        assert_recordvalidequal(records[1], np.datetime64("2018-06-21 12:26:48"), 1.0)


def test_invalid_order(tmp_path):
    with bth5.open(
        tmp_path / "example.h5", "/example", "w", value_dtype=np.float64
    ) as ds:
        ds.write(np.datetime64("2018-06-21 12:26:48"), 2.0)
        ds.write(np.datetime64("2018-06-21 12:26:47"), 1.0)

    with bth5.open(tmp_path / "example.h5", "/example", "r") as ds:
        assert_recordvalidequal(
            ds.record_idx[0], np.datetime64("2018-06-21 12:26:47"), 1.0
        )
        assert_recordvalidequal(
            ds.record_idx[1], np.datetime64("2018-06-21 12:26:48"), 2.0
        )


def test_interpolate(tmp_path):
    with bth5.open(
        tmp_path / "example.h5", "/example", "w", value_dtype=np.float64
    ) as ds:
        ds.write(np.datetime64("2018-06-21 12:26:47"), 2.0)
        ds.write(np.datetime64("2018-06-21 12:26:49"), 1.0)

    with bth5.open(tmp_path / "example.h5", "/example", "r") as ds:
        assert ds.interpolate_values("2018-06-21 12:26:48") == 1.5


def test_deduplication(tmp_path):
    with bth5.open(
        tmp_path / "example.h5", "/example", "w", value_dtype=np.float64
    ) as ds:
        ds.write(np.datetime64("2018-06-21 12:26:47"), 2.0)
        ds.write(np.datetime64("2018-06-21 12:26:49"), 1.0)

    with bth5.open(tmp_path / "example.h5", "/example", "a") as ds:
        ds.write(np.datetime64("2018-06-21 12:26:49"), 3.0)
        ds.write(np.datetime64("2018-06-21 12:26:51"), 1.0)

    with bth5.open(tmp_path / "example.h5", "/example", "r") as ds:
        records = ds.valid_time_idx[
            np.datetime64("2018-06-21 12:26:47") : np.datetime64("2018-06-21 12:26:51")
        ]

        assert len(records) == 3
        assert_recordvalidequal(records[1], np.datetime64("2018-06-21 12:26:49"), 3.0)


def assert_recordvalidequal(record, valid_time, value):
    assert record["valid_time"] == valid_time
    assert record["value"] == value
    assert not np.isnat(record["transaction_time"])
