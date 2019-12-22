import pathlib
import tempfile

import pandas
import pytest

import yosegi
import yosegi.io

formatters = [
    yosegi.io.Formats.joblib,
    yosegi.io.Formats.parquet,
]


@pytest.fixture
def data() -> yosegi.Data:
    yield yosegi.Data(
        features=pandas.DataFrame({
            'a': [0, 1],
        }),
        labels=pandas.Series([2, 3]),
    )


def test_joblib(data: yosegi.Data) -> None:
    with tempfile.TemporaryFile() as tf:
        data.save(tf, fmt=yosegi.io.Formats.joblib)
        tf.seek(0)
        loaded = yosegi.Data.load(tf, fmt=yosegi.io.Formats.joblib)
        assert isinstance(loaded, yosegi.Data)
        assert data == loaded


def test_parquet(data: yosegi.Data) -> None:
    with tempfile.TemporaryDirectory() as td:
        data.save(pathlib.Path(td), fmt=yosegi.io.Formats.parquet)
        loaded = yosegi.Data.load(pathlib.Path(td), fmt=yosegi.io.Formats.parquet)
        assert isinstance(loaded, yosegi.Data)
        assert data == loaded
