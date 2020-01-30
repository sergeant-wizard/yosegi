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
    index = ['s1', 's2', 's2']  # intentionally duplicate
    yield yosegi.Data(
        features=pandas.DataFrame({
            'a': [0, 1, 2],
            'b': [10, 11, 12],
        }, index=index),
        labels=pandas.Series(
            [2, 3, 4],
            index=index,
        ),
    )


def test_io(data: yosegi.Data) -> None:
    for fmt in yosegi.io.Formats:
        with tempfile.NamedTemporaryFile() as tf:
            data.save(tf.name, fmt=fmt)
            tf.seek(0)
            loaded = yosegi.Data.load(tf, fmt=fmt)
            assert isinstance(loaded, yosegi.Data)
            assert data == loaded
