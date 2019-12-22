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
            'b': [10, 11],
        }, index=['s1', 's2']),
        labels=pandas.Series(
            [2, 3],
            index=['s1', 's2'],
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
