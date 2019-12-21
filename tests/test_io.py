import tempfile

import pandas
import pytest

import yosegi
import yosegi.io


@pytest.fixture
def data() -> yosegi.Data:
    yield yosegi.Data(
        features=pandas.DataFrame({
            'a': [0, 1],
        }),
        labels=pandas.Series([2, 3]),
    )


def test_joblib(data) -> None:
    with tempfile.TemporaryFile() as tf:
        yosegi.io.JoblibIO.save(data, tf)
        tf.seek(0)
        loaded = yosegi.io.JoblibIO.load(tf)
        assert isinstance(loaded, yosegi.Data)
        # TODO: assert that the data is unchanged.
