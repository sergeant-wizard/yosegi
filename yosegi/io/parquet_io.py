import pathlib

import pandas

from ..data import Data
from .formatter import Formatter


class _ParquetIO(Formatter):
    @staticmethod
    def save(data: Data, path: pathlib.Path) -> None:
        data.to_dataframe().to_parquet(
            str(path), compression=None,
        )

    @staticmethod
    def load(path: pathlib.Path,) -> Data:
        return Data.from_dataframe(pandas.read_parquet(path))


ParquetIO = _ParquetIO()
