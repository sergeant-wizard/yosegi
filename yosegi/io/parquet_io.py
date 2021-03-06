import pathlib

import pandas

import yosegi
from .formatter import Formatter


class _ParquetIO(Formatter):
    @staticmethod
    def save(data: "yosegi.Data", path: pathlib.Path) -> None:
        data.to_dataframe().to_parquet(
            str(path), compression=None,
        )

    @staticmethod
    def load(path: pathlib.Path,) -> "yosegi.Data":
        return yosegi.Data.from_dataframe(pandas.read_parquet(path))


ParquetIO = _ParquetIO()
