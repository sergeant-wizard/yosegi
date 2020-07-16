import pathlib

import pandas

import yosegi
from .formatter import Formatter


class _CsvIO(Formatter):
    @staticmethod
    def save(data: "yosegi.Data", path: pathlib.Path) -> None:
        data.to_dataframe().to_csv(path)

    @staticmethod
    def load(path: pathlib.Path) -> "yosegi.Data":
        return yosegi.Data.from_dataframe(df=pandas.read_csv(path, index_col=0))


CsvIO = _CsvIO()
