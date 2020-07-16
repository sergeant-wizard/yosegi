import pathlib

import pandas

from ..data import Data
from .formatter import Formatter


class _CsvIO(Formatter):
    @staticmethod
    def save(data: Data, path: pathlib.Path) -> None:
        data.to_dataframe().to_csv(path)

    @staticmethod
    def load(path: pathlib.Path) -> Data:
        return Data.from_dataframe(df=pandas.read_csv(path, index_col=0))


CsvIO = _CsvIO()
