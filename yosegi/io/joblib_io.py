import pathlib

import joblib

from ..data import Data
from .formatter import Formatter


class _JoblibIO(Formatter):
    @staticmethod
    def save(data: Data, path: pathlib.Path) -> None:
        joblib.dump(data.to_dataframe(), path)

    @staticmethod
    def load(path: pathlib.Path,) -> Data:
        return Data.from_dataframe(df=joblib.load(path))


JoblibIO = _JoblibIO()
