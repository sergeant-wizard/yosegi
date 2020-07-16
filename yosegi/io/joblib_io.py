import pathlib

import joblib

import yosegi
from .formatter import Formatter


class _JoblibIO(Formatter):
    @staticmethod
    def save(data: yosegi.Data, path: pathlib.Path) -> None:
        joblib.dump(data.to_dataframe(), path)

    @staticmethod
    def load(path: pathlib.Path,) -> yosegi.Data:
        return yosegi.Data.from_dataframe(df=joblib.load(path))


JoblibIO = _JoblibIO()
