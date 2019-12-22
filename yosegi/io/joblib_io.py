import pathlib

import joblib

import yosegi


class JoblibIO:
    @staticmethod
    def save(
        data: 'yosegi.Data',
        path: pathlib.Path,
    ) -> None:
        joblib.dump(data, path)

    @staticmethod
    def load(
        path: pathlib.Path,
    ) -> 'yosegi.Data':
        return joblib.load(path)
