import pathlib

import pandas

import yosegi


class ParquetIO:
    @staticmethod
    def save(
        data: 'yosegi.Data',
        path: pathlib.Path,
    ) -> None:
        read_options = {
            'compression': None,
        }
        data.features.to_parquet(
            path / 'features',
            **read_options,
        )
        pandas.DataFrame({'label': data.labels}).to_parquet(
            path / 'labels',
            **read_options,
        )

    @staticmethod
    def load(
        path: pathlib.Path,
    ) -> 'yosegi.Data':
        return yosegi.Data(
            features=pandas.read_parquet(path / 'features'),
            labels=pandas.read_parquet(path / 'labels').label,
        )
