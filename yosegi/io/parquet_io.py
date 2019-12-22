import pathlib

import pandas

import yosegi


class ParquetIO:
    @staticmethod
    def save(
        data: 'yosegi.Data',
        path: pathlib.Path,
    ) -> None:
        data.to_dataframe().to_parquet(
            str(path),
            compression=None,
        )

    @staticmethod
    def load(
        path: pathlib.Path,
    ) -> 'yosegi.Data':
        return yosegi.Data.from_dataframe(
            pandas.read_parquet(path),
        )
