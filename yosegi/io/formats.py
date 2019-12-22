import enum

from .joblib_io import JoblibIO
from .parquet_io import ParquetIO


class Formats(enum.Enum):
    joblib = 1
    parquet = 2

    @property
    def formatter(self):
        return {
            Formats.joblib: JoblibIO,
            Formats.parquet: ParquetIO,
        }[self]
