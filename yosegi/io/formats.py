import enum

from .csv_io import CsvIO
from .joblib_io import JoblibIO
from .parquet_io import ParquetIO


class Formats(enum.Enum):
    joblib = 1
    parquet = 2
    csv = 3

    @property
    def formatter(self):
        return {
            Formats.joblib: JoblibIO,
            Formats.parquet: ParquetIO,
            Formats.csv: CsvIO,
        }[self]
