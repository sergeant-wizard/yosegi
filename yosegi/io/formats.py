import enum

from .joblib_io import JoblibIO


class Formats(enum.Enum):
    joblib = 1
    csv = 2

    @property
    def formatter(self):
        return {
            Formats.joblib: JoblibIO,
        }[self]
