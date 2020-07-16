import pathlib
from abc import ABC, abstractstaticmethod

from ..data import Data


class Formatter(ABC):
    @abstractstaticmethod
    @staticmethod
    def save(data: Data, path: pathlib.Path) -> None:
        ...

    @abstractstaticmethod
    @staticmethod
    def load(path: pathlib.Path) -> Data:
        ...
