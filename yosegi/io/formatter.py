import pathlib
from abc import ABC, abstractstaticmethod

import yosegi


class Formatter(ABC):
    @abstractstaticmethod
    @staticmethod
    def save(data: "yosegi.Data", path: pathlib.Path) -> None:
        ...

    @abstractstaticmethod
    @staticmethod
    def load(path: pathlib.Path) -> "yosegi.Data":
        ...
