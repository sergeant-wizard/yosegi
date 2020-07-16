import pathlib
from abc import ABC, abstractstaticmethod

import yosegi


class Formatter(ABC):
    @staticmethod
    @abstractstaticmethod
    def save(data: "yosegi.Data", path: pathlib.Path) -> None:
        ...

    @staticmethod
    @abstractstaticmethod
    def load(path: pathlib.Path) -> "yosegi.Data":
        ...
