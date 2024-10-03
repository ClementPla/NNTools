from typing import ClassVar

from nntools.utils.const import NNOpt


class NNToolSettings:
    __conf: ClassVar[dict] = {
        "dataset.on_error": NNOpt.SKIP_ON_ERROR,
    }

    @staticmethod
    def config(name):
        return NNToolSettings.__conf[name]

    @staticmethod
    def set(name, value):
        if name in NNToolSettings.__setters:
            NNToolSettings.__conf[name] = value
        else:
            raise NameError("Name not accepted in set() method")
