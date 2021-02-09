import warnings


class Log:
    verbose = False
    warning = True

    @classmethod
    def warn(cls, msg):
        if cls.warning:
            warnings.warn(msg, UserWarning)
