# User defined exceptions.


class Error(Exception):
    """
    Base class for all exceptions.
    """
    pass


class DataSetPortionMissingError(Error):
    """
    Raised when portion in dataset JSON is missing.
    """
    pass


class ReqdFileNotInSetError(Error):
    """
    Raised when the file asked for does not exist in the dataset.
    """
    pass
