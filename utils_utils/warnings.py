import warnings

class _BaseWarning:
    def warn(self, message):
        """Issue a warning with the specified message and category."""
        warnings.warn(message, self.__class__)

class DRAGNWarning(UserWarning, _BaseWarning):
    """Base class for warnings in the AGNCleaner package."""
    pass

class DRAGNDeprecationWarning(DeprecationWarning, _BaseWarning):
    """Base class for deprecation warnings in the AGNCleaner package."""
    pass

class AttributeMisuseWarning(UserWarning, _BaseWarning):
    """Warning raised when an attribute is misused."""
    pass
