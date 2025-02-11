# Class CustomException is refered from: https://github.com/getomni-ai/zerox

from typing import Optional, Dict
from ..static import Messages 

class CustomException(Exception):
    """
    Base class for custom exceptions
    """

    def __init__(
        self,
        message: str,
        extra_info: Optional[dict] = None,
    ):
        self.message = message
        self.extra_info = extra_info
        super().__init__(self.message)

    def __str__(self):
        if self.extra_info:
            return f"{self.message} (Extra Info: {self.extra_info})"
        return self.message
    
class MissingEnvironmentVariables(CustomException):
    """Exception raised when the model provider environment variables, API key(s) are missing."""

    def __init__(
        self,
        extra_info: Optional[Dict] = None,
    ):
        message = Messages.MISSING_ENVIRONMENT_VARIABLES
        super().__init__(message=message, extra_info=extra_info)

class DataFileError(CustomException):
    """Exception raised when the input data file has problem"""

    def __init__(
        self,
        extra_info: Optional[Dict] = None,
    ):
        message = Messages.DATAFILEFAILED
        super().__init__(message=message, extra_info=extra_info)
        
class JsonFileError(CustomException):
    """Exception raised when the input json file has problem"""

    def __init__(
        self,
        extra_info: Optional[Dict] = None,
    ):
        message = Messages.JSONFILEERROR
        super().__init__(message=message, extra_info=extra_info)

class ChainInvocationError(CustomException):
    """Exception raised when the chain has problem"""

    def __init__(
        self,
        extra_info: Optional[Dict] = None,
    ):
        message = Messages.CHAIN_INVOCATION_ERROR
        super().__init__(message=message, extra_info=extra_info)

class ChainConfigurationError(CustomException):
    """Exception raised when the chain is not properly configured"""

    def __init__(
        self,
        extra_info: Optional[Dict] = None,
    ):
        message = Messages.CHAIN_CONFIGURATION_ERROR
        super().__init__(message=message, extra_info=extra_info)

class DataClassError(CustomException):
    """Exception raised when the data class has problem"""

    def __init__(
        self,
        extra_info: Optional[Dict] = None,
    ):
        message = Messages.DATACLASSERROR
        super().__init__(message=message, extra_info=extra_info)

class DesignError(CustomException):
    """Exception raised when the table design is invalid"""

    def __init__(
        self,
        extra_info: Optional[Dict] = None,
    ):
        message = Messages.DESIGNERROR
        super().__init__(message=message, extra_info=extra_info)

class ResultTableError(CustomException):
    """Exception raised when the result table has inconsistent data"""

    def __init__(
        self,
        extra_info: Optional[Dict] = None,
    ):
        message = Messages.RESULTTABLEERROR
        super().__init__(message=message, extra_info=extra_info)

class OutputFileError(CustomException):
    """Exception raised when the output file has problem"""

    def __init__(
        self,
        extra_info: Optional[Dict] = None,
    ):
        message = Messages.OUTPUTFILEERROR
        super().__init__(message=message, extra_info=extra_info)
