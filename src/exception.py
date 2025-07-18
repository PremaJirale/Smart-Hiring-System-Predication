import sys
import traceback

from src.logger import logger  # or 'logging' if you really named it that


def error_message_detail(error, error_detail):
    """
    Create a detailed error message including file name and line number.
    
    Args:
        error (Exception): The caught exception object.
        error_detail (module): The sys module or traceback information.
        
    Returns:
        str: Formatted error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{error}]"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message: str, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message or "An unknown error occurred."
