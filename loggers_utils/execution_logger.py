"""
NOTE FOR USERS:

This module provides the `log_execution` decorator for logging the start and
end of function execution with custom messages.
It prints the messages in a visually distinct box format for easier tracking
of code execution in the console.

Typical usage:
    from loggers_utils.execution_logger import log_execution

    @log_execution("Starting...", "Done!")
    def my_function():
        ...

Use this decorator to add informative logs to your functions or methods.
This is just a utility module and does not contain any complex logic or data processing.
"""
import functools


def log_execution(start_message: str, end_message: str):
    """
    Decorator to log the execution of a function with start and end messages.
    The messages are printed in a box format with stars around them.

    :param start_message: The message to print at the start of the function execution.
    :type start_message: str
    :param end_message: The message to print at the end of the function execution.
    :type end_message: str
    :return: The decorated function.
    :rtype: function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def print_msg(message: str):
                message_length = len(message)
                if message_length + (3 * 2) < 89:
                    stars = (83 - message_length) // 2

                message = '〚 ' + '*' * stars + '  ' + message + '  ' + '*' * stars + ' 〛'
                if len(message) < 89:
                    message = message.ljust(89)
                print(message)

            print_msg(start_message)
            result = func(*args, **kwargs)
            print_msg(end_message)
            return result
        return wrapper
    return decorator
