#!/usr/bin/python3
# logger.py
################################################################
import sys
import os
from datetime import datetime
################################################################
# Default log file name
DEFAULT_LOG_FILE = "sparc.log"

class Logger:
    def __init__(self, log_file=None):
        self.log_file = log_file or DEFAULT_LOG_FILE
        self.console_output = sys.stdout
        self.start_time = datetime.now()
        if self.log_file:
            # Ensure the directory exists
            log_dir = os.path.dirname(os.path.abspath(self.log_file))
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self.file_output = open(self.log_file, 'a')
            # Write BEGIN message with timestamp
            self.file_output.write(f"\n{'=' * 72}\n")
            self.file_output.write(f"BEGIN CALCULATION - {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.file_output.write(f"{'=' * 72}\n\n")
            self.file_output.flush()
        else:
            self.file_output = None

    def write(self, message):
        # Write to console
        # if not message.endswith('\n'):
            # message += '\n'
        # Comment this to stop flushing the console output
        self.console_output.write(message)
        self.console_output.flush()
        
        # Write to file if specified
        if self.file_output:
            self.file_output.write(message)
            self.file_output.flush()

    def flush(self):
        # Only flush file output
        if self.file_output:
            self.file_output.flush()

    def close(self):
        if self.file_output:
            # Write END message with timestamp
            end_time = datetime.now()
            duration = end_time - self.start_time
            self.file_output.write(f"\n{'=' * 72}\n")
            self.file_output.write(f"END SPARC - {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.file_output.write(f"Duration: {duration}\n")
            self.file_output.write(f"{'=' * 72}\n")
            self.file_output.close()

# Global logger instance
global_logger = None

# def setup_logger(log_file=None):
#     global global_logger
#     global_logger = Logger(log_file)
#     sys.stdout = global_logger

# def SparcLog(message):
#     if global_logger:
#         # Ensure proper line breaks
#         if not message.endswith('\n'):
#             message += '\n'
#         global_logger.write(message)
#     else:
#         print(message)

def setup_logger(log_file=None, enable=True):
    global global_logger
    if enable:
        global_logger = Logger(log_file or DEFAULT_LOG_FILE)
        sys.stdout = global_logger

def SparcLog(message, level="INFO", origin="SPARC"):
    '''
    Flexible logging function
    - level INFO, WARNING, ERROR, etc.
    - origin: 'SPARC', 'ANALYSIS', etc.
    '''
    prefix = f"[{origin}][{level}] "
    if global_logger:
        if not message.endswith('\n'):
            message += '\n'
        global_logger.write(prefix + message)
    else:
        print(prefix + message)

def close_logger():
    global global_logger
    if global_logger:
        global_logger.close()
        sys.stdout = global_logger.console_output
        global_logger = None

# Set up the logger with the default log file when the module is imported
# setup_logger(DEFAULT_LOG_FILE) 