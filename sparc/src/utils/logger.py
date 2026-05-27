#!/usr/bin/python3
# logger.py

################################################################
import os
import sys
from datetime import datetime

################################################################
# Default log file name
DEFAULT_LOG_FILE = "Sparc.log"


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

            # Backup existing log file with numbered suffix
            if os.path.exists(self.log_file):
                # Find next available backup number
                i = 1
                while os.path.exists(f"{self.log_file}.{i}"):
                    i += 1

                # Rename current log to .1, shift others up
                os.rename(self.log_file, f"{self.log_file}.{i}")

            # Create fresh log file (write mode, not append!)
            self.file_output = open(self.log_file, "w")

            # Write header with timestamp
            self.file_output.write(f"{'=' * 80}\n")
            self.file_output.write(
                f"BEGIN CALCULATION - {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            self.file_output.write(f"{'=' * 80}\n\n")
            self.file_output.flush()
        else:
            self.file_output = None

    def write(self, message):
        # Write to console
        self.console_output.write(message)
        self.console_output.flush()

        # Write to file if enabled
        if self.file_output:
            self.file_output.write(message)
            self.file_output.flush()

    def flush(self):
        self.console_output.flush()
        if self.file_output:
            self.file_output.flush()

    def close(self):
        """Close the log file and write footer."""
        if self.file_output and not self.file_output.closed:
            end_time = datetime.now()
            duration = end_time - self.start_time

            self.file_output.write(f"\n{'=' * 80}\n")
            self.file_output.write(
                f"END CALCULATION - {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            self.file_output.write(f"Total Runtime: {duration}\n")
            self.file_output.write(f"{'=' * 80}\n")
            self.file_output.flush()
            self.file_output.close()

        # Restore original stdout
        sys.stdout = self.console_output

    def __del__(self):
        """Ensure log file is closed on object deletion."""
        self.close()


################################################################
# Global logger instance
global_logger = None


def SparcLog(message, level="INFO"):
    """
    Unified logging function for SPARC package.

    Args:
        message (str): Message to log
        level (str): Log level (INFO, WARNING, ERROR)
    """
    if global_logger is not None:
        prefix = f"[SPARC][{level}] "
        print(f"{prefix}{message}")
    else:
        # Fallback if logger not initialized
        print(f"[SPARC][{level}] {message}")


def setup_logger(log_file=None, enable=True):
    """
    Initialize the global logger.

    Args:
        log_file (str): Path to log file (default: Sparc.log)
        enable (bool): Enable logging to file
    """
    global global_logger

    if enable:
        global_logger = Logger(log_file or DEFAULT_LOG_FILE)
        sys.stdout = global_logger  # Redirect stdout to logger
    else:
        global_logger = None


def close_logger():
    """Close the global logger."""
    global global_logger
    if global_logger is not None:
        global_logger.close()
        global_logger = None


#################################################################
#                         END OF FILE                           #
#################################################################
