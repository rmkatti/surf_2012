# Standard library imports.
from cStringIO import StringIO
import os.path
import sys


class open_filename(object):
    """ A context manager that opens files but passes through file-like objects.
    """
    def __init__(self, filename, *args, **kwargs):
        self.is_filename = isinstance(filename, basestring)
        if self.is_filename:
            filename = os.path.expanduser(filename)
            self.fh = open(filename, *args, **kwargs)
        else:
            self.fh = filename

    def __enter__(self):
        return self.fh

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_filename:
            self.fh.close()
        return False


class redirect_output(object):
    """ A context manager for redirecting stdout/stderr.
    """
    def __init__(self, stdout=True, stderr=True):
        self.stdout = stdout
        self.stderr = stderr
    
    def __enter__(self):
        self.sys_stdout = sys.stdout
        self.sys_stderr = sys.stderr
        
        io = StringIO()
        if self.stdout:
            sys.stdout = io
        if self.stderr:
            sys.stderr = io
        return io
    
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.sys_stdout
        sys.stderr = self.sys_stderr
