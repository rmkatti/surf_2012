from __future__ import absolute_import

# Standard library imports.
from cStringIO import StringIO
from io import IOBase
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
    def __init__(self, stdout=True, stderr=True, echo=False):
        self.stdout = stdout
        self.stderr = stderr
        self.echo = echo
    
    def __enter__(self):
        self.sys_stdout = sys.stdout
        self.sys_stderr = sys.stderr
        
        io = StringIO()
        if self.stdout:
            sys.stdout = MultiplexedIO(sys.stdout, io) if self.echo else io
        if self.stderr:
            sys.stderr = MultiplexedIO(sys.stderr, io) if self.echo else io
        return io
    
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.sys_stdout
        sys.stderr = self.sys_stderr


class MultiplexedIO(IOBase):

    def __init__(self, *writers):
        super(MultiplexedIO, self).__init__()
        self.writers = writers

    def write(self, s):
        for writer in self.writers:
            writer.write(s)
