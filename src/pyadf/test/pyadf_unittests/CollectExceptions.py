import inspect
import functools
import traceback


def catch_assertions(test_func):
    """
    A function decorator that modifies assert-functions so that they
    catch and collect assertion errors.
    Collected AssertionErrors need to be re-raised later on.
    """
    @functools.wraps(test_func)
    def wrapper(*args, **kwargs):
        pyadftestobj = args[0]  # args[0] is the 'self' param of the test method
        try:
            return test_func(*args, **kwargs)
        except AssertionError as e:  # only AssertionErrors are caught!
            if pyadftestobj.do_not_catch_errors:
                raise e
            else:
                cerror = CaughtAssertionError(test_func, args, kwargs, e)
                pyadftestobj.append_error(cerror)
    return wrapper


def decorate_asserts(cls):
    """
    A class decorator that applies the catch_assertions function decorator to all
    assert methods of the class.

    The test case class that has been decorated
    """
    for name, method in inspect.getmembers(cls, inspect.isroutine):
        if name.startswith('assert') and callable(method):
            setattr(cls, name, catch_assertions(method))
    return cls


class CaughtAssertionError:
    """
    A class decorator that applies the catch_assertions function decorator to all
    assert methods of the class.

    Parameters
    ----------
    assert_function : function
        The decorated assert-function in which the error occurred.
    args : tuple
        The arguments passed to assert_function.
    kwargs : dict
        The keyword arguments passed to assert_function.
    exc : AssertionError
        The originally raised exception.
    """
    def __init__(self, assert_function, args, kwargs, exc):
        self.testobj = args[0]  # args[0] is the 'self' param of the test method
        self.function = assert_function
        self.args = args
        self.kwargs = kwargs
        self.exc = exc

        self.err_num = 1+len(self.testobj.errors)

        self._full_trace = traceback.format_stack()

    def __str__(self):
        """
        Return a human-readable string representation of the object, i.e. the
        error-message.

        Returns
        _______
        self.msg : str
            The string representation.
        """
        return self.msg

    @property
    def msg(self):
        """
        Generate a human-readable string representation of the caught error
        as a messag on demand.

        Returns
        _______
        self._gen_msg() : str
            The error message.
        """
        return self._gen_msg()

    @property
    def trace(self):
        """
        The part of the full trace that is needed to emulate the typical behavior
        for failing assertions.

        Returns
        _______
        trace : str
            The sliced trace as generated from the list called _full_trace.
        """
        trace = self._full_trace
        # slice the irrelevant part from the full trace
        # (the original message also only contains part of the traceback, so
        # here this is reproduced
        start_index = next((i for i, s in enumerate(trace)
                            if 'runTest' in s), None)
        if start_index:
            trace = trace[start_index:]
        end_index = next((i for i, s in enumerate(trace)
                          if 'CollectExceptions.py' in s), None)
        if end_index:
            trace = trace[:end_index]
        return trace

    def _gen_base_msg(self):
        """
        Generate a human-readable string representation of the caught error
        as a messag on demand. This is used to emulate the normal behavior
        if the error had not been caught.

        Returns
        _______
        msg : str
            The basic error msg emulating the normal behavior.
        """
        # this down here generates the old style message, everything before that is
        # optional if we just want to collect the calls
        msg = 'Traceback (most recent call last):\n'
        msg += ''.join(self.trace)
        msg += str(self.exc)
        return msg

    def _gen_msg(self):
        """
        Generate a human-readable string representation of the caught error
        as a messag on demand. The function adds some context to the base message.

        Returns
        _______
        msg : str
            The error msg with helpful additions.
        """
        msg = '\n\n' + 20 * '#' + f'\n{self.err_num}. Assertion\n'
        msg += 20 * '#' + '\n'
        msg += self._gen_base_msg()
        return msg
