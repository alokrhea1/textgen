from queue import Queue
from threading import Thread, current_thread

import modules.shared as shared
from modules.logging_colors import logger


class StopNowException(Exception):
    pass


class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).

    Adapted from: https://stackoverflow.com/a/9969000
    """

    def __init__(self, func, args=None, kwargs=None, callback=None, raise_exceptions=False):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.args = args or []
        self.kwargs = kwargs or {}
        self.stop_now = False
        self.raise_exceptions = raise_exceptions
        self.worker_exception = None
        self.worker_exception_raised = False

        def _callback(val):
            if self.stop_now or shared.stop_everything:
                raise StopNowException
            self.q.put(val)

        def gentask():
            ret = None
            try:
                ret = self.mfunc(callback=_callback, *self.args, **self.kwargs)
            except StopNowException:
                pass
            except Exception as error:
                self.worker_exception = error
                logger.exception("Failed in generation callback")

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            if self.raise_exceptions and self.worker_exception is not None:
                self.worker_exception_raised = True
                raise self.worker_exception
            raise StopIteration
        else:
            return obj

    def __del__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True
        # A stopped consumer must not release the generation lock while the
        # model worker still uses it. The next callback raises StopNowException;
        # wait for that worker to leave generation before returning to the caller.
        if self.thread is not current_thread():
            self.thread.join()
        # A sentence boundary can close the consumer before it reads the
        # sentinel. A real worker failure must still reach guarded generation;
        # intentional callback cancellation has no recorded exception.
        if (self.raise_exceptions and self.worker_exception is not None
                and not self.worker_exception_raised and exc_type in (None, GeneratorExit)):
            self.worker_exception_raised = True
            raise self.worker_exception
