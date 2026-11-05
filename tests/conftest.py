"""Shared fixtures."""

import sys
import threading
import time
import types

import numpy as np
import pytest


class CallbackStop(Exception):
    pass


class CallbackAbort(Exception):
    pass


@pytest.fixture
def fake_stream(monkeypatch):
    """A Stream that runs the callback in its own thread on a known input ramp.

    As in PortAudio, closing the stream (``__exit__``) stops the callbacks, so the
    caller must wait for ``finished_callback`` inside the ``with`` block.
    """
    seen = {}

    class Stream:
        def __init__(self, **kw):
            seen["kw"] = kw
            seen["out"] = []
            self.n_in, self.n_out = kw["channels"]
            self.kw = kw
            self.closed = threading.Event()

        def _run(self):
            frames = self.kw["blocksize"]
            start = 0
            while not self.closed.is_set():
                time.sleep(0.002)  # the audio thread lags behind the caller
                if self.closed.is_set():
                    break
                t = np.arange(start, start + frames, dtype=np.float32)
                indata = np.stack([t + 1000 * c for c in range(self.n_in)], axis=1)
                outdata = np.full((frames, self.n_out), 9.0, dtype=np.float32)
                try:
                    self.kw["callback"](indata, outdata, frames, None, None)
                except (CallbackStop, CallbackAbort) as exc:
                    seen["out"].append(outdata.copy())
                    seen["stopped_by"] = type(exc).__name__
                    break
                seen["out"].append(outdata.copy())
                start += frames
            self.kw["finished_callback"]()

        def __enter__(self):
            self.thread = threading.Thread(target=self._run)
            self.thread.start()
            return self

        def __exit__(self, *exc):
            self.closed.set()  # closing aborts a running stream
            self.thread.join()
            return False

    mod = types.SimpleNamespace(
        Stream=Stream, CallbackStop=CallbackStop, CallbackAbort=CallbackAbort
    )
    monkeypatch.setitem(sys.modules, "sounddevice", mod)
    return seen
