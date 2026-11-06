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
    """Stream and InputStream that run the callback in their own thread on an input ramp.

    As in PortAudio, closing the stream (``__exit__``) stops the callbacks, so the
    caller must wait for ``finished_callback`` inside the ``with`` block. Set
    ``seen["stall"] = True`` before the call for a device that never calls back (the
    stream only finishes when it is closed). ``seen["class"]`` is the class opened.
    """
    seen = {"stall": False}

    class Stream:
        duplex = True

        def __init__(self, **kw):
            seen["kw"] = kw
            seen["class"] = type(self).__name__
            seen["out"] = []
            if self.duplex:
                self.n_in, self.n_out = kw["channels"]
            else:
                self.n_in, self.n_out = kw["channels"], 0
            self.kw = kw
            self.closed = threading.Event()

        def _call(self, indata, outdata, frames):
            if self.duplex:
                self.kw["callback"](indata, outdata, frames, None, None)
            else:
                self.kw["callback"](indata, frames, None, None)

        def _run(self):
            frames = self.kw["blocksize"]
            start = 0
            while not self.closed.is_set():
                time.sleep(0.002)  # the audio thread lags behind the caller
                if self.closed.is_set() or seen["stall"]:
                    continue
                t = np.arange(start, start + frames, dtype=np.float32)
                indata = np.stack([t + 1000 * c for c in range(self.n_in)], axis=1)
                outdata = np.full((frames, self.n_out), 9.0, dtype=np.float32)
                try:
                    self._call(indata, outdata, frames)
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

    class InputStream(Stream):
        duplex = False

    mod = types.SimpleNamespace(
        Stream=Stream,
        InputStream=InputStream,
        CallbackStop=CallbackStop,
        CallbackAbort=CallbackAbort,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", mod)
    return seen


@pytest.fixture
def recording_streams():
    """The :func:`_recording_streams` factory."""
    return _recording_streams


def _recording_streams(calls):
    """Stream / InputStream fakes that record their arguments as ``calls["playrec"]`` /
    ``calls["rec"]`` and finish at once (playrec and rec run on streams, Q-C10)."""

    class Stream:
        key = "playrec"

        def __init__(self, **kw):
            calls[self.key] = kw
            self.kw = kw

        def __enter__(self):
            self.kw["finished_callback"]()
            return self

        def __exit__(self, *exc):
            return False

    class InputStream(Stream):
        key = "rec"

    return {"Stream": Stream, "InputStream": InputStream}
