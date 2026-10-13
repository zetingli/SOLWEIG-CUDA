from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import os
import sys

import numpy as np

CUDA_OPTIONS = ("--std=c++11", "--fmad=false", "--prec-div=true", "--prec-sqrt=true")
_DLL_HANDLES = []


@lru_cache(maxsize=1)
def configure_runtime() -> None:
    library = Path(sys.prefix) / "Library"
    if not library.is_dir():
        library = Path(sys.base_prefix) / "Library"
    env_bin = library / "bin"
    if os.name == "nt" and env_bin.is_dir():
        env_bin_str = str(env_bin)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if env_bin_str not in path_parts:
            os.environ["PATH"] = env_bin_str + os.pathsep + os.environ.get("PATH", "")
        try:
            _DLL_HANDLES.append(os.add_dll_directory(env_bin_str))
        except OSError:
            pass
    for variable, child in (("GDAL_DATA", "gdal"), ("PROJ_DATA", "proj")):
        folder = library / "share" / child
        if folder.is_dir():
            os.environ.setdefault(variable, str(folder))


@lru_cache(maxsize=1)
def cupy():
    configure_runtime()
    import cupy as cp
    return cp


@lru_cache(maxsize=None)
def _kernel_module(filename, device_id, options, precision):
    cp = cupy()
    source = (Path(__file__).parent / "kernels" / filename).read_text(encoding="utf-8-sig")
    if precision is not None:
        real = "double" if precision == "float64" else "float"
        options += (f"-DSOLWEIG_REAL={real}",)
    return cp.RawModule(code=source, options=options)


@lru_cache(maxsize=None)
def get_kernel(filename, name, device_id, options=CUDA_OPTIONS, precision=None):
    return _kernel_module(filename, device_id, options, precision).get_function(name)


@lru_cache(maxsize=None)
def elementwise_kernel(in_params, out_params, filename, name, options=()):
    source = (Path(__file__).parent / "kernels" / filename).read_text(encoding="utf-8-sig")
    return cupy().ElementwiseKernel(in_params, out_params, source, name, options=options)


@contextmanager
def compute_stream_scope():
    cp = cupy()
    caller = cp.cuda.get_current_stream()
    entry = cp.cuda.Event(disable_timing=True)
    entry.record(caller)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.wait_event(entry)
        try:
            yield stream
        finally:
            stream.synchronize()


class DeviceBatch:

    def __init__(self, capacity, *, cp=None):
        self.capacity = capacity
        self._cp = cupy() if cp is None else cp
        self.device_id = self._cp.cuda.Device().id
        self._producer = self._cp.cuda.get_current_stream()
        self._arrays = {}
        self._ready = None
        self._released = False
        self._copy_stream = None
        self._copy_complete = False
        self.count = 0

    def append(self, fields, crop=(slice(None), slice(None))):
        for name, source in fields.items():
            values = source[crop]
            if name not in self._arrays:
                slab = self._cp.empty((self.capacity, *values.shape), dtype=values.dtype)
                self._arrays[name] = slab
            slab = self._arrays[name]
            slab[self.count] = values
        self.count += 1

    def seal(self):
        with self._cp.cuda.Device(self.device_id):
            self._ready = self._cp.cuda.Event(disable_timing=True)
            self._ready.record(self._producer)
        return self

    def release(self):
        if self._released:
            return
        with self._cp.cuda.Device(self.device_id):
            if self._copy_stream is not None and not self._copy_complete:
                self._copy_stream.synchronize()
            if not self._copy_complete and self._ready is not None:
                self._ready.synchronize()
            elif not self._copy_complete:
                self._producer.synchronize()
            self._arrays.clear()
        self._released = True

    def __del__(self):
        if hasattr(self, '_released') and not self._released:
            try:
                self.release()
            except Exception:
                pass


class OutputTransfer:

    def __init__(self, *, cp=None):
        self._cp = cp
        self._stream = None

    def download(self, batch):
        cp = cupy() if self._cp is None else self._cp
        host = {}
        with cp.cuda.Device(batch.device_id):
            if self._stream is None:
                self._stream = cp.cuda.Stream(non_blocking=True)
            stream = self._stream
            batch._copy_stream = stream
            try:
                for name, slab in batch._arrays.items():
                    shape = (batch.count, *slab.shape[1:])
                    size = int(np.prod(shape)) * slab.dtype.itemsize
                    pinned = cp.cuda.alloc_pinned_memory(size)
                    host[name] = np.ndarray(shape, dtype=slab.dtype, buffer=pinned)
                del slab, pinned
                with stream:
                    stream.wait_event(batch._ready)
                    copy_done = cp.cuda.Event(disable_timing=True)
                    for name, slab in batch._arrays.items():
                        values = slab[:batch.count]
                        values.get(out=host[name], stream=stream, blocking=False)
                    del values, slab
                    copy_done.record(stream)
                copy_done.synchronize()
                batch._copy_complete = True
            finally:
                batch.release()
        return host
