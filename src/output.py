from collections import deque
import os
import threading
import traceback

import numpy as np

from .runtime import cupy


class BoundedOutput:

    def __init__(self, enabled=True, max_inflight=2):
        self.enabled = bool(enabled)
        self.max_inflight = max_inflight
        self._condition = threading.Condition()
        self._jobs = deque()
        self._reserved = 0
        self._inflight = 0
        self._closing = False
        self._aborting = False
        self._error = None
        self._thread = None
        if self.enabled:
            self._thread = threading.Thread(target=self._work, name='solweig-output', daemon=True)
            self._thread.start()

    def _raise_error(self):
        if self._error is not None:
            raise self._error.with_traceback(None) from None

    def check(self):
        with self._condition:
            self._raise_error()

    def reserve(self):
        with self._condition:
            self._raise_error()
            while self._inflight >= self.max_inflight:
                self._condition.wait()
                self._raise_error()
            self._reserved += 1
            self._inflight += 1

    def submit(self, function, *args):
        with self._condition:
            self._raise_error()
            self._reserved -= 1
            job = [function, args]
            if self.enabled:
                self._jobs.append(job)
                self._condition.notify_all()
                return
        self._execute(job)
        self.check()

    def _execute(self, job, skip=False):
        try:
            if not skip:
                job[0](*job[1])
        except BaseException as error:
            with self._condition:
                if self._error is None:
                    diagnostic = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
                    if hasattr(error, 'add_note'):
                        error.add_note('Output worker traceback:\n' + diagnostic)
                    else:
                        error.worker_traceback = diagnostic
                    self._error = error.with_traceback(None)
        finally:
            job.clear()
            with self._condition:
                self._inflight -= 1
                self._condition.notify_all()

    def _work(self):
        while True:
            with self._condition:
                while not self._jobs and not self._closing:
                    self._condition.wait()
                if not self._jobs:
                    return
                job = self._jobs.popleft()
                skip = self._aborting or self._error is not None
            self._execute(job, skip)

    def _close(self, abort):
        with self._condition:
            self._closing = True
            self._aborting = self._aborting or abort
            self._inflight -= self._reserved
            self._reserved = 0
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join()

    def finish(self):
        self._close(abort=False)
        self.check()

    def abort(self):
        self._close(abort=True)


def load_raster(dem_path):
    from osgeo import gdal
    gdal.UseExceptions()
    cp = cupy()
    dataset = gdal.Open(os.fspath(dem_path))
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray().astype(np.float64)
    return cp.array(array, copy=True), dataset


def _write_stack(path, arrays, template, variable, units, base_date, hours, minutes, utc_offset_hours):
    from osgeo import gdal
    gdal.UseExceptions()

    values = np.asarray(arrays, dtype=np.float32)
    rows, cols = values.shape[1:]
    driver = gdal.GetDriverByName("GTiff")
    output = driver.Create(
        os.fspath(path),
        cols,
        rows,
        values.shape[0],
        gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "TILED=YES", "INTERLEAVE=BAND", "BIGTIFF=IF_SAFER"],
    )
    output.SetGeoTransform(template.transform)
    output.SetProjection(template.projection)
    output.SetMetadata(
        {
            "model": "SOLWEIG-CUDA",
            "variable": variable,
            "units": units,
            "time_basis": "forcing_local_time",
        }
    )
    for index in range(values.shape[0]):
        band = output.GetRasterBand(index + 1)
        band.WriteArray(values[index])
        band.SetNoDataValue(float("nan"))
        hour = int(hours[index])
        minute = int(minutes[index])
        timestamp = base_date.replace(hour=hour, minute=minute).isoformat()
        band.SetDescription(f"{variable}_{hour:02d}:{minute:02d}_local")
        band.SetMetadata(
            {
                "Time": timestamp,
                "time_basis": "forcing_local_time",
                "utc_offset_hours": str(utc_offset_hours),
                "solar_sampling": "forcing_time_minus_half_timestep",
                "units": units,
            }
        )
        band.FlushCache()
    output.FlushCache()
    output = None


def _write_static_job(tasks, template):
    from osgeo import gdal
    gdal.UseExceptions()

    for path, values, variable, units, metadata in tasks:
        array = np.asarray(values, dtype=np.float32)
        output = gdal.GetDriverByName("GTiff").Create(
            os.fspath(path),
            array.shape[1],
            array.shape[0],
            1,
            gdal.GDT_Float32,
            options=["COMPRESS=DEFLATE", "TILED=YES"],
        )
        output.SetGeoTransform(template.transform)
        output.SetProjection(template.projection)
        tags = {"model": "SOLWEIG-CUDA", "variable": variable, "units": units}
        if metadata:
            tags.update(metadata)
        output.SetMetadata(tags)
        band = output.GetRasterBand(1)
        band.WriteArray(array)
        band.SetDescription(variable)
        band.SetMetadata({"units": units})
        output.FlushCache()
        band = output = array = None


def _read_raster(path):
    from osgeo import gdal
    gdal.UseExceptions()

    dataset = gdal.Open(os.fspath(path))
    values = dataset.GetRasterBand(1).ReadAsArray().astype(np.float32)
    dataset = None
    return values


def _write_device_static(batch, transfer, path, template):
    host = transfer.download(batch)
    _write_static_job([(path, host['SVF'][0], 'SVF', '1', None)], template)


def _write_device_chunk(batch, transfer, paths, template, base_date, hours,
                        minutes, utc, output_valid_mask=None):
    host = transfer.download(batch)
    invalid = None if output_valid_mask is None else ~output_valid_mask
    for variable, path in paths.items():
        values = host[variable]
        if invalid is not None:
            for band in values:
                band[invalid] = np.nan
        units = 'degree_Celsius' if variable in ('TMRT', 'UTCI') else (
            '1' if variable == 'Shadow' else 'W m-2')
        _write_stack(path, values, template, variable, units, base_date, hours, minutes, utc)
