from __future__ import annotations

from .runtime import cupy, get_kernel


def utci_calculator(Ta, RH, Tmrt, va10m):
    cp = cupy()
    arrays = (
        cp.asarray(Ta, dtype=cp.float32),
        cp.asarray(RH, dtype=cp.float32),
        cp.asarray(Tmrt, dtype=cp.float32),
        cp.asarray(va10m, dtype=cp.float32),
    )
    if any(arrays[index].ndim != 0 for index in (0, 1, 3)):
        inputs = tuple(cp.ascontiguousarray(array) for array in cp.broadcast_arrays(*arrays))
        output = cp.empty(inputs[0].shape, dtype=cp.float32)
        if output.size:
            kernel = get_kernel('utci.cu', 'utci_fused_kernel', cp.cuda.Device().id, options=())
            kernel(((output.size + 127) // 128,), (128,), (*inputs, output, cp.int64(output.size)))
            cp.cuda.get_current_stream().synchronize()
        return output
    tmrt = cp.ascontiguousarray(arrays[2])
    output = cp.empty(tmrt.shape, dtype=cp.float32)
    if output.size:
        coefficients = cp.empty(7, dtype=cp.float64)
        device_id = cp.cuda.Device().id
        prepare = get_kernel('utci.cu', 'utci_scalar_weather_prepare', device_id, options=())
        integrate = get_kernel('utci.cu', 'utci_scalar_weather_integrate', device_id, options=())
        prepare((1,), (1,), (arrays[0], arrays[1], arrays[3], coefficients))
        integrate(((output.size + 127) // 128,), (128,),
                  (arrays[0], arrays[1], tmrt, arrays[3], coefficients, output, cp.int64(output.size)))
        cp.cuda.get_current_stream().synchronize()
    return output
