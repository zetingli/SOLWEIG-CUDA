from __future__ import annotations

import numpy as np
from .runtime import cupy, get_kernel


DEFAULT_ANGLE_BATCH = 32
_COUNTS = {
    1: (30, 30, 24, 24, 18, 12, 6, 1),
    2: (31, 30, 28, 24, 19, 13, 7, 1),
    3: (62, 60, 56, 48, 38, 26, 14, 1),
    4: (62, 62, 60, 60, 56, 56, 48, 48, 38, 38, 26, 26, 14, 14, 1),
}
_EDGES = (0, 12, 24, 36, 48, 60, 72, 84, 90)
_ALTITUDES = (6, 18, 30, 42, 54, 66, 78, 90)
_STARTS = (0, 4, 2, 5, 8, 0, 10, 0)


def _prepare_rays_numpy(patch_option, rows, cols, amaxvalue, scale):
    patch_option = int(patch_option)
    rows, cols = int(rows), int(cols)
    amaxvalue, scale = float(amaxvalue), float(scale)
    counts = np.asarray(_COUNTS[patch_option], dtype=np.int32)
    if patch_option == 4:
        edges = np.array((0, 4.5, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 90))
        altitudes = np.array((3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 90))
        starts = np.array((0, 0, 4, 4, 2, 2, 5, 5, 8, 8, 0, 0, 10, 10, 0))
    else:
        edges = np.asarray(_EDGES, dtype=np.float64)
        altitudes, starts = np.asarray(_ALTITUDES), np.asarray(_STARTS)
    band_ids = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    band_offsets = np.r_[0, np.cumsum(counts)].astype(np.int32)
    azimuth = np.concatenate([np.arange(n) * (360. / n) + s for n, s in zip(counts, starts)])
    azimuth[azimuth > 360.] -= 360.
    altitude = altitudes[band_ids].astype(np.float64)
    sectors = ((azimuth >= 0) & (azimuth < 180)).astype(np.uint8)
    sectors |= (((azimuth >= 90) & (azimuth < 270)).astype(np.uint8) << 1)
    sectors |= (((azimuth >= 180) & (azimuth < 360)).astype(np.uint8) << 2)
    sectors |= (((azimuth >= 270) | (azimuth < 90)).astype(np.uint8) << 3)
    weights = np.zeros((len(counts), 2), dtype=np.float64)
    for b, n in enumerate(counts):
        for k in np.arange(edges[b] + 1, edges[b + 1] + 1):
            annulus = 91. - k
            w = (1. / (2. * np.pi)) * np.sin(np.pi / 180.) * np.sin((np.pi * (2. * annulus - 1.)) / 180.)
            weights[b, 0] += ((360. / n) * (np.pi / 180.)) * w
            weights[b, 1] += ((360. / np.ceil(n / 2.)) * (np.pi / 180.)) * w
    patches = len(azimuth)
    capacity = max(rows, cols) + 1
    indices = np.arange(capacity, dtype=np.float64)
    offsets = np.zeros((patches, capacity, 2), dtype=np.int32)
    heights = np.zeros((patches, capacity), dtype=np.float64)
    lengths = np.zeros(patches, dtype=np.int32)
    for p, (azi, alt) in enumerate(zip(azimuth, altitude)):
        if amaxvalue < 0:
            continue
        az, el = azi * (np.pi / 180.), alt * (np.pi / 180.)
        sn, cs, tn = np.sin(az), np.cos(az), np.tan(az)
        ydom = np.pi / 4 <= az < 3 * np.pi / 4 or 5 * np.pi / 4 <= az < 7 * np.pi / 4
        if ydom:
            dy = np.sign(sn) * indices
            dx = -np.sign(cs) * np.abs(np.round(indices / tn))
            ds = np.abs(1. / sn)
        else:
            dy = np.sign(sn) * np.abs(np.round(indices * tn))
            dx = -np.sign(cs) * indices
            ds = np.abs(1. / cs)
        dz = (ds * indices) * (np.tan(el) / scale)
        failed = np.flatnonzero((dz > amaxvalue) | (np.abs(dx) >= rows) | (np.abs(dy) >= cols))
        count = int(failed[0]) + 1 if len(failed) else capacity
        lengths[p] = count
        offsets[p, :count, 0], offsets[p, :count, 1] = dx[:count], dy[:count]
        heights[p, :count] = dz[:count]
    capacity = max(1, int(lengths.max()))
    offsets = np.ascontiguousarray(offsets[:, :capacity])
    heights = np.ascontiguousarray(heights[:, :capacity])
    return dict(band_offsets=band_offsets, sectors=sectors, weights=weights,
                offsets=offsets, heights=heights, heights32=heights.astype(np.float32),
                lengths=lengths, patches=patches, bands=len(counts), capacity=capacity)


def svf_calculator(patch_option, amaxvalue, a, vegdem, vegdem2, bush, scale,
                   angle_batch=DEFAULT_ANGLE_BATCH, transmissivity=0.03):
    cp = cupy()
    a, top, bottom, bush = (cp.asarray(value, dtype=cp.float64, order="C")
                            for value in (a, vegdem, vegdem2, bush))
    rows, cols = a.shape
    trans = float(transmissivity)
    batch = int(angle_batch)
    metadata = _prepare_rays_numpy(patch_option, rows, cols, amaxvalue, scale)
    metadata = {name: cp.asarray(value) if isinstance(value, np.ndarray) else value
                for name, value in metadata.items()}
    patches = metadata["patches"]
    shmat = cp.empty((rows, cols, patches), dtype=cp.float32)
    vegmat = cp.empty_like(shmat)
    vbmat = cp.empty_like(shmat)
    af, tf, bf = (v.astype(cp.float32) for v in (a, top, bottom))
    bushplant = (bush > 1).astype(cp.uint8)
    device_id = cp.cuda.Device().id
    visibility = get_kernel("geometry.cu", "svf_visibility", device_id)
    for start in range(0, patches, batch):
        end = min(start + batch, patches)
        visibility(((a.size + 255) // 256, end - start), (256,), (
            af, tf, bf, a, top, bottom, bushplant,
            np.int32(rows), np.int32(cols), np.int32(patches), np.int32(metadata["capacity"]),
            metadata["offsets"], metadata["heights32"], metadata["heights"],
            metadata["lengths"], np.int32(start), np.int32(end), shmat, vegmat, vbmat))
    reduced = cp.empty((16, rows, cols), dtype=cp.float64)
    get_kernel("geometry.cu", "svf_reduce", device_id)(((a.size + 127) // 128,), (128,), (
        shmat, vegmat, vbmat, bottom, metadata["sectors"], metadata["band_offsets"],
        metadata["weights"], np.int64(a.size), np.int32(patches), np.int32(metadata["bands"]),
        np.float64(trans), reduced))
    cp.cuda.get_current_stream().synchronize()
    order = (0, 10, 1, 11, 6, 4, 14, 9, 2, 12, 7, 5, 3, 13, 8)
    return tuple(reduced[i] for i in order) + (vegmat, vbmat, shmat, reduced[15])


def compute_diffsh(shmat, vegshmat, transmissivity):
    cp = cupy()
    sh_cp = cp.asarray(shmat, dtype=cp.float32, order="C")
    veg_cp = cp.asarray(vegshmat, dtype=cp.float32, order="C")
    diff_cp = cp.empty_like(sh_cp, dtype=cp.float32)
    n = int(diff_cp.size)
    if n == 0:
        return diff_cp
    block = 256
    grid = ((n + block - 1) // block,)
    kernel = get_kernel("geometry.cu", "compute_diffsh_kernel", cp.cuda.Device().id)
    trans = cp.float32(float(transmissivity))
    kernel(grid, (block,), (sh_cp.ravel(), veg_cp.ravel(), diff_cp.ravel(), cp.int32(n), trans))
    cp.cuda.Device().synchronize()
    return diff_cp
