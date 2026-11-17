"""Evidence at each query m/z, from one alphatims load of a Bruker timsTOF MALDI .d (BENCH "Bench closed").

One frame = one pixel. Pass 1 sums, per frame, the events (all mobility scans) in +-ppm windows at M-1, M, M+1 and
M+2 of every query; pass 2 builds the M m/z profile (1 ppm bins, +-25 ppm) over the query's own top-decile pixels.
Intensities are integer counts, so float32 sums are exact. From these, per query (one row of the evidence table Q):
  iso_r      r(M, M+1) over on-tissue pixels (NaN when M or M+1 is flat)
  mm1_r      r(M-1, M) over on-tissue pixels (0 when M-1 is empty; NaN without on-tissue M signal)
  env0..2    M mean over all pixels and the M+1, M+2 slopes through the origin (sum I0 Ik / sum I0^2) times it
  p90, isum  90th percentile of the non-zero M pixels, and the M sum, over all pixels
  cen        signed m/z centroid (ppm) of the +-25 ppm profile over the top-decile M pixels (NaN without signal)
  valid      the M image is not flat after the coloc transform (it has coloc partners)
and the coloc matrix: on-tissue M images, optionally TIC-normalised per pixel, minus the mean image over all queries
(common mode), centred and scaled to unit norm; r = row dot product.
"""
import sqlite3
import time

import numpy as np
import pandas as pd

from msi_picasso.candidates import NEUTRON

CHUNK = 2000  # queries per chunk of the per-query statistics
PROF_PPM, CEN_PPM = 40, 25  # bin index as the bench's +-40 ppm profile; the centroid uses bins -25..25


def tdf_meta(d):
    """(frame -> flat (y, x) grid index (-1: no pixel), grid shape, frame TIC, acquired m/z range) from analysis.tdf."""
    with sqlite3.connect(f"{d}/analysis.tdf") as con:
        f = pd.read_sql("SELECT Frame, XIndexPos, YIndexPos FROM MaldiFrameInfo ORDER BY Frame", con)
        tic = pd.read_sql("SELECT Id, SummedIntensities FROM Frames", con).set_index("Id").SummedIntensities
        g = pd.read_sql("SELECT Key, Value FROM GlobalMetadata", con).set_index("Key").Value
    x, y = (f.XIndexPos - f.XIndexPos.min()).to_numpy(), (f.YIndexPos - f.YIndexPos.min()).to_numpy()
    grid = np.full(int(f.Frame.max()) + 1, -1, np.int64)
    grid[f.Frame.to_numpy()] = y * (x.max() + 1) + x
    return grid, (int(y.max()) + 1, int(x.max()) + 1), tic, (float(g["MzAcqRangeLower"]), float(g["MzAcqRangeUpper"]))


def tissue_mask(tic, q):
    """On-tissue frames: TIC >= its q-quantile over measured (TIC > 0) frames."""
    tic = np.asarray(tic, float)
    return tic >= np.quantile(tic[tic > 0], q) if q > 0 else tic > 0


def _windows(lo, hi, tof_mz):
    """TOF index -> the windows [lo, hi] (m/z) that hold it, as CSR (pointer, window ids)."""
    t0, t1 = np.searchsorted(tof_mz, lo, "left"), np.searchsorted(tof_mz, hi, "right")
    n = t1 - t0
    t = np.concatenate([np.arange(a, b) for a, b in zip(t0, t1)]) if n.sum() else np.zeros(0, np.int64)
    w = np.repeat(np.arange(len(lo)), n)
    ptr = np.zeros(len(tof_mz) + 1, np.int64)
    np.add.at(ptr, t + 1, 1)
    return np.cumsum(ptr), w[np.argsort(t, kind="stable")].astype(np.int64)


def _kernels():
    import numba

    @numba.njit(parallel=True)
    def frame_sums(indptr, tof, inten, n_scan, frames, ptr, win, out):
        """out[i, w] = summed intensity of frame frames[i] in window w (threads own frames: no races)."""
        for i in numba.prange(len(frames)):
            f = frames[i]
            for e in range(indptr[f * n_scan], indptr[(f + 1) * n_scan]):
                t = tof[e]
                for k in range(ptr[t], ptr[t + 1]):
                    out[i, win[k]] += inten[e]

    @numba.njit(parallel=True)
    def profile(indptr, tof, inten, tof_mz, n_scan, frames, ptr, win, qmz, hi, n_chunks, out):
        """out[c, q, b - 15]: intensity in bin b = round((mz / q - 1) 1e6) + PROF_PPM of query q, over its own top-decile
        pixels hi[q, i]; frames split in n_chunks, chunk c writes only out[c]."""
        nf = len(frames)
        for c in numba.prange(n_chunks):
            for i in range(c * nf // n_chunks, (c + 1) * nf // n_chunks):
                f = frames[i]
                for e in range(indptr[f * n_scan], indptr[(f + 1) * n_scan]):
                    t = tof[e]
                    for k in range(ptr[t], ptr[t + 1]):
                        q = win[k]
                        if hi[q, i]:
                            b = int((tof_mz[t] / qmz[q] - 1.0) * 1e6 + PROF_PPM + 0.5) - (PROF_PPM - CEN_PPM)
                            if 0 <= b <= 2 * CEN_PPM:
                                out[c, q, b] += inten[e]
    return frame_sums, profile


def _p90(x):
    """The int(0.9 n)-th smallest of the n values (0 when empty)."""
    k = int(0.9 * len(x))
    return np.partition(x, k)[k] if len(x) else 0.0


def _r(a, b):
    """Row-wise Pearson r of two (n, p) float64 arrays."""
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    with np.errstate(all="ignore"):
        return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def extract(d, mzs, ppm=10.0, tic_quantile=0.25, tic_normalize=False, mask=None, threads=32, log=print):
    """Evidence table Q (one row per query m/z in mzs, sorted), the unit-norm coloc rows X (valid rows only, float32),
    the on-tissue images (n_q, 3 windows M-1 / M / M+1, n_on), the top-decile-pixel m/z profiles (n_q, 51 bins,
    -25..25 ppm) and the image layout (grid index of each on-tissue pixel, grid shape).
    mask: optional (H * W,) bool on-tissue mask (bench parity); default tissue_mask(frame TIC, tic_quantile)."""
    import alphatims.bruker as atb
    import numba
    threads = min(threads, numba.config.NUMBA_NUM_THREADS)
    numba.set_num_threads(threads)
    t0 = time.time()
    mzs = np.asarray(mzs, np.float64)
    grid, shape, tic, _ = tdf_meta(d)
    tims = atb.TimsTOF(str(d))
    n_scan = int(tims.scan_max_index)
    indptr = np.asarray(tims.push_indptr, np.int64)
    n_frames = (len(indptr) - 1) // n_scan
    tof_mz = np.asarray(tims.mz_values, np.float64)
    tof, inten = tims.tof_indices, tims.intensity_values
    frames = np.flatnonzero(grid[:n_frames] >= 0)
    frames = frames[np.argsort(grid[frames], kind="stable")]  # pixel (grid) order
    on = mask[grid[frames]] if mask is not None else tissue_mask(tic.reindex(frames).fillna(0).to_numpy(), tic_quantile)
    log(f"[{time.time() - t0:.0f}s] loaded {d}: {len(tof):,} events, {len(frames)} pixels ({on.sum()} on tissue), "
        f"{len(mzs)} queries")
    frame_sums, profile = _kernels()
    # pass 1: windows w = 4 q + c, c = 0 (M-1), 1 (M), 2 (M+1), 3 (M+2)
    cm = (mzs[:, None] + NEUTRON * np.arange(-1, 3)[None]).ravel()
    ptr, win = _windows(cm * (1 - ppm * 1e-6), cm * (1 + ppm * 1e-6), tof_mz)
    A = np.zeros((len(frames), len(cm)), np.float32)
    frame_sums(indptr, tof, inten, n_scan, frames, ptr, win, A)
    A = A.reshape(len(frames), len(mzs), 4)
    log(f"[{time.time() - t0:.0f}s] pass 1 done ({A.nbytes / 1e9:.1f} GB)")
    Q = pd.DataFrame({"mz": mzs})
    M = A[:, :, 1]
    Q["isum"] = M.sum(0, dtype=np.float64)
    # p90: the int(0.9 n)-th smallest of the n non-zero pixel values, as the earlier package
    Q["p90"] = [_p90(col[col > 0]) for col in M.T]
    m0 = (Q.isum.to_numpy() / len(frames)).astype(np.float32)
    s00 = np.einsum("ij,ij->j", M, M, dtype=np.float64)
    Q["env0"] = m0
    for c, name in ((2, "env1"), (3, "env2")):
        s0k = np.einsum("ij,ij->j", M, A[:, :, c], dtype=np.float64)
        Q[name] = (np.divide(s0k, s00, out=np.zeros_like(s00), where=s00 > 0) * m0).astype(np.float32)
    Q.loc[m0 <= 0, ["env0", "env1", "env2"]] = np.nan  # no M signal: no envelope
    img = np.empty((len(mzs), 3, int(on.sum())), np.float32)  # (q, M-1 / M / M+1, on-tissue pixel)
    for s in range(0, len(mzs), CHUNK):
        img[s:s + CHUNK] = A[on, s:s + CHUNK, :3].transpose(1, 2, 0)
    del A, M
    iso, mm1 = np.full(len(mzs), np.nan), np.full(len(mzs), np.nan)
    hi = np.zeros((len(mzs), img.shape[2]), bool)  # each query's top-decile on-tissue pixels
    for s in range(0, len(mzs), CHUNK):
        sl = slice(s, s + CHUNK)
        a = img[sl].astype(np.float64)
        ok = a[:, 1].std(1) > 1e-10
        den = np.sqrt(((a[:, 2] - a[:, 2].mean(1, keepdims=True)) ** 2).sum(1))
        iso[sl] = np.where(ok & (den > 1e-10), _r(a[:, 1], a[:, 2]), np.nan)
        mm1[sl] = np.where(a[:, 1].sum(1) > 0, np.nan_to_num(_r(a[:, 0], a[:, 1])), np.nan)
        m = img[sl, 1]
        hi[sl] = (m >= np.quantile(m, 0.9, axis=1, keepdims=True)) & (m > 0)
    Q["iso_r"], Q["mm1_r"] = iso, mm1
    # pass 2: m/z profile over each query's own top-decile on-tissue pixels
    w = (CEN_PPM + 1) * 1e-6
    ptr, win = _windows(mzs * (1 - w), mzs * (1 + w), tof_mz)
    acc = np.zeros((threads, len(mzs), 2 * CEN_PPM + 1))
    profile(indptr, tof, inten, tof_mz, n_scan, frames[on], ptr, win, mzs, hi, threads, acc)
    prof = acc.sum(0)
    x = np.arange(-CEN_PPM, CEN_PPM + 1, dtype=float)
    with np.errstate(all="ignore"):
        Q["cen"] = (prof * x).sum(1) / prof.sum(1)
    log(f"[{time.time() - t0:.0f}s] pass 2 done")
    del tims, tof, inten, indptr, hi  # the event arrays: not needed past pass 2
    X, Q["valid"] = coloc_rows(img[:, 1], tic_normalize)
    log(f"[{time.time() - t0:.0f}s] evidence done")
    return Q, X, img, prof, (grid[frames[on]], shape)


def coloc_rows(flat, tic_normalize):
    """Unit-norm, centred rows of the on-tissue M images after the per-pixel TIC normalisation (optional) and the
    common-mode removal (minus the mean image over all queries); flat rows dropped. r(i, j) = X[i] . X[j]."""
    X = flat.astype(np.float32, copy=True)
    if tic_normalize:
        col = X.sum(axis=0, keepdims=True)
        np.divide(X, col, out=X, where=col > 0)
    X -= X.mean(axis=0, keepdims=True)
    valid = X.std(axis=1) > 1e-10
    X = X[valid]
    X -= X.mean(axis=1, keepdims=True)
    X /= np.where((n := np.sqrt((X * X).sum(axis=1, keepdims=True))) > 1e-10, n, 1.0)
    return X, valid


def pair_r_fn(X, valid):
    """pair_r(i, j) for Q rows i, j (valid rows only): the row dot products of the coloc rows, each unordered pair
    once (pairs repeat: (i, j) and (j, i), and decoys of several seeds on one ion)."""
    row = np.cumsum(valid) - 1

    def pair_r(i, j):
        a, b = row[i], row[j]
        u, inv = np.unique(np.minimum(a, b) * len(X) + np.maximum(a, b), return_inverse=True)
        ua, ub = u // len(X), u % len(X)
        r = np.empty(len(u))
        for s in range(0, len(u), CHUNK):
            r[s:s + CHUNK] = np.einsum("ij,ij->i", X[ua[s:s + CHUNK]], X[ub[s:s + CHUNK]], dtype=np.float64)
        return r[inv]
    return pair_r


if __name__ == "__main__":  # self-checks of the pure parts
    ptr, win = _windows(np.array([1.0, 1.5]), np.array([2.0, 3.0]), np.array([0.5, 1.2, 1.8, 2.5]))
    assert list(np.diff(ptr)) == [0, 1, 2, 1] and list(win) == [0, 0, 1, 1]
    X, v = coloc_rows(np.array([[1, 2, 3, 4], [2, 4, 6, 8], [4, 3, 2, 1], [0, 0, 0, 0]], np.float32), False)
    assert v.all() and np.allclose((X * X).sum(1), 1) and X[0] @ X[2] < 0  # the empty image is valid after common mode
    r = pair_r_fn(X, v)(np.array([0, 1]), np.array([1, 0]))
    assert np.isclose(r[0], r[1]) and np.isclose(r[0], X[0] @ X[1])
    assert list(tissue_mask(np.array([0, 1, 2, 3, 4.0]), 0.5)) == [False, False, False, True, True]
    print("extract ok")
