"""Evidence at each query m/z, from one alphatims load of a Bruker timsTOF MALDI .d.

One frame = one pixel. Pass 1 sums, per frame, the events (all mobility scans) in +-ppm windows at M-1, M, M+1 and
M+2 of every query; pass 2 builds the M-1 and M m/z profiles (1 ppm bins, +-60 ppm) over the query's own top-decile
pixels; passes 3-5 split each query's mobilogram into fixed-width components (mobility_slots, isogate1).
Intensities are integer counts, so float32 sums are exact. From these, per query (one row of the evidence table Q):
  iso_r      r(M, M+1) over on-tissue pixels (NaN when M or M+1 is flat)
  mm1_r      r(M-1, M) over on-tissue pixels (0 when M-1 is empty; NaN without on-tissue M signal)
  env0..2    M mean over all pixels and the M+1, M+2 slopes through the origin (sum I0 Ik / sum I0^2) times it
  p90, isum  90th percentile of the non-zero M pixels, and the M sum, over all pixels
  cen        signed m/z centroid (ppm) of the +-25 ppm profile over the top-decile M pixels (NaN without signal)
  valid      the M image is not flat after the coloc transform (it has coloc partners)
  mob_switch the query's coloc image is its best mobility component's image (mobility_slots)
and the coloc matrix: on-tissue M images, optionally TIC-normalised per pixel, minus the mean image over all queries
(common mode), centred and scaled to unit norm; r = row dot product.
"""
import functools
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

from msi_picasso.candidates import NEUTRON

CHUNK = 2000  # queries per chunk of the per-query statistics
PW, CEN_PPM = 60, 25  # M-1 and M profiles: 1 ppm bins, +-PW ppm; the centroid (and viewer profile) uses bins -25..25
# mobility components (bench 2 `isogate1`, bench2/mobcomp2.py): 4-scan bins, fixed 1.5% FWHM Gaussians on a 0.2% grid
# within the M apex +-8%, at most 8 per query, shares >= 5%; a query switches to its best slot (highest M / M+1 r) when
# that r beats the full window's iso r by 0.1
MOB_BIN, MOB_W, MOB_K, MOB_BAND, MOB_STEP, MOB_MIN, MOB_BASE = 4, 1.5, 8, 0.08, 0.2, 0.05, 0.02
MOB_CHUNKS = 64  # fixed: float32 mobilogram sums per chunk as the bench, so the fits match it exactly
SLOT_GATE = 0.1


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


@functools.cache  # one JIT per process
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
    def profile(indptr, tof, inten, tof_mz, n_scan, frames, ptr, win, cmz, hi, n_chunks, out):
        """out[c, q, j, b]: intensity in bin b = round((mz / cmz[w] - 1) 1e6) + PW of window w = 2 q + j (j = 0: M-1,
        1: M), over query q's own top-decile pixels hi[q, i]; frames split in n_chunks, chunk c writes only out[c]."""
        nf = len(frames)
        for c in numba.prange(n_chunks):
            for i in range(c * nf // n_chunks, (c + 1) * nf // n_chunks):
                f = frames[i]
                for e in range(indptr[f * n_scan], indptr[(f + 1) * n_scan]):
                    t = tof[e]
                    for k in range(ptr[t], ptr[t + 1]):
                        w = win[k]
                        q = w // 2
                        if hi[q, i]:
                            b = int((tof_mz[t] / cmz[w] - 1.0) * 1e6 + PW + 1000.5) - 1000
                            if 0 <= b <= 2 * PW:
                                out[c, q, w % 2, b] += inten[e]

    @numba.njit(parallel=True)
    def mobilograms(indptr, tof, inten, n_scan, frames, ptr, win, n_chunks, out):
        """out[c, w, s // MOB_BIN]: summed intensity of window w per mobility bin; chunk c writes only out[c]."""
        nf = len(frames)
        for c in numba.prange(n_chunks):
            for i in range(c * nf // n_chunks, (c + 1) * nf // n_chunks):
                f = frames[i]
                for s in range(n_scan):
                    for e in range(indptr[f * n_scan + s], indptr[f * n_scan + s + 1]):
                        t = tof[e]
                        for k in range(ptr[t], ptr[t + 1]):
                            out[c, win[k], s // MOB_BIN] += inten[e]

    @numba.njit(parallel=True)
    def slot_moments(indptr, tof, inten, n_scan, frames, ptr, win, R, n_chunks, out):
        """Pearson moments of the component images, never stored: per pixel, every M (w = 2 q) and M+1 (w = 2 q + 1)
        event is split over query q's components by R[q, bin, k]; out[c, q, k] += (x, y, xx, yy, xy) of that pixel."""
        nq, nk = R.shape[0], R.shape[2]
        nf = len(frames)
        for c in numba.prange(n_chunks):
            buf = np.zeros((2 * nq, nk), np.float32)  # this pixel's component values (float32 sums, as the bench)
            seen = np.zeros(2 * nq, np.bool_)
            touched = np.empty(2 * nq, np.int64)
            for i in range(c * nf // n_chunks, (c + 1) * nf // n_chunks):
                f = frames[i]
                nt = 0
                for s in range(n_scan):
                    b = s // MOB_BIN
                    for e in range(indptr[f * n_scan + s], indptr[f * n_scan + s + 1]):
                        t = tof[e]
                        for kk in range(ptr[t], ptr[t + 1]):
                            w = win[kk]
                            if not seen[w]:
                                seen[w] = True
                                touched[nt] = w
                                nt += 1
                            for k in range(nk):
                                buf[w, k] += inten[e] * R[w // 2, b, k]
                for j in range(nt):
                    w = touched[j]
                    q = w // 2
                    for k in range(nk):
                        v = np.float64(buf[w, k])
                        if w % 2 == 0:
                            y = np.float64(buf[w + 1, k])
                            out[c, q, k, 0] += v
                            out[c, q, k, 2] += v * v
                            out[c, q, k, 4] += v * y
                        else:
                            out[c, q, k, 1] += v
                            out[c, q, k, 3] += v * v
                for j in range(nt):  # reset only what this pixel touched (the M+1 row is read before its reset)
                    w = touched[j]
                    seen[w] = False
                    buf[w] = 0.0

    @numba.njit(parallel=True)
    def slot_image(indptr, tof, inten, n_scan, frames, ptr, win, Rs, out):
        """out[i, j] = frame i's M events of switched query j, weighted by its chosen component Rs[j, bin]."""
        for i in numba.prange(len(frames)):
            f = frames[i]
            for s in range(n_scan):
                b = s // MOB_BIN
                for e in range(indptr[f * n_scan + s], indptr[f * n_scan + s + 1]):
                    t = tof[e]
                    for k in range(ptr[t], ptr[t + 1]):
                        out[i, win[k]] += inten[e] * Rs[win[k], b]
    return frame_sums, profile, mobilograms, slot_moments, slot_image


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


class _Progress:
    """alphatims progress callback (any object with `max` and `value`): one line that grows, redrawn with \\r on each
    whole-% step, so a terminal and the GUI log show a single bar instead of a tqdm line per update."""
    max = 1

    def __init__(self):
        self._v, self._pct = 0, -1

    @property
    def value(self):
        return self._v

    @value.setter
    def value(self, v):
        self._v, pct = v, int(100 * v / max(self.max, 1))
        if pct != self._pct:
            self._pct = pct
            bar = "#" * (pct // 5) + "." * (20 - pct // 5)
            sys.stdout.write(f"\r{' ' * 24}loading the .d [{bar}] {pct:3d}%" + ("\n" if pct >= 100 else ""))
            sys.stdout.flush()


def extract(d, mzs, ppm=10.0, tic_quantile=0.25, tic_normalize=False, mask=None, threads=32, log=print):
    """Evidence table Q (one row per query m/z in mzs, sorted), the unit-norm coloc rows X (valid rows only, float32),
    the on-tissue images (n_q, 3 windows M-1 / M / M+1, n_on), the top-decile-pixel M-1 / M m/z profiles (n_q, 2,
    121 bins, -60..60 ppm; the M -25..25 ppm part is prof2[:, 1, PW - CEN_PPM:PW + CEN_PPM + 1]), the image layout
    (grid index of each on-tissue pixel, grid shape) and the mobility components (mobility_slots: mob["XS"] = the
    switched queries' best-slot coloc rows (n_switch, n_on)).
    mask: optional (H * W,) bool on-tissue mask (bench parity); default tissue_mask(frame TIC, tic_quantile)."""
    import warnings
    import alphatims.bruker as atb
    import alphatims.utils
    import numba
    threads = min(threads, numba.config.NUMBA_NUM_THREADS)
    numba.set_num_threads(threads)
    t0 = time.time()
    mzs = np.asarray(mzs, np.float64)
    grid, shape, tic, _ = tdf_meta(d)
    alphatims.utils.set_progress_callback(_Progress())
    with warnings.catch_warnings():  # pandas ChainedAssignment FutureWarning inside alphatims' frame table setup
        warnings.simplefilter("ignore", FutureWarning)
        tims = atb.TimsTOF(str(d))
    alphatims.utils.set_progress_callback(True)
    n_scan = int(tims.scan_max_index)
    indptr = np.asarray(tims.push_indptr, np.int64)
    n_frames = (len(indptr) - 1) // n_scan
    tof_mz = np.asarray(tims.mz_values, np.float64)
    tof, inten = tims.tof_indices, tims.intensity_values
    frames = np.flatnonzero(grid[:n_frames] >= 0)
    frames = frames[np.argsort(grid[frames], kind="stable")]  # pixel (grid) order
    on = mask[grid[frames]] if mask is not None else tissue_mask(tic.reindex(frames).fillna(0).to_numpy(), tic_quantile)
    log(f"{len(tof):,} events, {len(frames):,} pixels ({int(on.sum()):,} on tissue), {time.time() - t0:.0f} s")
    frame_sums, profile = _kernels()[:2]
    # pass 1: windows w = 4 q + c, c = 0 (M-1), 1 (M), 2 (M+1), 3 (M+2)
    cm = (mzs[:, None] + NEUTRON * np.arange(-1, 3)[None]).ravel()
    ptr, win = _windows(cm * (1 - ppm * 1e-6), cm * (1 + ppm * 1e-6), tof_mz)
    A = np.zeros((len(frames), len(cm)), np.float32)
    frame_sums(indptr, tof, inten, n_scan, frames, ptr, win, A)
    A = A.reshape(len(frames), len(mzs), 4)
    log(f"isotope images (M-1 .. M+2) summed per pixel, {time.time() - t0:.0f} s")
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
    # pass 2: M-1 and M m/z profiles (+-PW ppm) over each query's own top-decile on-tissue pixels
    cm = (mzs[:, None] + NEUTRON * np.arange(-1, 1)[None]).ravel()
    w = (PW + 1) * 1e-6
    ptr, win = _windows(cm * (1 - w), cm * (1 + w), tof_mz)
    acc = np.zeros((threads, len(mzs), 2, 2 * PW + 1))
    profile(indptr, tof, inten, tof_mz, n_scan, frames[on], ptr, win, cm, hi, threads, acc)
    prof2 = acc.sum(0)
    del acc, hi
    prof = prof2[:, 1, PW - CEN_PPM:PW + CEN_PPM + 1]  # M, -25..25 ppm
    x = np.arange(-CEN_PPM, CEN_PPM + 1, dtype=float)
    with np.errstate(all="ignore"):
        Q["cen"] = (prof * x).sum(1) / prof.sum(1)
    log(f"m/z profiles (M-1 and M, +-{PW} ppm) of the brightest pixels, {time.time() - t0:.0f} s")
    # pass 3-5: mobility components; a query whose best component beats its full window on iso switches its coloc
    # image to that component's image (isogate1)
    k0 = np.asarray(tims.mobility_values, np.float64)
    sw, mob = mobility_slots(indptr, tof, inten, tof_mz, k0, n_scan, frames[on], mzs, ppm, Q.iso_r.to_numpy(),
                            img[:, 1], tic_normalize, threads)
    Q["mob_switch"] = sw
    log(f"mobility components: {int(sw.sum()):,} queries switch to their best component, {time.time() - t0:.0f} s")
    del tims, tof, inten, indptr  # the event arrays live on only in mob["component_images"] (viewer)
    X, Q["valid"] = coloc_rows(img[:, 1], tic_normalize)
    return Q, X, img, prof2, (grid[frames[on]], shape), mob


def mob_fit(y, x):
    """Fixed-width (MOB_W % FWHM) NNLS components of mobilogram y on the 1/K0 values x: [(centre, area share)]
    (bench/mobdeconv.fit). Centres on a MOB_STEP % log grid plus a flat background; adjacent non-zero grid points form
    one component, components closer than MOB_W / 2 % merge, shares below MOB_MIN are dropped."""
    from scipy.optimize import nnls
    o = np.argsort(x)
    x, y = x[o], np.clip(y[o], 0, None)
    if y.sum() <= 0 or len(x) < 3:
        return []
    c = x.min() * np.exp(np.arange(0, np.log(x.max() / x.min()), MOB_STEP / 100))
    sd = c * MOB_W / 100 / 2.3548
    A = np.c_[np.exp(-0.5 * ((x[:, None] - c[None]) / sd[None]) ** 2), np.ones(len(x))]
    coef, _ = nnls(A, y / y.max(), maxiter=50 * A.shape[1])
    g = coef[:-1] * sd * np.sqrt(2 * np.pi)
    nz = np.flatnonzero(g > 0)
    comps, start = [], 0
    for k in range(1, len(nz) + 1):
        if k == len(nz) or nz[k] - nz[k - 1] > 1:
            run = nz[start:k]
            comps.append((np.average(c[run], weights=g[run]), g[run].sum()))
            start = k
    merged = []
    for m, a in comps:
        if merged and (m / merged[-1][0] - 1) * 100 < 0.5 * MOB_W:
            m0, a0 = merged[-1]
            merged[-1] = ((m0 * a0 + m * a) / (a0 + a), a0 + a)
        else:
            merged.append((m, a))
    tot = sum(a for _, a in merged)
    return [(m, a / tot) for m, a in merged if a / tot >= MOB_MIN]


def slot_gate(slots, siso, iso_r):
    """(switch, best slot) per query: best = the slot with the highest M / M+1 r (NaN r: -2); switch when that r beats
    the full window's iso r (NaN: the median over queries) by SLOT_GATE."""
    best = np.argmax(np.where(slots, np.nan_to_num(siso, nan=-2.0), -np.inf), 1)
    iso_full = np.where(np.isnan(iso_r), np.nanmedian(iso_r), iso_r)
    sb = np.nan_to_num(siso[np.arange(len(best)), best], nan=-2.0)
    return slots.any(1) & (sb > iso_full + SLOT_GATE), best


def mobility_slots(indptr, tof, inten, tof_mz, k0, n_scan, frames, mzs, ppm, iso_r, full, tic_normalize, threads):
    """isogate1 (bench 2, 2026-11-19). Per query, label-blind: fixed-width components of its on-tissue M mobilogram
    (mob_fit, within the apex +-MOB_BAND); every M and M+1 event split over them by responsibility
    a_k g_k / (sum_j a_j g_j + MOB_BASE max height); best slot = highest M / M+1 image r. The query switches when that
    r > its full-window iso r (NaN: the median) + SLOT_GATE. Returns (switch (n_q,) bool, mob): mob["XS"]
    (n_switch, n_on) = the switched queries' best-slot M images transformed as coloc rows (TIC per pixel when
    tic_normalize, minus the common mode scaled by the slot's share of the window, centred, unit norm); for the
    viewer only: "component_images" (q -> every component image of queries q, raw), "mg" (n_q, n_bin) on-tissue M mobilograms, "kb" the bins' 1/K0, "cen" /
    "area" (n_q, MOB_K) component centres and shares (NaN: no slot), "siso" their M / M+1 r, "best" the best slot."""
    from scipy.ndimage import gaussian_filter1d
    mobilograms, slot_moments, slot_image = _kernels()[2:]
    nq = len(mzs)
    nb = (n_scan + MOB_BIN - 1) // MOB_BIN
    kb = k0[np.minimum(np.arange(nb) * MOB_BIN + MOB_BIN // 2, n_scan - 1)]  # 1/K0 of each bin
    ptr, win = _windows(mzs * (1 - ppm * 1e-6), mzs * (1 + ppm * 1e-6), tof_mz)
    acc = np.zeros((MOB_CHUNKS, nq, nb), np.float32)
    mobilograms(indptr, tof, inten, n_scan, frames, ptr, win, MOB_CHUNKS, acc)
    mg = acc.sum(0, dtype=np.float64)
    del acc
    apex = gaussian_filter1d(mg, 1.5, axis=1).argmax(1)
    cen, area = np.full((nq, MOB_K), np.nan), np.full((nq, MOB_K), np.nan)
    for i in range(nq):
        m = (kb >= kb[apex[i]] * (1 - MOB_BAND)) & (kb <= kb[apex[i]] * (1 + MOB_BAND))
        for k, (cc, a) in enumerate(sorted(mob_fit(mg[i, m], kb[m]), key=lambda t: -t[1])[:MOB_K]):
            cen[i, k], area[i, k] = cc, a
    slots = np.isfinite(cen)
    # responsibilities (n_q, n_bin, K)
    sd = cen * MOB_W / 100 / 2.3548
    h = np.where(slots, area / (sd * np.sqrt(2 * np.pi)), 0.0)
    g = h[:, None, :] * np.exp(-0.5 * ((kb[None, :, None] - np.nan_to_num(cen)[:, None, :])
                                       / np.where(slots, sd, 1)[:, None, :]) ** 2)
    g[np.broadcast_to(~slots[:, None, :], g.shape)] = 0
    R = (g / (g.sum(2, keepdims=True) + MOB_BASE * np.maximum(h.max(1), 1e-30)[:, None, None])).astype(np.float32)
    del g
    # each slot's M / M+1 image r
    cm2 = np.c_[mzs, mzs + NEUTRON].ravel()
    ptr, win = _windows(cm2 * (1 - ppm * 1e-6), cm2 * (1 + ppm * 1e-6), tof_mz)
    mom = np.zeros((threads, nq, MOB_K, 5))
    slot_moments(indptr, tof, inten, n_scan, frames, ptr, win, R, threads, mom)
    sx, sy, sxx, syy, sxy = np.moveaxis(mom.sum(0), -1, 0)
    n = len(frames)
    with np.errstate(all="ignore"):
        siso = (sxy - sx * sy / n) / np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))
    sw, best = slot_gate(slots, siso, iso_r)
    # the switched queries' best-slot M images
    qs = np.flatnonzero(sw)
    ptr, win = _windows(mzs[qs] * (1 - ppm * 1e-6), mzs[qs] * (1 + ppm * 1e-6), tof_mz)
    Y = np.zeros((len(frames), len(qs)), np.float32)
    slot_image(indptr, tof, inten, n_scan, frames, ptr, win, np.ascontiguousarray(R[qs, :, best[qs]]), Y)
    Y = np.ascontiguousarray(Y.T)
    tic = full.sum(0) if tic_normalize else np.ones(full.shape[1], np.float32)
    tic = np.where(tic > 0, tic, 1).astype(np.float32)
    cmode = (full / tic[None]).mean(0)
    fsum = full[qs].sum(1)
    share = Y.sum(1) / np.where(fsum > 0, fsum, 1)
    Y /= tic[None]
    Y -= share[:, None] * cmode[None]
    Y -= Y.mean(1, keepdims=True)
    Y /= np.maximum(np.sqrt((Y * Y).sum(1, keepdims=True)), 1e-10)

    def component_images(q):
        """(qi, k, raw M images (n, n_on)) of every component of the queries q (viewer only; keeps the event arrays
        alive until the caller drops it)."""
        q = np.asarray(q, np.int64)
        qq, kk = np.nonzero(slots[q])
        qq = q[qq]
        ptr, win = _windows(mzs[qq] * (1 - ppm * 1e-6), mzs[qq] * (1 + ppm * 1e-6), tof_mz)
        out = np.zeros((len(frames), len(qq)), np.float32)
        slot_image(indptr, tof, inten, n_scan, frames, ptr, win, np.ascontiguousarray(R[qq, :, kk]), out)
        return qq, kk, np.ascontiguousarray(out.T)

    return sw, dict(XS=Y, component_images=component_images, mg=mg.astype(np.float32), kb=kb, cen=cen, area=area, siso=siso, best=best)


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


def slot_r_fn(XS, switch, X, valid):
    """pair_r (pair_r_fn) with isogate1: a switched Q row i reads its best-slot image (XS) against j's coloc row."""
    srow, row, pair_r = np.cumsum(switch) - 1, np.cumsum(valid) - 1, pair_r_fn(X, valid)

    def slot_r(i, j):
        r, s = pair_r(i, j), np.flatnonzero(switch[i])
        for k in range(0, len(s), CHUNK):
            t = s[k:k + CHUNK]
            r[t] = np.einsum("ij,ij->i", XS[srow[i[t]]], X[row[j[t]]], dtype=np.float64)
        return r
    return slot_r


if __name__ == "__main__":  # self-checks of the pure parts
    ptr, win = _windows(np.array([1.0, 1.5]), np.array([2.0, 3.0]), np.array([0.5, 1.2, 1.8, 2.5]))
    assert list(np.diff(ptr)) == [0, 1, 2, 1] and list(win) == [0, 0, 1, 1]
    X, v = coloc_rows(np.array([[1, 2, 3, 4], [2, 4, 6, 8], [4, 3, 2, 1], [0, 0, 0, 0]], np.float32), False)
    assert v.all() and np.allclose((X * X).sum(1), 1) and X[0] @ X[2] < 0  # the empty image is valid after common mode
    r = pair_r_fn(X, v)(np.array([0, 1]), np.array([1, 0]))
    assert np.isclose(r[0], r[1]) and np.isclose(r[0], X[0] @ X[1])
    assert list(tissue_mask(np.array([0, 1, 2, 3, 4.0]), 0.5)) == [False, False, False, True, True]
    sl = np.array([[True, True], [True, False], [False, False]])  # slot r .9 / .5; .3 (NaN iso -> median .4); none
    sw, b = slot_gate(sl, np.array([[.5, .9], [.3, np.nan], [.9, .9]]), np.array([.7, np.nan, .1]))
    assert list(sw) == [True, False, False] and b[0] == 1
    print("extract ok")
