"""Results viewer of picasso-gui (structure after CompOmics/visDIA): one q threshold, a search box and a decoy switch
for every view; Overview / Proteins / Peptides / Run QC; tables drive detail panels (protein -> its peptides ->
peptide evidence). Reads only the run's outputs (run.json, peptides.tsv, proteins.tsv, rows.parquet,
evidence.parquet, viewer/); nothing is recomputed except display z-scores and image correlations.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import dash_ag_grid as dag
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, Patch, ctx, dcc, html, no_update
from plotly.subplots import make_subplots

from msi_picasso.extract import MOB_BAND, MOB_W, PW
from msi_picasso.score import CRAT_PPM, ION_PPM, _z, averagine_dist3, term_weights

NAVY, PURPLE, BLUE, CYAN = "#001050", "#3d0f8f", "#0060c0", "#00a0b0"
GREEN, YELLOW, ORANGE, RED, GREY = "#4fb020", "#f0b000", "#f08000", "#d82814", "#9aa3b8"
HEAD, TEXT = "Silkscreen, monospace", "Departure Mono, monospace"
ION_SCALE = "Viridis"  # as the CLI's figures/
PAGES = ("overview", "proteins", "peptides", "qc")
QS = (0.01, 0.05, 0.1)
N_PROT_IMAGES = 12

TEMPLATE = go.layout.Template(layout=go.Layout(
    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(family=TEXT, size=13, color=NAVY),
    title=dict(font=dict(family=HEAD, size=15)), colorway=[BLUE, ORANGE, GREEN, PURPLE, RED, CYAN, YELLOW],
    margin=dict(l=72, r=20, t=44, b=48), hoverlabel=dict(font=dict(family=TEXT)),
    xaxis=dict(linecolor=NAVY, linewidth=2, gridcolor="#e3e8f2", zeroline=False, ticks="outside"),
    yaxis=dict(linecolor=NAVY, linewidth=2, gridcolor="#e3e8f2", zeroline=False, ticks="outside",
               exponentformat="power", automargin=True, title_standoff=8),
    legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor=NAVY, borderwidth=1)))

GRID_THEME = {"function": "themeQuartz.withParams(" + json.dumps(dict(
    fontFamily="Departure Mono, monospace", headerFontFamily="Silkscreen, monospace", fontSize=13, headerFontSize=12,
    foregroundColor=NAVY, headerTextColor="#ffffff", headerBackgroundColor=NAVY, borderColor="#c8d0e0",
    accentColor=ORANGE, selectedRowBackgroundColor="#ffe2b8", rowHoverColor="#e6f6f8", borderRadius=0,
    wrapperBorderRadius=0, spacing=5)) + ")"}
NUM = {"function": "params.value == null ? '' : d3.format('.3~f')(params.value)"}


# ------------------------------------------------------------------ data
@lru_cache(maxsize=4)
def _load(outdir: str, mtime: float) -> dict:
    """Everything a results dir holds, read once per (dir, run.json mtime)."""
    out = Path(outdir)
    info = json.loads((out / "run.json").read_text())
    pt = pd.read_csv(out / "peptides.tsv", sep="\t").set_index("peptide")
    prt = pd.read_csv(out / "proteins.tsv", sep="\t").set_index("key")
    rows = pd.read_parquet(out / "rows.parquet")
    Q = pd.read_parquet(out / "evidence.parquet")
    kept = rows[~rows.removed]
    best = kept.loc[kept.groupby("peptide").s.idxmax()].set_index("peptide")  # each reported peptide's row
    pt["qi"] = best.qi.reindex(pt.index)
    pt["mz"] = best.feature_mz.reindex(pt.index)
    pt["n_rows"] = rows.groupby("peptide").size().reindex(pt.index)
    terms, w = term_weights(rows)  # five terms (bench 1 run) or eight (bench 2)
    z = pd.DataFrame(_z(rows[terms].to_numpy(float)) * w, index=rows.index, columns=terms)  # weighted, as in s0
    v = out / "viewer"
    meta = dict(np.load(v / "meta.npz")) if (v / "meta.npz").exists() else None
    imgs = np.load(v / "images.npy", mmap_mode="r") if (v / "images.npy").exists() else None
    slot = np.load(v / "slot_images.npy", mmap_mode="r") if (v / "slot_images.npy").exists() else None
    return dict(info=info, pt=pt, prt=prt, rows=rows, Q=Q, best=best, z=z, terms=terms, w=w, meta=meta, imgs=imgs,
                slot=slot)


def run_state(outdir):
    """What the results stores hold for a run: its dir and run.json mtime (the cache key)."""
    return {"dir": str(outdir), "mtime": (Path(outdir) / "run.json").stat().st_mtime}


def _data(state):
    return _load(state["dir"], state["mtime"]) if state else None


def ion_image(d, qi, channel=1):
    """(H, W) image of query qi (channel 0/1/2 = M-1/M/M+1) from viewer/, NaN off tissue; None if not saved."""
    m = d["meta"]
    if m is None or qi is None or np.isnan(qi):
        return None
    k = np.searchsorted(m["qi"], int(qi))
    if k >= len(m["qi"]) or m["qi"][k] != int(qi):
        return None
    h, w = (int(x) for x in m["shape"])
    im = np.full(h * w, np.nan, np.float32)
    im[m["pix"]] = np.asarray(d["imgs"][k, channel], np.float32) * m["scale"][k, channel]
    return im.reshape(h, w)


def _saved(d, qi, key="qi"):
    """Index of Q row qi in the viewer's saved list meta[key], or None (not saved, or a run without it)."""
    m = d["meta"]
    if m is None or key not in m or qi is None or np.isnan(qi):
        return None
    k = np.searchsorted(m[key], int(qi))
    return k if k < len(m[key]) and m[key][k] == int(qi) else None


def slot_image(d, qi):
    """(H, W) best-mobility-component M image of a switched query from viewer/, NaN off tissue; None if not saved."""
    k = _saved(d, qi, "slot_qi")
    if k is None or d["slot"] is None:
        return None
    m = d["meta"]
    h, w = (int(x) for x in m["shape"])
    im = np.full(h * w, np.nan, np.float32)
    im[m["pix"]] = np.asarray(d["slot"][k], np.float32) * m["slot_scale"][k]
    return im.reshape(h, w)


# ------------------------------------------------------------------ figures (pure functions of the loaded data)
def _fig(title="", **kw):
    f = go.Figure()
    f.update_layout(template=TEMPLATE, title=title, **kw)
    return f


def empty_fig(msg):
    f = _fig()
    f.update_layout(annotations=[dict(text=msg, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
                                      font=dict(size=14))], xaxis_visible=False, yaxis_visible=False, height=260)
    return f


def _qcurve(q):
    q = np.sort(q[np.isfinite(q)])
    return q, np.arange(1, len(q) + 1)


def ids_y_range(d, x_hi):
    """y range of IDs vs FDR that fits the curves inside [0, x_hi] (the visible part), so a long tail of
    q ~ 1 targets cannot flatten the curves."""
    pt, prt = d["pt"], d["prt"]
    n = max(int(((pt.q <= x_hi) & ~pt.is_decoy).sum()), int(((prt.q_picked <= x_hi) & ~prt.is_decoy).sum()), 1)
    return [0, n * 1.08 + 1]


def ids_vs_fdr(d, thr):
    pt, prt = d["pt"], d["prt"]
    f = _fig("IDs vs FDR", xaxis_title="q-value", yaxis_title="targets reported", height=340)
    for x, name, color in ((pt.q[~pt.is_decoy], "peptides", BLUE), (prt.q_picked[~prt.is_decoy], "proteins (picked)", GREEN)):
        q, n = _qcurve(x.to_numpy(float))
        f.add_trace(go.Scatter(x=q, y=n, mode="lines", line=dict(color=color, width=3, shape="hv"), name=name,
                               hovertemplate="q %{x:.4f}<br>%{y} reported<extra>" + name + "</extra>"))
    if d["info"].get("gt"):
        g = pt[pt.is_gt & ~pt.is_decoy]
        q, n = _qcurve(g.q.to_numpy(float))
        f.add_trace(go.Scatter(x=q, y=n, mode="lines", line=dict(color=PURPLE, width=2, dash="dash", shape="hv"),
                               name="GT peptides"))
    f.add_vline(x=thr, line=dict(color=ORANGE, width=2, dash="dot"), annotation_text=f"{thr:.0%}",
                annotation_font_color=ORANGE)
    f.update_xaxes(range=[0, 0.1])
    f.update_yaxes(range=ids_y_range(d, 0.1))
    return f


def _log_ticks(vmax):
    """Tick positions on a log10(1 + n) axis and their labels (true counts), both directions of a mirror."""
    vals = [v for v in (1, 10, 100, 1000, 10000, 100000) if v <= max(vmax, 1) * 1.5]
    pos = [np.log10(1 + v) for v in vals]
    return [-x for x in pos[::-1]] + pos, [f"{v:,}" for v in vals[::-1]] + [f"{v:,}" for v in vals]  # 0 = zeroline


def _k(dec):
    """Decoys per target (dec: bool array)."""
    return dec.sum() / max((~dec).sum(), 1)


def _mirror_bars(f, x, dec, edges, unit, target_kw=None, **at):
    """Mirror histogram: targets per bin up, decoys / k down (k = decoys per target), both on log10(1 + count) so a
    tail bin of 1 stays visible next to a peak of ~1000. Hover gives true counts. target_kw: extra target Bar args."""
    k = _k(dec)
    mid, w = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
    nt, nd = np.histogram(x[~dec], edges)[0], np.histogram(x[dec], edges)[0] / k
    f.add_trace(go.Bar(**dict(dict(x=mid, y=np.log10(1 + nt), width=w, name=f"target {unit}", marker_color=BLUE,
                                   text=nt, textposition="none", legendgroup="t",
                                   hovertemplate=f"%{{x:.3g}}: %{{text}} target {unit}<extra></extra>"),
                              **(target_kw or {}))), **at)
    f.add_trace(go.Bar(x=mid, y=-np.log10(1 + nd), width=w, name=f"decoy {unit} / {k:.1f}", marker_color=ORANGE,
                       text=np.round(nd, 1), textposition="none", legendgroup="d",
                       hovertemplate=f"%{{x:.3g}}: %{{text}} decoy {unit} / k<extra></extra>"), **at)
    tv, tt = _log_ticks(max(nt.max(), nd.max()))
    f.update_yaxes(tickvals=tv, ticktext=tt, zeroline=True, zerolinecolor=NAVY, zerolinewidth=1, **at)


def _mirror(f, pt):
    """Mirror histogram of peptide scores; a target bar's hover lists its peptides and its customdata is the bin's
    best peptide (click opens it)."""
    s, dec = pt.s.to_numpy(float), pt.is_decoy.to_numpy(bool)
    edges = np.linspace(np.quantile(s, 0.002), s.max() + 1e-9, 91)
    tg = pt[~dec].sort_values("s", ascending=False)
    b = np.clip(np.searchsorted(edges, tg.s.to_numpy(), "right") - 1, 0, len(edges) - 2)
    names = pd.Series(tg.index, index=b).groupby(level=0).apply(list).reindex(range(len(edges) - 1))
    best = [n[0] if isinstance(n, list) else "" for n in names]
    listing = ["<br>".join(n[:6]) + (f"<br>… {len(n) - 6} more" if len(n) > 6 else "") if isinstance(n, list) else ""
               for n in names]
    _mirror_bars(f, s, dec, edges, "peptides", dict(customdata=best, hovertext=listing,
                 hovertemplate="s %{x:.2f}: %{text} targets<br>%{hovertext}<extra></extra>"))
    f.update_yaxes(title_text="decoys / k  |  targets")


def tail_counts(pt, n=300):
    """(s grid, targets with score >= s, decoys / k with score >= s) over the upper half of the target scores."""
    s, dec = pt.s.to_numpy(float), pt.is_decoy.to_numpy(bool)
    k = _k(dec)
    grid = np.linspace(np.median(s[~dec]), s.max(), n)
    ts, ds = np.sort(s[~dec]), np.sort(s[dec])
    return grid, len(ts) - np.searchsorted(ts, grid), (len(ds) - np.searchsorted(ds, grid)) / k


LEGEND_TOP = dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom", bordercolor="rgba(0,0,0,0)")


def _cut(pt, thr):
    """Lowest target score reported at q <= thr, and how many targets that is."""
    m = (pt.q <= thr) & ~pt.is_decoy
    return pt.s[m].min(), int(m.sum())


def _cut_line(f, pt, thr):
    cut, n = _cut(pt, thr)
    if np.isfinite(cut):
        f.add_vline(x=cut, line=dict(color=NAVY, dash="dot"), annotation_text=f"{n} at q <= {thr:.0%}")


def mirror_fig(d, thr):
    f = _fig("Peptide scores (click a target bar to open its best peptide)", xaxis_title="score s (best row)",
             height=380, bargap=0, barmode="overlay", legend=LEGEND_TOP, margin=dict(t=80))
    _mirror(f, d["pt"])
    _cut_line(f, d["pt"], thr)
    return f


def tail_fig(d, thr):
    """Targets vs decoys / k scoring at least s: the gap is the estimated number of true matches, the ratio the FDR
    estimate at that s."""
    grid, T, D = tail_counts(d["pt"])
    f = _fig("Score tail: peptides scoring at least s", xaxis_title="score s", yaxis_title="peptides with score >= s",
             height=340, legend=LEGEND_TOP, margin=dict(t=80))
    f.add_trace(go.Scatter(x=grid, y=T, mode="lines", line=dict(color=BLUE, width=3, shape="hv"), name="targets",
                           customdata=np.c_[D, D / np.maximum(T, 1)],
                           hovertemplate="s >= %{x:.2f}<br>%{y} targets<br>%{customdata[0]:.1f} decoys / k"
                                         "<br>FDR estimate %{customdata[1]:.3f}<extra></extra>"))
    f.add_trace(go.Scatter(x=grid, y=np.maximum(D, 0.1), mode="lines", line=dict(color=ORANGE, width=2, shape="hv"),
                           name="decoys / k", hoverinfo="skip"))
    _cut_line(f, d["pt"], thr)
    f.update_yaxes(type="log")
    return f


def pp_plot(d):
    """Decoy vs target score CDF: on the diagonal where targets are mostly false (decoys calibrated), below it where
    true targets pile up at high scores."""
    pt = d["pt"]
    dsc, tsc = np.sort(pt.s[pt.is_decoy].to_numpy()), np.sort(pt.s[~pt.is_decoy].to_numpy())
    f = _fig("PP plot (decoy vs target score CDF)", xaxis_title="decoy CDF", yaxis_title="target CDF", height=320)
    f.add_trace(go.Scatter(x=np.searchsorted(dsc, tsc, "right") / len(dsc), y=np.arange(1, len(tsc) + 1) / len(tsc),
                           mode="lines", line=dict(color=BLUE, width=2), showlegend=False,
                           hovertemplate="decoy CDF %{x:.3f}<br>target CDF %{y:.3f}<extra></extra>"))
    f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREY, dash="dot"), showlegend=False))
    f.update_yaxes(scaleanchor="x")
    return f


def term_dists(d):
    """The score's terms per candidate row as mirror histograms, like the score: target rows up, decoy rows / k down
    (k = decoy rows per target row), log10(1 + count) so the few high-evidence targets stay visible. 5 per row."""
    rows, terms, w = d["rows"], d["terms"], d["w"]
    dec = rows.is_decoy.to_numpy(bool)
    nr = (len(terms) + 4) // 5
    f = make_subplots(nr, 5, subplot_titles=[f"{t} ({x:+g})" for t, x in zip(terms, w)], horizontal_spacing=0.065,
                      vertical_spacing=0.22)
    for i, t in enumerate(terms):
        x = rows[t].to_numpy(float)
        _mirror_bars(f, x, dec, np.linspace(np.nanmin(x), np.nanmax(x) + 1e-9, 51), "rows",
                     row=i // 5 + 1, col=i % 5 + 1)
    f.for_each_trace(lambda tr: tr.update(showlegend=False) if tr.xaxis != "x" else None)
    f.update_layout(template=TEMPLATE, barmode="overlay", bargap=0, height=90 + 250 * nr,
                    title=f"The {len(terms)} terms (weight in the score)", margin=dict(t=90),
                    legend=dict(LEGEND_TOP, y=1 + 0.16 / nr))
    f.update_yaxes(tickfont=dict(size=11))
    for r in range(1, nr + 1):
        f.update_yaxes(title_text="decoys / k  |  targets", row=r, col=1)
    f.update_xaxes(nticks=4, tickfont=dict(size=11))
    return f


def score_vs_mz(d, thr):
    pt = d["pt"].dropna(subset=["mz"])
    f = _fig("Score vs m/z (click a point to open the peptide)", xaxis_title="query m/z", yaxis_title="score s",
             height=380)
    dec, rep = pt.is_decoy, (pt.q <= thr) & ~pt.is_decoy
    for m, name, color, size in ((dec, "decoys", GREY, 4), (~dec & ~rep, "targets", CYAN, 5),
                                 (rep, f"targets at q <= {thr:.0%}", ORANGE, 8)):
        sub = pt[m]
        f.add_trace(go.Scattergl(x=sub.mz, y=sub.s, mode="markers", name=name, customdata=sub.index,
                                 marker=dict(color=color, size=size, opacity=0.55 if name == "decoys" else 0.85),
                                 hovertemplate="%{customdata}<br>m/z %{x:.4f}<br>s %{y:.2f}<extra></extra>"))
    g = pt[pt.is_gt]
    if len(g):
        f.add_trace(go.Scattergl(x=g.mz, y=g.s, mode="markers", name="GT", customdata=g.index,
                                 marker=dict(color="rgba(0,0,0,0)", size=13, line=dict(color=PURPLE, width=2)),
                                 hovertemplate="GT %{customdata}<br>s %{y:.2f}<extra></extra>"))
    return f


def tissue_figs(d):
    m = d["meta"]
    if m is None:
        return empty_fig("viewer/ not found (run made before viewer output)")
    h, w = (int(x) for x in m["shape"])
    tic = m["tic"].reshape(h, w)
    on = np.zeros(h * w)
    on[m["pix"]] = 1
    f = make_subplots(1, 2, subplot_titles=("frame TIC (log10)", "on-tissue pixels"), horizontal_spacing=0.08)
    with np.errstate(divide="ignore"):
        f.add_trace(go.Heatmap(z=np.log10(np.where(tic > 0, tic, np.nan)), colorscale="Viridis", showscale=False,
                               hovertemplate="x %{x}, y %{y}<br>log10 TIC %{z:.2f}<extra></extra>"), 1, 1)
    f.add_trace(go.Heatmap(z=np.where(np.isnan(tic), np.nan, on.reshape(h, w)), showscale=False,
                           colorscale=[[0, "#e6eaf3"], [1, NAVY]], hoverinfo="skip"), 1, 2)
    f.update_layout(template=TEMPLATE, height=360)
    for i in (1, 2):
        f.update_yaxes(autorange="reversed", scaleanchor=f"x{'' if i == 1 else i}", showgrid=False, visible=False,
                       row=1, col=i)
        f.update_xaxes(showgrid=False, visible=False, row=1, col=i)
    return f


def mz_dist(d):
    rows = d["rows"]
    f = _fig("Query m/z of all candidates", xaxis_title="m/z", yaxis_title="density", barmode="overlay", height=300)
    for dec, name, color in ((False, "targets", BLUE), (True, "decoys", ORANGE)):
        f.add_trace(go.Histogram(x=rows.feature_mz[rows.is_decoy == dec], name=name, marker_color=color,
                                 opacity=0.6, nbinsx=80, histnorm="probability density"))
    return f


def centroid_dist(d, thr):
    """The flank term's input: the apex estimate dgrad (bench 2 runs; ppm from the offset, median 0) or the
    top-decile centroid (bench 1 runs; ppm from the query, with the offset line)."""
    pt, Q = d["pt"], d["Q"]
    o = d["info"]["mz_offset"]["used"] or 0
    col, mark = ("dgrad", 0) if "dgrad" in Q else ("cen", o)
    f = _fig("Peak apex of the brightest pixels (dgrad)" if col == "dgrad" else "m/z centroid of the brightest pixels",
             xaxis_title="ppm from the offset" if col == "dgrad" else "ppm from the query m/z",
             yaxis_title="density", barmode="overlay", height=300)
    rep = pt[(pt.q <= thr) & ~pt.is_decoy].qi.dropna().astype(int)
    v = Q[col].to_numpy()
    lo, hi = np.nanpercentile(v[pt.qi.dropna().astype(int)], [1, 99])
    f.update_xaxes(range=[min(lo, mark) - 1, max(hi, mark) + 1])  # the bulk of the data
    for qi, name, color in ((pt[pt.is_decoy].qi.dropna().astype(int), "decoys", ORANGE),
                            (rep, f"targets at q <= {thr:.0%}", BLUE)):
        f.add_trace(go.Histogram(x=v[qi], name=name, marker_color=color, opacity=0.6,
                                 histnorm="probability density", xbins=dict(start=-25, end=25, size=0.5)))
    if col == "dgrad":
        f.add_vrect(x0=-1, x1=1, fillcolor=GREY, opacity=0.15, line_width=0, annotation_text="no flank penalty")
    else:
        f.add_vline(x=o, line=dict(color=NAVY, dash="dot"), annotation_text=f"offset {o:+.2f} ppm")
    return f


def _image_trace(im, show_scale=False):
    return go.Heatmap(z=im, colorscale=ION_SCALE, showscale=show_scale, hoverongaps=False,
                      hovertemplate="x %{x}, y %{y}<br>intensity %{z:.3g}<extra></extra>")


def _square(f, n):
    f.update_xaxes(visible=False, showgrid=False)
    f.update_yaxes(autorange="reversed", visible=False, showgrid=False)
    for i in range(1, n + 1):
        f.layout[f"yaxis{'' if i == 1 else i}"].scaleanchor = f"x{'' if i == 1 else i}"


def peptide_images(d, pep):
    qi = d["pt"].qi.get(pep)
    ims = [ion_image(d, qi, c) for c in (0, 1, 2)]
    if ims[1] is None:
        return empty_fig("Ion images not saved for this peptide (config: viewer_all_targets / viewer_all_decoys)")
    f = make_subplots(1, 3, subplot_titles=("M-1", "M", "M+1"), horizontal_spacing=0.03,
                      shared_xaxes="all", shared_yaxes="all")
    for i, im in enumerate(ims, 1):
        f.add_trace(_image_trace(im), 1, i)
    f.update_layout(template=TEMPLATE, height=330, margin=dict(l=10, r=10, t=40, b=10))
    _square(f, 3)
    return f


def envelope_fig(d, pep):
    qi = d["pt"].qi.get(pep)
    if qi is None or np.isnan(qi):
        return empty_fig("no evidence row")
    qi, Q = int(qi), d["Q"]
    env = Q.loc[qi, ["env0", "env1", "env2"]].to_numpy(float)
    ref = averagine_dist3((float(Q.mz[qi]),))[0]
    f = _fig(f"Envelope vs averagine (KL {d['best'].kl.get(pep, np.nan):.3g})", yaxis_title="fraction", height=300,
             barmode="group")
    f.add_trace(go.Bar(x=["M", "M+1", "M+2"], y=env / np.nansum(env), name="observed", marker_color=BLUE))
    f.add_trace(go.Bar(x=["M", "M+1", "M+2"], y=ref, name="averagine", marker_color=GREY))
    return f


def profile_fig(d, pep):
    qi, m = d["pt"].qi.get(pep), d["meta"]
    if m is None or qi is None or np.isnan(qi):
        return empty_fig("no m/z profile saved")
    qi, k = int(qi), _saved(d, qi)
    prof = m["prof"][qi]
    half = (len(prof) - 1) // 2
    if "dgrad" in d["Q"]:  # bench 2: unsaved queries show the +-25 ppm M profile only (empty bins are skipped)
        p2 = m["prof2"][k] if k is not None else np.zeros((2, 2 * PW + 1))
        if k is None:
            p2[1, PW - half:PW + half + 1] = prof
        return profile2_fig(d, qi, p2)
    cen, o = float(d["Q"].cen[qi]), d["info"]["mz_offset"]["used"] or 0.0
    f = _fig(f"m/z profile, centroid {cen:+.2f} ppm", xaxis_title="ppm from the query m/z", yaxis_title="intensity",
             height=300)
    x, k = np.arange(-half, half + 1), prof > 0  # only the sampled bins: TOF spacing leaves empty 1 ppm bins
    f.add_trace(go.Scatter(x=x[k], y=prof[k], mode="lines+markers", line=dict(color=BLUE, width=2),
                           marker=dict(size=4), fill="tozeroy", fillcolor="rgba(0,96,192,0.12)", name="profile"))
    f.add_vline(x=cen, line=dict(color=ORANGE, width=2), annotation_text="centroid")
    f.add_vline(x=o, line=dict(color=NAVY, dash="dot"), annotation_text="offset", annotation_position="bottom right")
    return f


def profile2_fig(d, qi, p2):
    """Bench 2 view: the M profile (dgrad: apex, crat: the grey +-2.5 ppm window at the offset) and the M-1 profile
    (ppm from the exact M-1 position, scaled to the M max; m1c: an apex at M-1, m1g: its slope at the run's M-1
    apex shift), top-decile pixels, sampled bins only."""
    Q, info = d["Q"], d["info"]
    o = info["mz_offset"]["used"] or 0.0
    x = np.arange(-PW, PW + 1)
    f = _fig(f"m/z profiles: dgrad {Q.dgrad[qi]:+.2f} ppm, crat {Q.crat[qi]:.2f}, m1c {Q.m1c[qi]:.2f}, "
             f"m1g {Q.m1g[qi]:.2f}", xaxis_title="ppm from the expected position", yaxis_title="intensity",
             height=330, legend=LEGEND_TOP, margin=dict(t=80))
    pm, p1 = p2[1], p2[0]
    sc = pm.max() / max(p1.max(), 1e-30)
    for p, name, color, fill in ((pm, "M", BLUE, "rgba(0,96,192,0.12)"),
                                 *([(p1 * sc, f"M-1 (x{sc:.2g})", ORANGE, None)] if p1.max() > 0 else [])):
        k = p > 0  # only the sampled bins: TOF spacing leaves empty 1 ppm bins
        f.add_trace(go.Scatter(x=x[k], y=p[k], mode="lines+markers", name=name, marker=dict(size=4),
                               line=dict(color=color, width=2), fill="tozeroy" if fill else None, fillcolor=fill))
    f.add_vrect(x0=o - CRAT_PPM, x1=o + CRAT_PPM, fillcolor=GREY, opacity=0.2, line_width=0)
    f.add_vline(x=o + float(Q.dgrad[qi]), line=dict(color=BLUE, width=2), annotation_text="M apex")
    f.add_vline(x=o, line=dict(color=NAVY, dash="dot"), annotation_text="offset", annotation_position="bottom right")
    f.add_vline(x=o + info["profile"]["m1_shift"], line=dict(color=ORANGE, dash="dash"), annotation_text="run M-1 apex",
                annotation_position="top left")
    f.update_xaxes(range=[-30, 30])
    return f


def mobility_fig(d, pep):
    """The query's on-tissue M mobilogram and its fixed-width (1.5% FWHM) components (isogate1), each labelled with
    its area share and M / M+1 image r; the best one bold. The query's coloc reads that component's image when it
    switched (its r beats the full window's iso r by 0.1)."""
    qi = d["pt"].qi.get(pep)
    k, m = _saved(d, qi), d["meta"]
    if k is None or "mob_mg" not in m:
        return empty_fig("no mobilogram saved (run without mobility viewer output, or peptide not saved)")
    qi = int(qi)
    kb, y = m["mob_kb"], m["mob_mg"][k].astype(float)
    cen, area, siso, best = m["mob_cen"][k], m["mob_area"][k], m["mob_siso"][k], int(m["mob_best"][k])
    sw = bool(d["Q"].mob_switch[qi])
    f = _fig(f"Mobilogram: {'switched to component' if sw else 'full window'} (iso r {d['Q'].iso_r[qi]:.2f})",
             xaxis_title="1/K0 (V s / cm²)", yaxis_title="intensity", height=330, legend=LEGEND_TOP,
             margin=dict(t=80))
    f.add_trace(go.Scatter(x=kb, y=y, mode="lines", name="M", line=dict(color=NAVY, width=2), fill="tozeroy",
                           fillcolor="rgba(0,16,80,0.08)"))
    a = int(np.argmax(y))
    band = (kb >= kb[a] * (1 - MOB_BAND)) & (kb <= kb[a] * (1 + MOB_BAND))
    dk = np.median(np.abs(np.diff(kb)))
    tot = y[band].sum() * dk
    xx = np.linspace(kb[band].min(), kb[band].max(), 300)
    for j in np.flatnonzero(np.isfinite(cen)):
        sd = cen[j] * MOB_W / 100 / 2.3548
        g = area[j] * tot * np.exp(-0.5 * ((xx - cen[j]) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        f.add_trace(go.Scatter(x=xx, y=g, mode="lines", name=f"{area[j]:.0%}, r {siso[j]:.2f}",
                               line=dict(color=ORANGE if j == best else CYAN, width=3 if j == best else 1.5,
                                         dash=None if j == best else "dot")))
    f.update_xaxes(range=[kb[band].min(), kb[band].max()])
    return f


def mobility_images(d, pep):
    """The M image (full mobility window) next to the best component's M image, for a switched query."""
    qi = d["pt"].qi.get(pep)
    full, comp = ion_image(d, qi, 1), slot_image(d, qi)
    if comp is None:
        return None
    f = make_subplots(1, 2, subplot_titles=("M, full mobility window", "M, best component (what coloc reads)"),
                      horizontal_spacing=0.03, shared_xaxes="all", shared_yaxes="all")
    f.add_trace(_image_trace(full), 1, 1)
    f.add_trace(_image_trace(comp), 1, 2)
    f.update_layout(template=TEMPLATE, height=330, margin=dict(l=10, r=10, t=40, b=10))
    _square(f, 2)
    return f


def head_anatomy(d, thr):
    """The profile features of the score's head: the targets at q <= thr against the same number (at least 50) of
    top-scoring decoys. Shows which feature keeps decoys out of the reported set."""
    pt, Q = d["pt"], d["Q"]
    if "dgrad" not in Q:
        return empty_fig("run without profile features (bench 1 score)")
    rep = pt[(pt.q <= thr) & ~pt.is_decoy].dropna(subset=["qi"])
    dec = pt[pt.is_decoy].dropna(subset=["qi"]).nlargest(max(50, len(rep)), "s")
    f = make_subplots(1, 2, subplot_titles=("M peak: apex dgrad vs centre ratio crat",
                                            "M-1: centre ratio m1c vs slope m1g"), horizontal_spacing=0.1)
    for sub, name, color in ((dec, f"top {len(dec)} decoys", ORANGE), (rep, f"targets at q <= {thr:.0%}", BLUE)):
        q = sub.qi.astype(int).to_numpy()
        for col, (a, b) in enumerate((("dgrad", "crat"), ("m1c", "m1g")), 1):
            f.add_trace(go.Scattergl(x=Q[a].to_numpy()[q], y=Q[b].to_numpy()[q], mode="markers", name=name,
                                     showlegend=col == 1, customdata=sub.index, marker=dict(color=color, size=6,
                                     opacity=0.7), hovertemplate="%{customdata}<br>" + a + " %{x:.2f}<br>" + b
                                     + " %{y:.2f}<extra></extra>"), 1, col)
            f.update_xaxes(title_text=a, row=1, col=col)
            f.update_yaxes(title_text=b, row=1, col=col)
    f.update_layout(template=TEMPLATE, height=380, legend=LEGEND_TOP, margin=dict(t=80))
    return f


def terms_fig(d, pep):
    b = d["best"]
    if pep not in b.index:
        return empty_fig("this peptide has no surviving row")
    rows = d["rows"]
    idx = rows.index[(rows.peptide == pep) & ~rows.removed]
    zz = d["z"].loc[idx].iloc[int(np.argmax(rows.s[idx].to_numpy()))]
    f = _fig(f"Term contributions (z x weight), s {b.s[pep]:.2f}", xaxis_title="contribution to s0",
             height=120 + 30 * len(d["terms"]))
    f.add_trace(go.Bar(y=d["terms"], x=zz.to_numpy(), orientation="h",
                       marker_color=[BLUE if v >= 0 else ORANGE for v in zz.to_numpy()],
                       customdata=b.loc[pep, d["terms"]].to_numpy(float),
                       hovertemplate="%{y}: %{x:+.2f} (raw %{customdata:.3g})<extra></extra>"))
    f.update_yaxes(autorange="reversed")
    return f


def score_position(d, pep):
    pt = d["pt"]
    f = _fig("Where this score sits", xaxis_title="peptide score s", height=340, bargap=0, barmode="overlay",
             legend=LEGEND_TOP, margin=dict(t=80))
    _mirror(f, pt)
    if pep in pt.index:
        f.add_vline(x=pt.s[pep], line=dict(color=RED, width=3), annotation_text=f"q {pt.q[pep]:.3f}")
    return f


def protein_images(d, prot, thr):
    pt = d["pt"]
    p = pt[(pt.protein == prot) & ~pt.is_decoy].sort_values("s", ascending=False)
    ims = [ion_image(d, qi) for qi in p.qi]
    keep = [im is not None for im in ims]
    p, ims = p[keep].head(N_PROT_IMAGES), [im for im in ims if im is not None][:N_PROT_IMAGES]
    if p.empty:
        return empty_fig("No ion images saved for this protein's peptides"), empty_fig("")
    n = len(p)
    rws = (n + 3) // 4
    titles = [f"{pep[:16]}{'…' if len(pep) > 16 else ''}<br>q {q:.3f}{' ✓' if q <= thr else ''}"
              for pep, q in zip(p.index, p.q)]
    f = make_subplots(rws, 4, subplot_titles=titles, horizontal_spacing=0.02, vertical_spacing=0.12 / rws,
                      shared_xaxes="all", shared_yaxes="all")
    for i, im in enumerate(ims):
        f.add_trace(_image_trace(im), i // 4 + 1, i % 4 + 1)
    f.update_layout(template=TEMPLATE, height=230 * rws + 40, margin=dict(l=10, r=10, t=50, b=10))
    f.update_annotations(font=dict(size=11, family=TEXT))
    _square(f, n)
    # image correlation (plain Pearson of the on-tissue M images; the score's coloc also removes the common mode)
    X = np.array([im[np.isfinite(im)] for im in ims], np.float64)
    r = np.atleast_2d(np.corrcoef(X))
    labels = [x if len(x) <= 12 else x[:11] + "…" for x in p.index]
    c = _fig("Image correlation of these peptides (Pearson r)", height=300 + 12 * n)
    c.add_trace(go.Heatmap(z=r, x=labels, y=labels, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
                           hovertemplate="%{x} / %{y}<br>r %{z:.2f}<extra></extra>"))
    c.update_yaxes(autorange="reversed", automargin=True)
    c.update_xaxes(tickangle=-45, automargin=True)
    return f, c


# ------------------------------------------------------------------ tables
def _grid(gid, cols, height=420, extra=None, rows=()):
    return dag.AgGrid(
        id=gid, columnDefs=cols, rowData=list(rows), columnSize="responsiveSizeToFit", className="grid-wrap",
        defaultColDef={"sortable": True, "filter": True, "resizable": True, "minWidth": 80},
        dashGridOptions={"theme": GRID_THEME, "rowSelection": {"mode": "singleRow", "checkboxes": False,
                         "enableClickSelection": True}, "animateRows": False, "pagination": True,
                         "paginationPageSize": 50, **(extra or {})},
        csvExportParams={"fileName": f"{gid}.csv"}, style={"height": f"{height}px"})


PROT_COLS = [dict(field="protein", pinned="left", minWidth=140), dict(field="q_picked", valueFormatter=NUM),
             dict(field="q", valueFormatter=NUM), dict(field="s", headerName="best s", valueFormatter=NUM),
             dict(field="n_peptides"), dict(field="n_reported"), dict(field="n_gt", headerName="GT"),
             dict(field="decoy")]
PEP_COLS = [dict(field="peptide", pinned="left", minWidth=200), dict(field="protein"), dict(field="s", valueFormatter=NUM),
            dict(field="q", valueFormatter=NUM), dict(field="mz", headerName="m/z", valueFormatter=NUM),
            dict(field="decoy"), dict(field="gt", headerName="GT"), dict(field="seed"), dict(field="images")]


def protein_rows(d, thr, decoys):
    pt, prt = d["pt"], d["prt"]
    t = pt[~pt.is_decoy]
    g = t.groupby("protein")
    pr = prt if decoys else prt[~prt.is_decoy]
    out = pd.DataFrame({"protein": pr.protein, "q_picked": pr.q_picked, "q": pr.q,
                        "s": pr.s, "decoy": pr.is_decoy})
    out["n_peptides"] = out.protein.map(g.size()).fillna(0).astype(int)
    out["n_reported"] = out.protein.map(g.q.apply(lambda q: int((q <= thr).sum()))).fillna(0).astype(int)
    out["n_gt"] = out.protein.map(g.is_gt.sum()).fillna(0).astype(int)
    return out.sort_values(["q_picked", "s"], ascending=[True, False]).to_dict("records")


def peptide_rows(d, decoys, protein=None):
    pt = d["pt"]
    p = pt if decoys else pt[~pt.is_decoy]
    if protein is not None:
        p = p[p.protein == protein]
    saved = set(d["meta"]["qi"].tolist()) if d["meta"] is not None else set()
    out = pd.DataFrame({"peptide": p.index, "protein": p.protein.to_numpy(), "s": p.s.to_numpy(), "q": p.q.to_numpy(),
                        "mz": p.mz.to_numpy(), "decoy": p.is_decoy.to_numpy(), "gt": p.is_gt.to_numpy(),
                        "seed": p.seed.to_numpy(), "images": [qi in saved for qi in p.qi.fillna(-1).astype(int)]})
    return out.sort_values("q").to_dict("records")


def _small_table(df):
    return dag.AgGrid(rowData=df.to_dict("records"), columnDefs=[
        dict(field=c, valueFormatter=NUM) if df[c].dtype.kind == "f" else dict(field=c) for c in df.columns],
        columnSize="responsiveSizeToFit", className="grid-wrap", dashGridOptions={"theme": GRID_THEME, "domLayout": "autoHeight"},
        defaultColDef={"sortable": True, "resizable": True})


def _stat(value, label):
    return html.Div(className="stat", children=[html.Div(str(value), className="stat-value"),
                                                html.Div(label, className="stat-label")])


def _pill(text, cls):
    return html.Span(text, className=f"pill {cls}")


def _graph(fig, gid=None):
    kw = {"id": gid} if gid else {}
    return dcc.Graph(figure=fig, config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}}, **kw)


# ------------------------------------------------------------------ layout
def layout():
    nav = [html.Button("Run QC" if p == "qc" else p.capitalize(),
                       id={"role": "subnav", "page": p}, n_clicks=0, className="btn btn-small") for p in PAGES]
    page = lambda p, kids: html.Div(id=f"page-{p}", children=kids, hidden=p != "overview")
    return html.Div(className="content", children=[
        dcc.Store(id="res-state"), dcc.Store(id="sel-protein"), dcc.Store(id="sel-peptide"),
        dcc.Store(id="res-page", data="overview"),
        html.Div(className="card", children=[
            html.Div(className="res-strip", children=[
                html.Div(style={"flex": "3 1 360px"}, children=[html.Label("Results directory"), html.Div(
                    style={"display": "flex", "gap": "8px"}, children=[
                        dcc.Input(id="results-dir", type="text", value="", debounce=True),
                        html.Button("Browse", id="browse-results-dir", n_clicks=0, className="btn btn-small"),
                        html.Button("Load", id="load-results", n_clicks=0, className="btn btn-small")])]),
                html.Div(style={"flex": "0 1 130px"}, children=[html.Label("q threshold"), dcc.Dropdown(
                    id="res-q", options=[{"label": f"{q:.0%}", "value": q} for q in QS], value=0.05,
                    clearable=False)]),
                html.Div(style={"flex": "2 1 220px"}, children=[html.Label("Search"), dcc.Input(
                    id="res-search", type="text", placeholder="peptide or protein", debounce=False)]),
                html.Div(style={"flex": "0 1 150px"}, children=[html.Label("Decoys"), dcc.Checklist(
                    id="res-decoys", options=[{"label": " show decoys", "value": "on"}], value=[])]),
            ]),
            html.Div(id="res-msg", className="hint", style={"marginTop": "8px"}),
            html.Div(id="res-stats", className="stat-row", style={"marginTop": "12px"}),
        ]),
        html.Div(className="res-shell", children=[
            html.Div(className="subnav", children=nav),
            html.Div(style={"minWidth": 0}, children=[
                page("overview", [html.Div(id="ov-body")]),
                page("proteins", [
                    html.Div(className="card", children=[
                        html.H2("Proteins"), _grid("prot-grid", PROT_COLS),
                        html.Button("Export CSV", id="prot-csv", n_clicks=0, className="btn btn-small",
                                    style={"marginTop": "10px"})]),
                    html.Div(id="prot-detail")]),
                page("peptides", [
                    html.Div(className="card", children=[
                        html.H2("Peptides"), _grid("pep-grid", PEP_COLS),
                        html.Button("Export CSV", id="pep-csv", n_clicks=0, className="btn btn-small",
                                    style={"marginTop": "10px"})]),
                    html.Div(id="pep-detail")]),
                page("qc", [html.Div(id="qc-body")]),
            ]),
        ]),
    ])


# ------------------------------------------------------------------ callbacks
def register(app):
    @app.callback(Output("res-state", "data"), Output("res-msg", "children"),
                  Input("load-results", "n_clicks"), Input("results-dir", "n_submit"), State("results-dir", "value"),
                  prevent_initial_call=True)
    def load(_n, _s, outdir):
        out = Path(outdir or "")
        need = [f for f in ("run.json", "peptides.tsv", "proteins.tsv", "rows.parquet", "evidence.parquet")
                if not (out / f).exists()]
        if not outdir or need:
            return no_update, f"Not a finished run here (missing: {', '.join(need) or 'directory'})."
        msg = "" if (out / "viewer").is_dir() else "No viewer/ in this run (made before it existed): no ion images."
        return run_state(out), msg

    @app.callback(Output("res-stats", "children"), Output("ov-body", "children"), Output("qc-body", "children"),
                  Input("res-state", "data"), Input("res-q", "value"), prevent_initial_call=True)
    def overview(state, thr):
        d = _data(state)
        if d is None:
            return no_update, no_update, no_update
        info, pt, prt = d["info"], d["pt"], d["prt"]
        tg = pt[~pt.is_decoy]
        o = info["mz_offset"]["used"]
        secs = sum(s["seconds"] for s in info.get("steps", []))
        stats = [_stat(int((tg.q <= thr).sum()), f"peptides at q <= {thr:.0%}"),
                 _stat(int(((prt.q_picked <= thr) & ~prt.is_decoy).sum()), f"proteins (picked) at {thr:.0%}"),
                 *([_stat(int((tg.q <= thr)[tg.is_gt].sum()), f"of {int(tg.is_gt.sum())} GT peptides")]
                   if info.get("gt") else []),
                 _stat(f"{info['n_targets']:,} / {info['n_decoys']:,}", "target / decoy peptides"),
                 _stat("–" if o is None else f"{o:+.2f} ppm", "m/z offset used"),
                 _stat(f"{int(secs // 60)}:{int(secs % 60):02d}", "run time")]
        ov = [html.Div(className="card", children=[html.H2("Identifications"), html.Div(className="row", children=[
                  html.Div(className="col", children=_graph(ids_vs_fdr(d, thr), "ids-fig")),
                  html.Div(className="col", children=_graph(tail_fig(d, thr)))])]),
              html.Div(className="card", children=[html.H2("Scores"), _graph(mirror_fig(d, thr), "scorehist-fig"),
                                                    _graph(score_vs_mz(d, thr), "scatter-fig"),
                                                    _graph(term_dists(d))])]
        qc = [html.Div(className="card", children=[html.H2("Tissue"), _graph(tissue_figs(d))]),
              html.Div(className="card", children=[html.H2("m/z"), html.Div(className="row", children=[
                  html.Div(className="col", children=_graph(mz_dist(d))),
                  html.Div(className="col", children=_graph(centroid_dist(d, thr)))])]),
              html.Div(className="card", children=[html.H2("Score calibration"), html.Div(className="row", children=[
                  html.Div(className="col", children=_graph(pp_plot(d))), html.Div(className="col")])])]
        pr = info.get("profile")
        if pr:
            qc.insert(2, html.Div(className="card", children=[
                html.H2("Run constants (estimated label-free from this run)"), html.Div(className="stat-row", children=[
                    _stat(f"{pr['sigma_run']:.1f} ppm", "m/z peak width (sigma), scales dgrad and m1g"),
                    _stat(f"{pr['m1_shift']:+.1f} ppm", "M-1 apex shift vs M (m1g reads the slope there)"),
                    _stat(f"{info['mob_switched']:,}", "queries switched to a mobility component")]),
                html.H3("Head anatomy"), _graph(head_anatomy(d, thr))]))
        return stats, ov, qc

    @app.callback(Output("ids-fig", "figure"), Input("ids-fig", "relayoutData"), State("res-state", "data"),
                  prevent_initial_call=True)
    def refit_ids(rel, state):  # y follows the visible x range on zoom / pan
        d = _data(state)
        if d is None or not rel:
            return no_update
        hi = rel.get("xaxis.range[1]", (rel.get("xaxis.range") or [None, None])[1])
        if rel.get("xaxis.autorange"):
            hi = 1.0
        if hi is None:
            return no_update
        p = Patch()
        p["layout"]["yaxis"]["range"] = ids_y_range(d, float(hi))
        return p

    @app.callback(Output("prot-grid", "rowData"), Output("pep-grid", "rowData"),
                  Input("res-state", "data"), Input("res-q", "value"), Input("res-decoys", "value"),
                  prevent_initial_call=True)
    def tables(state, thr, decoys):
        d = _data(state)
        if d is None:
            return no_update, no_update
        return protein_rows(d, thr, bool(decoys)), peptide_rows(d, bool(decoys))

    @app.callback(Output("prot-grid", "dashGridOptions"), Output("pep-grid", "dashGridOptions"),
                  Input("res-search", "value"))
    def search(text):
        a, b = Patch(), Patch()
        a["quickFilterText"] = b["quickFilterText"] = text or ""
        return a, b

    @app.callback(Output("prot-grid", "exportDataAsCsv"), Input("prot-csv", "n_clicks"), prevent_initial_call=True)
    def prot_csv(_n):
        return True

    @app.callback(Output("pep-grid", "exportDataAsCsv"), Input("pep-csv", "n_clicks"), prevent_initial_call=True)
    def pep_csv(_n):
        return True

    # navigation: sub-nav buttons, a protein / peptide pick, a scatter click
    @app.callback(Output("res-page", "data", allow_duplicate=True),
                  Input({"role": "subnav", "page": ALL}, "n_clicks"), prevent_initial_call=True)
    def nav(_clicks):
        return ctx.triggered_id["page"] if ctx.triggered_id and any(_clicks) else no_update

    @app.callback(*[Output(f"page-{p}", "hidden") for p in PAGES],
                  Output({"role": "subnav", "page": ALL}, "className"), Input("res-page", "data"))
    def show(page):
        return (*[p != page for p in PAGES], [f"btn btn-small{' active' if p == page else ''}" for p in PAGES])

    @app.callback(Output("sel-protein", "data", allow_duplicate=True), Input("prot-grid", "selectedRows"),
                  prevent_initial_call=True)
    def pick_protein(sel):
        return sel[0]["protein"] if sel else no_update

    @app.callback(Output("sel-peptide", "data", allow_duplicate=True), Input("pep-grid", "selectedRows"),
                  prevent_initial_call=True)
    def pick_peptide(sel):
        return sel[0]["peptide"] if sel else no_update

    # two callbacks: each source exists at a different time (the protein's grid only once a protein is open)
    @app.callback(Output("sel-peptide", "data", allow_duplicate=True), Output("res-page", "data", allow_duplicate=True),
                  Input("scatter-fig", "clickData"), prevent_initial_call=True)
    def scatter_to_peptide(click):
        return (click["points"][0]["customdata"], "peptides") if click else (no_update, no_update)

    @app.callback(Output("sel-peptide", "data", allow_duplicate=True), Output("res-page", "data", allow_duplicate=True),
                  Input("scorehist-fig", "clickData"), prevent_initial_call=True)
    def scorehist_to_peptide(click):  # a target bar opens the best peptide of its bin
        pep = (click or {}).get("points", [{}])[0].get("customdata")
        return (pep, "peptides") if isinstance(pep, str) and pep else (no_update, no_update)

    @app.callback(Output("sel-protein", "data", allow_duplicate=True), Output("res-page", "data", allow_duplicate=True),
                  Input("pep-protein-link", "n_clicks"), State("pep-protein-link", "children"),
                  prevent_initial_call=True)
    def peptide_to_protein(n, prot):  # the protein name on a peptide page opens that protein
        return (prot, "proteins") if n else (no_update, no_update)

    @app.callback(Output("sel-peptide", "data", allow_duplicate=True), Output("res-page", "data", allow_duplicate=True),
                  Input("protpep-grid", "selectedRows"), prevent_initial_call=True)
    def protein_to_peptide(sel):
        return (sel[0]["peptide"], "peptides") if sel else (no_update, no_update)

    @app.callback(Output("prot-detail", "children"), Input("sel-protein", "data"), Input("res-q", "value"),
                  State("res-state", "data"), State("res-decoys", "value"), prevent_initial_call=True)
    def protein_detail(prot, thr, state, decoys):
        d = _data(state)
        if d is None or not prot:
            return no_update
        pr = d["prt"][d["prt"].protein == prot]
        row = pr[~pr.is_decoy].iloc[0] if (~pr.is_decoy).any() else pr.iloc[0]
        imgs, corr = protein_images(d, prot, thr)
        peps = pd.DataFrame(peptide_rows(d, bool(decoys), prot))
        return html.Div(className="card", children=[
            html.Div(prot, className="detail-title"),
            html.Div([_pill(f"picked q {row.q_picked:.3f}", "ok" if row.q_picked <= thr else "no"),
                      _pill(f"{len(peps)} peptides", "no")], style={"marginBottom": "10px"}),
            html.H3("Peptides (click one to open its evidence)"),
            _grid("protpep-grid", PEP_COLS, 300, {"pagination": False}, peps.to_dict("records")),
            html.H3(f"M ion images (best {N_PROT_IMAGES} saved peptides, each on its own scale)"), _graph(imgs),
            _graph(corr)])

    @app.callback(Output("pep-detail", "children"), Input("sel-peptide", "data"), Input("res-q", "value"),
                  State("res-state", "data"), prevent_initial_call=True)
    def peptide_detail(pep, thr, state):
        d = _data(state)
        if d is None or not pep or pep not in d["pt"].index:
            return no_update
        pt, rows = d["pt"], d["rows"]
        r = pt.loc[pep]
        pills = [_pill("decoy", "dec") if r.is_decoy else _pill("target", "no"),
                 _pill(f"q {r.q:.3f}", "ok" if r.q <= thr and not r.is_decoy else "no")]
        if r.is_gt:
            pills.append(_pill("GT", "gt"))
        mz = r.mz
        ion = rows[(abs(rows.feature_mz - mz) <= mz * ION_PPM * 1e-6)][
            ["peptide", "protein", "is_decoy", "seed", "s0", "s", "removed"]].sort_values("s0", ascending=False)
        parent = pep if not r.is_decoy else rows.loc[rows.peptide == pep, "parent_peptide"].iloc[0]
        fam = rows[rows.peptide.eq(parent) | rows.parent_peptide.eq(parent)][["peptide", "is_decoy", "seed", "feature_mz", "s", "removed"]]
        b = d["best"].loc[pep] if pep in d["best"].index else None
        kv = [("protein", html.Button(r.protein, id="pep-protein-link", n_clicks=0, className="linklike",
                                      title="Open this protein")), ("query m/z", f"{mz:.5f}"), ("score s", f"{r.s:.3f}"),
              ("coloc partners", "–" if b is None else f"{int(b.n_partners)}"),
              *([("coloc image", "best mobility component" if d["Q"].mob_switch[int(r.qi)] else "full window")]
                if "mob_switch" in d["Q"] and np.isfinite(r.qi) else []),
              ("candidate rows", f"{int(r.n_rows)}")]
        return html.Div([
            html.Div(className="card", children=[
                html.Div(pep, className="detail-title"), html.Div(pills, style={"marginBottom": "8px"}),
                html.Div(className="kv", children=[x for k, v in kv for x in (html.Span(k), html.Span(v))]),
                html.H3("Ion images (zoom one, all follow)"), _graph(peptide_images(d, pep))]),
            html.Div(className="card", children=[html.H2("Evidence"), html.Div(className="row", children=[
                html.Div(className="col", children=_graph(terms_fig(d, pep))),
                html.Div(className="col", children=_graph(score_position(d, pep)))]),
                html.Div(className="row", children=[
                    html.Div(className="col", children=_graph(envelope_fig(d, pep))),
                    html.Div(className="col", children=_graph(profile_fig(d, pep)))])]),
            html.Div(className="card", children=[html.H2("Mobility"), _graph(mobility_fig(d, pep)),
                                                 *([_graph(mi)] if (mi := mobility_images(d, pep)) else [])]),
            html.Div(className="card", children=[
                html.H2("Competition"),
                html.H3(f"All candidates on this ion (within {ION_PPM} ppm)"), _small_table(ion),
                html.H3("This peptide and its decoys" if not r.is_decoy else "This decoy, its parent and siblings"),
                _small_table(fam.sort_values("s", ascending=False))])])

