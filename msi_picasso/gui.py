"""Pixelated Dash GUI for MSI-PICASSO (palette derived from the logo).

A thin wrapper: it writes a TOML config (keys = ``cli.DEFAULTS``), launches the
``picasso`` CLI as a subprocess, tails its log, and renders the on-disk results
(run.json counts, peptides.tsv / proteins.tsv, Plotly plots, the CLI's figures/).
It never re-implements pipeline logic, so a GUI run is a CLI run.

Run:  ``picasso-gui``            (browser at http://127.0.0.1:8050)
      ``picasso-gui --desktop``  (native window, needs the ``desktop`` extra)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx, ALL, no_update

from msi_picasso.cli import DEFAULTS

# --- Logo-derived palette (must match assets/picasso.css) ---
NAVY, PURPLE, BLUE, CYAN = "#001050", "#3d0f8f", "#0060c0", "#00a0b0"
GREEN, YELLOW, ORANGE, RED = "#4fb020", "#f0b000", "#f08000", "#d82814"
PAPER, INK = "#f5f7fb", "#001050"
PIXEL_FONT = "VT323, PressStart2P, monospace"

_ASSETS = os.path.join(os.path.dirname(__file__), "assets")
_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
_CONSOLE_LOG = "gui_console.log"

# ponytail: single-user local app — one job at a time, module-global handle.
_JOB: dict = {"proc": None, "outdir": None, "cancelled": False}

# --------------------------------------------------------------------------
# Config: the form is cli.DEFAULTS, key for key. A blank field = the CLI default.
# --------------------------------------------------------------------------
_PATH_KEYS = ("maldi_d", "lcms_peptides", "output_dir")  # own widgets with a Browse button
_DECOY_METHODS = ("d1c_raw", "d1b")
_STANDARD_KEYS = ("psm_utils_reader", "tic_quantile", "tic_normalize", "mz_offset_ppm",
                  "decoy_min_shift_da", "gt_peptides", "figures")
_ADVANCED_KEYS = ("decoy_seeds", "extraction_ppm", "mz_range", "tissue_mask", "threads")
_CFG_KEYS = _STANDARD_KEYS + _ADVANCED_KEYS
# A new CLI key must get a field here, or a GUI run could not reproduce a CLI run.
assert set(_PATH_KEYS + ("decoy_method",) + _CFG_KEYS) == set(DEFAULTS), "GUI form out of sync with cli.DEFAULTS"


def _show(v):
    """Config value -> form text (TOML literal for numbers / lists, bare text for strings)."""
    if v is None or isinstance(v, (str, bool)):
        return v
    return json.dumps(v)


def _parse(s):
    """Form text -> config value: a TOML literal if it is one (0.5, 0, [0, 1], true), else the text (paths, auto)."""
    if isinstance(s, str):
        s = s.strip()
        try:
            return tomllib.loads(f"v = {s}")["v"]
        except tomllib.TOMLDecodeError:
            return s
    return s


def config_from_toml(text: str) -> tuple[dict, list[str]]:
    """[MSI-PICASSO] table -> (known keys with underscore spelling, ignored unknown keys)."""
    sec = {k.replace("-", "_"): v for k, v in tomllib.loads(text).get("MSI-PICASSO", {}).items()}
    return {k: v for k, v in sec.items() if k in DEFAULTS}, sorted(set(sec) - set(DEFAULTS))


def section_from_form(values: dict) -> dict:
    """Form values (key -> text / bool / None) -> the [MSI-PICASSO] table; blank fields are left out."""
    return {k: _parse(v) for k, v in values.items() if v not in (None, "")}


def _render_field(key: str):
    fid = {"role": "cfg", "key": key}
    if isinstance(DEFAULTS[key], bool):
        ctrl = dcc.Dropdown(id=fid, options=[{"label": "true", "value": True},
                                             {"label": "false", "value": False}],
                            placeholder=f"default: {str(DEFAULTS[key]).lower()}", clearable=True)
    else:
        ctrl = dcc.Input(id=fid, type="text", debounce=True, placeholder=f"default: {_show(DEFAULTS[key])}")
    return html.Div([html.Label(key.replace("_", " ")), ctrl])


# --------------------------------------------------------------------------
# Plotly template
# --------------------------------------------------------------------------
def _picasso_template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family=PIXEL_FONT, size=16, color=INK),
        colorway=[BLUE, ORANGE, GREEN, PURPLE, RED, CYAN, YELLOW],
        margin=dict(l=60, r=24, t=48, b=52),
        xaxis=dict(linecolor=NAVY, linewidth=2, gridcolor="#dde3ef", zeroline=False, ticks="outside"),
        yaxis=dict(linecolor=NAVY, linewidth=2, gridcolor="#dde3ef", zeroline=False, ticks="outside"),
        legend=dict(bgcolor="#ffffff", bordercolor=NAVY, borderwidth=2),
    )
    return t


TEMPLATE = _picasso_template()


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def _read_tsv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return None


# --------------------------------------------------------------------------
# Figure builders (from peptides.tsv / proteins.tsv — no pipeline re-run)
# --------------------------------------------------------------------------
def build_ids_vs_fdr(peptides: pd.DataFrame, proteins: pd.DataFrame | None) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, title="IDs vs FDR", xaxis_title="q-value", yaxis_title="reported targets")
    for df, col, name, color in ((peptides, "q", "peptides", BLUE), (proteins, "q_picked", "proteins (picked)", GREEN)):
        if df is None or col not in df:
            continue
        q = np.sort(df.loc[~df["is_decoy"], col].dropna().to_numpy())
        fig.add_trace(go.Scatter(x=q, y=np.arange(1, len(q) + 1), mode="lines", line=dict(color=color, width=3),
                                 line_shape="hv", name=name, hovertemplate="q=%{x:.4f}<br>IDs=%{y}<extra></extra>"))
    fig.add_vline(x=0.01, line=dict(color=ORANGE, width=2, dash="dot"),
                  annotation_text="1% FDR", annotation_font_color=ORANGE)
    fig.update_xaxes(range=[0, 0.1])
    return fig


def build_score_dist(peptides: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, title="Peptide score distribution",
                      xaxis_title="score s (best row per peptide)", yaxis_title="peptides", barmode="overlay")
    # blue=target, orange=decoy — CVD-safe pair from the logo; legend = 2nd encoding.
    for dec, name, color in ((False, "target", BLUE), (True, "decoy", ORANGE)):
        x = peptides.loc[peptides["is_decoy"] == dec, "s"].dropna()
        fig.add_trace(go.Histogram(x=x, name=name, marker_color=color, opacity=0.7, nbinsx=80,
                                   histnorm="probability density"))
    return fig


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, annotations=[dict(text=msg, showarrow=False,
                      font=dict(size=18, color=INK), xref="paper", yref="paper", x=0.5, y=0.5)],
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
def _preset_options():
    if not _CONFIGS_DIR.exists():
        return []
    return [{"label": p.name, "value": str(p)} for p in sorted(_CONFIGS_DIR.glob("*.toml"))]


def _labeled(label, comp):
    return html.Div([html.Label(label), comp])


def _path_field(field_id, label):
    return _labeled(label, html.Div([
        dcc.Input(id=f"in-{field_id}", type="text", value="", debounce=True,
                  style={"width": "78%", "display": "inline-block"}),
        html.Button("Browse", id=f"browse-{field_id}", n_clicks=0,
                    className="btn btn-small", style={"marginLeft": "8px"}),
    ]))


def _header():
    return html.Div([
        html.Div(className="picasso-header", children=[
            html.Img(src="/assets/MSI-PICASSO-logo-t.png", alt="MSI-PICASSO"),
        ]),
        html.Div(className="rainbow-rule"),
    ])


def _tab_configure():
    return html.Div(className="content", children=[
        html.Div(className="card", children=[
            html.H2("1 · Load a config"),
            html.Div(className="row", children=[
                html.Div(className="col", children=_labeled("Preset", dcc.Dropdown(
                    id="preset", options=_preset_options(), placeholder="Load a preset…", clearable=False))),
                html.Div(className="col", children=_labeled("Upload a .toml", dcc.Upload(
                    id="upload-toml", className="upload-box", children="⬆ Drop or click to upload a .toml"))),
            ]),
            html.P("A preset or uploaded file fills the fields below. You can then edit anything.",
                   className="hint"),
        ]),
        html.Div(className="card", children=[
            html.H2("2 · Essentials"),
            _path_field("maldi_d", "MALDI .d directory"),
            _path_field("lcms_peptides", "LC-MS/MS PSM table"),
            _path_field("output_dir", "Output directory"),
            _labeled("Decoy method", dcc.Dropdown(
                id="in-decoy_method", options=[{"label": m, "value": m} for m in _DECOY_METHODS],
                placeholder=f"default: {DEFAULTS['decoy_method']}", clearable=True)),
        ]),
        html.Div(className="card", children=[
            html.H2("3 · Settings"),
            html.Div(className="field-grid", children=[_render_field(k) for k in _STANDARD_KEYS]),
            html.Details([
                html.Summary("Advanced settings"),
                html.Div(className="details-body field-grid",
                         children=[_render_field(k) for k in _ADVANCED_KEYS]),
            ]),
            html.P("Blank = the CLI default (shown greyed). Numbers, lists ([0, 1]) and true/false are "
                   "read as TOML; anything else as text.", className="hint"),
        ]),
        html.Div(className="card", children=[
            html.H2("4 · Run"),
            html.Button("▶ RUN PICASSO", id="run-btn", n_clicks=0, className="btn btn-run"),
            html.Span(id="run-msg", className="hint", style={"marginLeft": "16px"}),
        ]),
    ])


def _tab_log():
    return html.Div(className="content", children=[
        html.Div(className="card", children=[
            html.H2("Live log"),
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "14px"}, children=[
                html.Div(id="log-status", className="log-status idle", children="idle"),
                html.Button("■ CANCEL RUN", id="cancel-btn", className="btn", disabled=True),
            ]),
            html.Pre(id="log-pre", className="log-pre", children="No run started yet."),
        ]),
        dcc.Interval(id="log-interval", interval=2000, n_intervals=0),
    ])


def _tab_results():
    return html.Div(className="content", children=[
        html.Div(className="card", children=[
            html.H2("Results"),
            _labeled("Results directory", html.Div([
                dcc.Input(id="results-dir", type="text", value="",
                          style={"width": "70%", "display": "inline-block"}),
                html.Button("Browse", id="browse-results-dir", n_clicks=0,
                            className="btn btn-small", style={"marginLeft": "8px"}),
                html.Button("Load", id="load-results", n_clicks=0,
                            className="btn btn-small", style={"marginLeft": "8px"}),
            ])),
            html.Div(id="results-summary"),
        ]),
        html.Div(id="results-body"),
    ])


def make_layout():
    return html.Div([
        dcc.Store(id="loaded-config"),
        dcc.Store(id="fb-target"), dcc.Store(id="fb-cwd"), dcc.Store(id="fb-entries"),
        _header(),
        dcc.Tabs(id="tabs", value="cfg", className="picasso-tabs", children=[
            dcc.Tab(label="CONFIGURE & RUN", value="cfg", className="tab", selected_className="tab--selected",
                    children=_tab_configure()),
            dcc.Tab(label="LIVE LOG", value="log", className="tab", selected_className="tab--selected",
                    children=_tab_log()),
            dcc.Tab(label="RESULTS", value="res", className="tab", selected_className="tab--selected",
                    children=_tab_results()),
        ]),
        html.Div(id="fb-modal-wrap"),
    ])


app = Dash(__name__, assets_folder=_ASSETS, title="MSI-PICASSO", suppress_callback_exceptions=True)
app.layout = make_layout()


# --------------------------------------------------------------------------
# Preset / upload → loaded-config store
# --------------------------------------------------------------------------
def _load(text: str, what: str):
    try:
        cfg, unknown = config_from_toml(text)
    except Exception as exc:
        return no_update, f"Could not read TOML: {exc}"
    msg = f"{what} loaded — fields populated."
    if unknown:
        msg += f" Ignored keys the CLI does not know: {', '.join(unknown)}."
    return cfg, msg


@app.callback(
    Output("loaded-config", "data", allow_duplicate=True),
    Output("run-msg", "children", allow_duplicate=True),
    Input("preset", "value"),
    prevent_initial_call=True,
)
def preset_to_store(path):
    if not path:
        return no_update, no_update
    return _load(Path(path).read_text(), Path(path).name)


@app.callback(
    Output("loaded-config", "data", allow_duplicate=True),
    Output("run-msg", "children", allow_duplicate=True),
    Input("upload-toml", "contents"),
    prevent_initial_call=True,
)
def upload_to_store(contents):
    if not contents:
        return no_update, no_update
    return _load(base64.b64decode(contents.split(",", 1)[1]).decode("utf-8"), "Config")


@app.callback(
    Output("in-maldi_d", "value"), Output("in-lcms_peptides", "value"),
    Output("in-output_dir", "value"), Output("in-decoy_method", "value"),
    Output({"role": "cfg", "key": ALL}, "value"),
    Input("loaded-config", "data"),
    State({"role": "cfg", "key": ALL}, "id"),
    prevent_initial_call=True,
)
def populate_fields(cfg, cfg_ids):
    cfg = cfg or {}
    return (*(cfg.get(k) for k in _PATH_KEYS + ("decoy_method",)),
            [_show(cfg.get(cid["key"])) for cid in cfg_ids])


# --------------------------------------------------------------------------
# File browser
# --------------------------------------------------------------------------
def _list_dir(path: str):
    entries, p = [], Path(path)
    try:
        for e in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if e.name.startswith("."):
                continue
            kind = "d" if (e.is_dir() and e.name.endswith(".d")) else ("dir" if e.is_dir() else "file")
            entries.append({"name": e.name, "path": str(e), "kind": kind})
    except PermissionError:
        pass
    return entries


def _fb_view(cwd, entries):
    rows = [html.Div([html.Span("⬆"), html.Span(".. (parent)")], className="fb-entry is-dir",
                     id={"role": "fb-entry", "idx": -1})]
    for i, e in enumerate(entries):
        icon = {"dir": "📁", "d": "🗂", "file": "📄"}[e["kind"]]
        cls = {"dir": "fb-entry is-dir", "d": "fb-entry is-d", "file": "fb-entry"}[e["kind"]]
        rows.append(html.Div([html.Span(icon), html.Span(e["name"])], className=cls,
                             id={"role": "fb-entry", "idx": i}))
    return html.Div(className="fb-modal-backdrop", children=[
        html.Div(className="fb-modal", children=[
            html.Div("Select a file or folder", style={"fontFamily": PIXEL_FONT}),
            html.Div(cwd, className="fb-path"),
            html.Div(rows, className="fb-list"),
            html.Div(className="fb-actions", children=[
                html.Button("Choose this folder", id="fb-choose-dir", n_clicks=0, className="btn btn-small"),
                html.Button("Cancel", id="fb-cancel", n_clicks=0, className="btn btn-small"),
            ]),
        ]),
    ])


@app.callback(
    Output("fb-modal-wrap", "children", allow_duplicate=True),
    Output("fb-target", "data"), Output("fb-cwd", "data"), Output("fb-entries", "data"),
    Input("browse-maldi_d", "n_clicks"), Input("browse-lcms_peptides", "n_clicks"),
    Input("browse-output_dir", "n_clicks"), Input("browse-results-dir", "n_clicks"),
    State("in-maldi_d", "value"), State("in-output_dir", "value"),
    prevent_initial_call=True,
)
def open_browser(n1, n2, n3, n4, maldi_v, out_v):
    target = {"browse-maldi_d": "maldi_d", "browse-lcms_peptides": "lcms_peptides",
              "browse-output_dir": "output_dir", "browse-results-dir": "results-dir"}[ctx.triggered_id]
    start = maldi_v or out_v or str(Path.cwd())
    start = str(Path(start).parent if Path(start).is_file() else Path(start))
    if not Path(start).is_dir():
        start = str(Path.cwd())
    entries = _list_dir(start)
    return _fb_view(start, entries), target, start, entries


@app.callback(
    Output("fb-modal-wrap", "children", allow_duplicate=True),
    Output("fb-cwd", "data", allow_duplicate=True), Output("fb-entries", "data", allow_duplicate=True),
    Output("in-maldi_d", "value", allow_duplicate=True),
    Output("in-lcms_peptides", "value", allow_duplicate=True),
    Output("in-output_dir", "value", allow_duplicate=True),
    Output("results-dir", "value", allow_duplicate=True),
    Input({"role": "fb-entry", "idx": ALL}, "n_clicks"),
    Input("fb-choose-dir", "n_clicks"), Input("fb-cancel", "n_clicks"),
    State("fb-cwd", "data"), State("fb-entries", "data"), State("fb-target", "data"),
    prevent_initial_call=True,
)
def browser_action(entry_clicks, choose, cancel, cwd, entries, target):
    trig = ctx.triggered_id
    keep = (no_update,) * 4

    def selection(path):
        vals = {"maldi_d": 0, "lcms_peptides": 1, "output_dir": 2, "results-dir": 3}
        out = [no_update, no_update, no_update, no_update]
        out[vals[target]] = path
        return tuple(out)

    if trig == "fb-cancel":
        return (None, no_update, no_update, *keep)
    if trig == "fb-choose-dir":
        return (None, no_update, no_update, *selection(cwd))
    if not any(entry_clicks or []):
        return (no_update,) * 7
    idx = trig["idx"]
    if idx == -1:
        new = str(Path(cwd).parent)
    else:
        e = entries[idx]
        if e["kind"] in ("d", "file"):
            return (None, no_update, no_update, *selection(e["path"]))
        new = e["path"]
    ent = _list_dir(new)
    return _fb_view(new, ent), new, ent, *keep


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def launch(section: dict) -> subprocess.Popen:
    """Write <output_dir>/gui_config.toml and start `picasso -c` on it; console -> gui_console.log."""
    import tomli_w
    out = Path(section["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "gui_config.toml"
    cfg_path.write_text(tomli_w.dumps({"MSI-PICASSO": section}))
    console = open(out / _CONSOLE_LOG, "w")
    # Launch via the installed `picasso` console script, NOT `python -m msi_picasso.cli`:
    # `-m` prepends the cwd to sys.path, so a cwd holding a psm_utils/ checkout would shadow
    # the installed package. Console scripts don't add cwd; cwd=out is a neutral dir as a further guard.
    picasso_exe = Path(sys.executable).with_name("picasso")
    cmd = ([str(picasso_exe), "-c", str(cfg_path)] if picasso_exe.exists()
           else [sys.executable, "-m", "msi_picasso.cli", "-c", str(cfg_path)])
    env = dict(os.environ, PYTHONUNBUFFERED="1")  # the CLI logs with print; unbuffered = a live log
    return subprocess.Popen(cmd, stdout=console, stderr=subprocess.STDOUT, cwd=str(out), env=env)


@app.callback(
    Output("run-msg", "children"), Output("run-btn", "disabled", allow_duplicate=True),
    Output("results-dir", "value", allow_duplicate=True),
    Input("run-btn", "n_clicks"),
    State("in-maldi_d", "value"), State("in-lcms_peptides", "value"),
    State("in-output_dir", "value"), State("in-decoy_method", "value"),
    State({"role": "cfg", "key": ALL}, "value"), State({"role": "cfg", "key": ALL}, "id"),
    prevent_initial_call=True,
)
def run_picasso(n, maldi_d, lcms, outdir, decoy, cfg_vals, cfg_ids):
    if _JOB["proc"] is not None and _JOB["proc"].poll() is None:
        return "A run is already in progress.", True, no_update
    values = dict(zip(_PATH_KEYS + ("decoy_method",), (maldi_d, lcms, outdir, decoy)))
    values.update({cid["key"]: v for cid, v in zip(cfg_ids, cfg_vals)})
    section = section_from_form(values)
    missing = [k for k in _PATH_KEYS if k not in section]
    if missing:
        return f"Set {', '.join(missing)} first.", False, no_update
    try:
        proc = launch(section)
    except Exception as exc:
        return f"Could not start the run: {exc}", False, no_update
    _JOB.update(proc=proc, outdir=section["output_dir"], cancelled=False)
    return f"Launched (pid {proc.pid}). Watch the Live Log tab.", True, section["output_dir"]


@app.callback(
    Output("log-status", "children", allow_duplicate=True),
    Output("log-status", "className", allow_duplicate=True),
    Output("run-btn", "disabled", allow_duplicate=True),
    Output("cancel-btn", "disabled", allow_duplicate=True),
    Input("cancel-btn", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_run(_n):
    proc = _JOB["proc"]
    if proc is None or proc.poll() is not None:
        return no_update, no_update, no_update, True
    _JOB["cancelled"] = True
    proc.terminate()  # SIGTERM; poll_log reports "cancelled" once it exits.
    return "cancelling…", "log-status failed", False, True


@app.callback(
    Output("log-status", "children"), Output("log-status", "className"),
    Output("log-pre", "children"), Output("run-btn", "disabled", allow_duplicate=True),
    Output("cancel-btn", "disabled"),
    Input("log-interval", "n_intervals"),
    prevent_initial_call=True,
)
def poll_log(_n):
    proc, outdir = _JOB["proc"], _JOB["outdir"]
    if proc is None or outdir is None:
        return "idle", "log-status idle", "No run started yet.", False, True
    log_path = Path(outdir) / _CONSOLE_LOG
    text = ""
    if log_path.exists():
        # split on \n only and keep the text after the last \r: a tqdm bar shows as its latest state, one line
        lines = [ln.rsplit("\r", 1)[-1] for ln in log_path.read_text(errors="replace").split("\n")]
        text = "\n".join(lines[-400:]) or "(waiting for output…)"
    rc = proc.poll()
    if rc is None:  # running → cancel enabled, run disabled
        return "running…", "log-status running", text, True, False
    if _JOB.get("cancelled"):
        return "cancelled ✗", "log-status failed", text, False, True
    if rc == 0:
        return "finished ✓", "log-status finished", text, False, True
    return f"failed (exit {rc})", "log-status failed", text, False, True


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
def _stat(value, label):
    return html.Div(className="stat", children=[
        html.Div(f"{value}", className="stat-value"), html.Div(label, className="stat-label")])


def _table(df: pd.DataFrame | None):
    if df is None or df.empty:
        return html.P("(empty)", className="hint")
    show = df.copy()
    for c in show.select_dtypes("float").columns:
        show[c] = show[c].round(4)
    return dash_table.DataTable(
        data=show.to_dict("records"), columns=[{"name": c, "id": c} for c in show.columns],
        page_size=12, sort_action="native", filter_action="native", style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "VT323, monospace", "fontSize": "16px",
                    "border": f"2px solid {NAVY}", "padding": "4px 8px", "backgroundColor": "#ffffff"},
        style_header={"fontFamily": PIXEL_FONT, "fontSize": "15px", "backgroundColor": NAVY,
                      "color": "#ffffff", "border": f"2px solid {NAVY}"})


def _gallery(fig_dir: Path):
    """figures/ written by the CLI (figures = true): check plots, protein and per-candidate ion images."""
    pngs = sorted(fig_dir.rglob("*.png")) if fig_dir.is_dir() else []
    if not pngs:
        return html.Div()
    # ponytail: base64-inlined (~10 MB for a full figures/ dir); serve via a Flask route if that gets slow.
    figs = [html.Figure([html.Img(src="data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()),
                         html.Figcaption(str(p.relative_to(fig_dir)))]) for p in pngs]
    return html.Div(className="card", children=[html.H2("Figures"), html.Div(figs, className="gallery")])


def _counts(d: dict | None) -> str:
    return " / ".join(str(v) for v in d.values()) if d else "–"


@app.callback(
    Output("results-summary", "children"), Output("results-body", "children"),
    Input("load-results", "n_clicks"), State("results-dir", "value"),
    prevent_initial_call=True,
)
def load_results(_n, outdir):
    if not outdir or not Path(outdir).is_dir():
        return html.P("Set a valid results directory.", className="hint"), None
    out = Path(outdir)
    peptides = _read_tsv(out / "peptides.tsv")
    proteins = _read_tsv(out / "proteins.tsv")
    if peptides is None or not (out / "run.json").exists():
        return html.P("No peptides.tsv / run.json here (run not finished?).", className="hint"), None
    info = json.loads((out / "run.json").read_text())
    off = info.get("mz_offset", {}).get("used")
    summary = html.Div(className="stat-row", children=[
        _stat(_counts(info.get("ids")), "peptides at 1 / 5 / 10% FDR"),
        _stat(_counts(info.get("proteins_picked")), "proteins (picked) at 1 / 5 / 10%"),
        *([_stat(_counts(info["gt"]), "GT peptides at 1 / 5 / 10%")] if info.get("gt") else []),
        _stat(f"{info.get('n_targets', '–')} / {info.get('n_decoys', '–')}", "target / decoy peptides"),
        _stat("–" if off is None else f"{off:+.2f} ppm", "m/z offset used"),
    ])

    plots = html.Div(className="card", children=[html.H2("Plots"), html.Div(className="row", children=[
        html.Div(className="col", children=dcc.Graph(figure=build_ids_vs_fdr(peptides, proteins))),
        html.Div(className="col", children=dcc.Graph(figure=build_score_dist(peptides))),
    ])])
    tp = peptides[~peptides["is_decoy"]].sort_values("q")
    tables = html.Div(className="card", children=[html.H2("Tables"),
        html.H3("Target peptides (by q)"), _table(tp),
        html.H3("Proteins (by picked q)"),
        _table(None if proteins is None else proteins[~proteins["is_decoy"]].sort_values("q_picked"))])
    return summary, html.Div([plots, tables, _gallery(out / "figures")])


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(prog="picasso-gui", description="MSI-PICASSO graphical interface")
    parser.add_argument("--desktop", action="store_true",
                        help="Open in a native desktop window (needs the 'desktop' extra).")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if args.desktop:
        import threading
        import webview  # ponytail: ~10-line thread + pywebview wrapper, no Electron.
        threading.Thread(target=lambda: app.run(host=args.host, port=args.port), daemon=True).start()
        webview.create_window("MSI-PICASSO", f"http://{args.host}:{args.port}")
        webview.start()
    else:
        app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
