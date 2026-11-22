"""Pixelated Dash GUI for MSI-PICASSO (palette derived from the logo).

A thin wrapper: it writes a TOML config (keys = ``cli.DEFAULTS``), launches the
``picasso`` CLI as a subprocess, tails its log, and renders the on-disk results
(gui_results.py: run.json, the tables, rows / evidence and viewer/ as interactive views).
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

from dash import Dash, dcc, html, Input, Output, State, ctx, ALL, no_update

from msi_picasso import gui_results
from msi_picasso.cli import DEFAULTS

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
_ADVANCED_KEYS = ("decoy_seeds", "query_seeds", "extraction_ppm", "mz_range", "tissue_mask", "threads",
                  "viewer_all_targets", "viewer_all_decoys")
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


def read_log(path: Path, n: int = 400) -> str:
    """The console log as shown: read as bytes (text mode would turn \r into newlines), split on \n only, and keep
    the text after a line's last \r, so a progress bar redrawn with \r shows as its latest state on one line."""
    raw = path.read_bytes().decode(errors="replace")
    return "\n".join(ln.rsplit("\r", 1)[-1] for ln in raw.split("\n")[-n:]).rstrip("\n")


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


def _topbar():
    tab = lambda label, value: dcc.Tab(label=label, value=value, className="tab", selected_className="tab--selected")
    return html.Div(className="topbar", children=[
        html.Img(src="/assets/MSI-PICASSO-logo-t.png", alt="MSI-PICASSO"),
        dcc.Tabs(id="tabs", value="cfg", className="picasso-tabs", parent_className="picasso-tabs-parent",
                 children=[tab("Configure", "cfg"), tab("Live log", "log"), tab("Results", "res")]),
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
            html.Div(className="log-head", children=[
                html.Div(id="log-status", className="log-status idle", children="idle"),
                html.Button("■ Cancel run", id="cancel-btn", className="btn", disabled=True),
            ]),
            html.Pre(id="log-pre", className="log-pre"),  # filled by assets/log.js (typewriter)
        ]),
        dcc.Store(id="log-text", data="No run started yet."),
        dcc.Interval(id="log-interval", interval=1000, n_intervals=0),
    ])


def make_layout():
    return html.Div([
        dcc.Store(id="loaded-config"),
        dcc.Store(id="fb-target"), dcc.Store(id="fb-cwd"), dcc.Store(id="fb-entries"),
        _topbar(),
        html.Div(id="pane-cfg", children=_tab_configure()),
        html.Div(id="pane-log", children=_tab_log(), hidden=True),
        html.Div(id="pane-res", children=gui_results.layout(), hidden=True),
        html.Div(id="fb-modal-wrap"),
        html.Div(className="footer", children="Fonts: Silkscreen and Departure Mono (SIL OFL); "
                 "Px437 IBM VGA 8x16 by VileR, int10h.org (CC BY-SA 4.0)."),
    ])


app = Dash(__name__, assets_folder=_ASSETS, title="MSI-PICASSO", suppress_callback_exceptions=True)
app.layout = make_layout()
gui_results.register(app)


@app.callback(Output("pane-cfg", "hidden"), Output("pane-log", "hidden"), Output("pane-res", "hidden"),
              Input("tabs", "value"))
def show_pane(tab):  # panes stay mounted, so form values and loaded results survive a tab switch
    return tab != "cfg", tab != "log", tab != "res"


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
            html.Div("Select a file or folder", className="fb-title"),
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
    Output("results-dir", "value", allow_duplicate=True), Output("tabs", "value"),
    Input("run-btn", "n_clicks"),
    State("in-maldi_d", "value"), State("in-lcms_peptides", "value"),
    State("in-output_dir", "value"), State("in-decoy_method", "value"),
    State({"role": "cfg", "key": ALL}, "value"), State({"role": "cfg", "key": ALL}, "id"),
    prevent_initial_call=True,
)
def run_picasso(n, maldi_d, lcms, outdir, decoy, cfg_vals, cfg_ids):
    if _JOB["proc"] is not None and _JOB["proc"].poll() is None:
        return "A run is already in progress.", True, no_update, "log"
    values = dict(zip(_PATH_KEYS + ("decoy_method",), (maldi_d, lcms, outdir, decoy)))
    values.update({cid["key"]: v for cid, v in zip(cfg_ids, cfg_vals)})
    section = section_from_form(values)
    missing = [k for k in _PATH_KEYS if k not in section]
    if missing:
        return f"Set {', '.join(missing)} first.", False, no_update, no_update
    try:
        proc = launch(section)
    except Exception as exc:
        return f"Could not start the run: {exc}", False, no_update, no_update
    _JOB.update(proc=proc, outdir=section["output_dir"], cancelled=False, shown=False)
    return f"Launched (pid {proc.pid}).", True, section["output_dir"], "log"


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
    Output("log-text", "data"), Output("run-btn", "disabled", allow_duplicate=True),
    Output("cancel-btn", "disabled"),
    Output("tabs", "value", allow_duplicate=True), Output("results-dir", "value", allow_duplicate=True),
    Output("res-state", "data", allow_duplicate=True),
    Input("log-interval", "n_intervals"),
    prevent_initial_call=True,
)
def poll_log(_n):
    proc, outdir = _JOB["proc"], _JOB["outdir"]
    keep = (no_update,) * 3
    if proc is None or outdir is None:
        return "idle", "log-status idle", no_update, False, True, *keep
    log_path = Path(outdir) / _CONSOLE_LOG
    text = (read_log(log_path) if log_path.exists() else "") or "(waiting for output…)"
    rc = proc.poll()
    if rc is None:  # running → cancel enabled, run disabled
        return "running…", "log-status running", text, True, False, *keep
    ok = rc == 0 and not _JOB.get("cancelled")
    status = "finished ✓" if ok else "cancelled ✗" if _JOB.get("cancelled") else f"failed (exit {rc})"
    jump = keep
    if ok and not _JOB.get("shown") and (Path(outdir) / "run.json").exists():  # once per run: open its results
        _JOB["shown"] = True
        jump = ("res", outdir, gui_results.run_state(outdir))
    return status, f"log-status {'finished' if ok else 'failed'}", text, False, True, *jump


# Hand each new log text to the typewriter in assets/log.js.
app.clientside_callback(
    "function(text) { if (window.picassoType) { window.picassoType(text); } return window.dash_clientside.no_update; }",
    Output("log-pre", "title"), Input("log-text", "data"))


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
