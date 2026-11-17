"""Reference dumps of the closed bench's adopted method (real world, one draw), for check_bench_parity.py.

Runs the bench's own code (bench/trial.py on the frozen msi_picasso, bench/msi_picasso symlink), so run it with the
bench interpreter: python MSI-PICASSO/scripts/make_bench_ref.py [rep]
Writes /home/robbe/MALDI_MSI_score/port_ref/<trial id>_rep<rep>_{rows,pep,prot}.parquet: every scored row before per-ion
removal (s0 = M1, gone, s = final score), the peptide table and the protein table, and checks the IDs against
bench/results/worlds/<id>.tsv.
"""
import sys
import time
import tomllib
from pathlib import Path

BENCH = Path("/home/robbe/MALDI_MSI_score/bench")
sys.path.insert(0, str(BENCH))  # bench first: its msi_picasso symlink is the frozen package
import trial  # noqa: E402  (sets BLAS threads before numpy)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("/home/robbe/MALDI_MSI_score/port_ref")
IDS = {"kidney": ("429c0e4a59", "df0c1354fe"), "amyloidosis": ("7417ddd3e6", "771c8db220"),
       "her2": ("230058d2e4", "78ba6a068c")}


def world(spec, rep):
    df = trial.build_world(spec, "real", rep)
    s0 = trial.score(df, spec, seed=rep)
    dec, mz = df.is_decoy.to_numpy(bool), df.feature_mz.to_numpy(float)
    pk = df.groupby("peptide").is_decoy.first()
    k = pk.sum() / (~pk).sum()  # k_eff before removal (one_world)
    gone = trial.ion_removed(s0, dec, mz, spec["ion_competition_ppm"])
    s = s0 - trial.ion_soft_penalty(s0, dec, mz, spec["ion_competition_ppm"], spec["ion_soft_tau"])
    kept = df[~gone].reset_index(drop=True)
    pt = trial.peptide_table(kept, s[~gone], k, False, None)
    return df.assign(s0=s0, gone=gone, s=s), pt, trial.protein_table(pt)


if __name__ == "__main__":
    rep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    OUT.mkdir(exist_ok=True)
    trials = tomllib.loads((BENCH / "specs/selfwit_seeds20.toml").read_text())["trial"]
    for t, tid in zip(trials, [i for ds in ("kidney", "amyloidosis", "her2") for i in IDS[ds]]):
        spec = {**trial.DEFAULT, **t, "id": tid}
        if trial._S.get("peaklist") != spec["peaklist"] or trial._S.get("ds") != spec["ds"]:
            trial.load_store(spec["ds"], spec["peaklist"])
        t0 = time.time()
        rows, pt, prt = world(spec, rep)
        ref = pd.read_csv(BENCH / "results/worlds" / f"{tid}.tsv", sep="\t")
        ref = ref[(ref.world == "real") & (ref.rep == rep)].iloc[0]
        got = [trial.ids(pt, a) for a in (.01, .05, .1)]
        assert got == [ref.ids1, ref.ids5, ref.ids10], (tid, got, ref[["ids1", "ids5", "ids10"]].tolist())
        rows.drop(columns=[c for c in rows.columns if rows[c].dtype == object and c not in
                           ("peptide", "protein", "parent_peptide", "source")]).to_parquet(OUT / f"{tid}_rep{rep}_rows.parquet")
        pt.to_parquet(OUT / f"{tid}_rep{rep}_pep.parquet")
        prt.to_parquet(OUT / f"{tid}_rep{rep}_prot.parquet")
        print(f"{tid} {spec['ds']} rep {rep}: IDs {got} = bench; {len(rows)} rows; {time.time() - t0:.0f} s", flush=True)
