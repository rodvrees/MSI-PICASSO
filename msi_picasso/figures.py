"""Figures for checking a run (cli --figures):
  ids_vs_fdr.png        target peptides and proteins (picked) reported at each q threshold
  scores.png            target / decoy peptide score distributions and the PP plot (decoy vs target CDF)
  terms.png             target / decoy distribution of each score term (all rows)
  mz.png                target / decoy query m/z distribution
  proteins/*.png        on-tissue M images of the target peptides of the best reported proteins (+ GT proteins)
  candidates/*.png      per candidate: M-1 / M / M+1 images, envelope vs averagine, the +-25 ppm top-decile-pixel
                        m/z profile, and the weighted z-scored term contributions (reported targets, top decoys, GT)
"""
from pathlib import Path

import numpy as np

from msi_picasso.extract import CEN_PPM
from msi_picasso.score import _z, averagine_dist3, ids, term_weights


def _td_hist(ax, x, dec, bins):
    """Target and decoy densities of x on one axis."""
    ax.hist(x[~dec], bins, density=True, alpha=.5, label="targets")
    ax.hist(x[dec], bins, density=True, alpha=.5, label="decoys")


def _image(ax, v, layout, title):
    idx, shape = layout
    im = np.full(shape[0] * shape[1], np.nan)
    im[idx] = v
    ax.imshow(im.reshape(shape), cmap="viridis")
    ax.set_title(title, fontsize=7)
    ax.axis("off")


N_PROT, N_CAND = 12, 20  # proteins and reported targets drawn (plus half as many top decoys, and every GT peptide)


def save_figures(out, rows, pt, prt, Q, img, prof, layout, gt=(), offset=0.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    for sub in ("proteins", "candidates"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    dec = pt.is_decoy.to_numpy(bool)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    a = np.linspace(0, 0.2, 201)
    ax.plot(a, [ids(pt, x) for x in a], label="peptides")
    ax.plot(a, [ids(prt, x, "q_picked") for x in a], label="proteins (picked)")
    ax.set(xlabel="q threshold", ylabel="targets reported")
    ax.legend()
    fig.savefig(out / "ids_vs_fdr.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
    s = pt.s.to_numpy(float)
    bins = np.linspace(s.min(), s.max(), 80)
    _td_hist(a1, s, dec, bins)
    a1.set(xlabel="peptide score (best row)", yscale="log")
    a1.legend()
    d = np.sort(s[dec])
    t = np.sort(s[~dec])
    a2.plot(np.searchsorted(d, t, "right") / len(d), np.arange(1, len(t) + 1) / len(t))
    a2.plot([0, 1], [0, 1], "k:", lw=.8)
    a2.set(xlabel="decoy CDF", ylabel="target CDF", title="PP plot")
    fig.savefig(out / "scores.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    rdec = rows.is_decoy.to_numpy(bool)
    terms, w = term_weights(rows)
    fig, axs = plt.subplots(1, len(terms), figsize=(3.2 * len(terms), 3))
    for ax, term in zip(axs, terms):
        x = rows[term].to_numpy(float)
        _td_hist(ax, x, rdec, np.linspace(np.nanmin(x), np.nanmax(x), 60))
        ax.set(title=term, yscale="log")
    axs[0].legend(fontsize=7)
    fig.savefig(out / "terms.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    mz = rows.feature_mz.to_numpy()
    _td_hist(ax, mz, rdec, np.linspace(mz.min(), mz.max(), 80))
    ax.set(xlabel="query m/z")
    ax.legend()
    fig.savefig(out / "mz.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    tg = pt[~pt.is_decoy]
    rep = prt[~prt.is_decoy & (prt.q_picked <= 0.1)].sort_values("q_picked").protein.head(N_PROT).tolist()
    for prot in dict.fromkeys(rep + tg[tg.is_gt].protein.tolist()):
        p = tg[tg.protein == prot].sort_values("s", ascending=False).head(8)
        fig, axs = plt.subplots(1, len(p), figsize=(2.2 * len(p), 2.4), squeeze=False)
        for ax, (pep, r) in zip(axs[0], p.iterrows()):
            _image(ax, img[r.qi, 1], layout, f"{pep[:18]}\nq {r.q:.3f}{' GT' if r.is_gt else ''}")
        fig.suptitle(prot, fontsize=8)
        fig.savefig(out / "proteins" / f"{prot.replace('|', '_')}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    Z = _z(rows[terms].to_numpy(float)) * w
    bi = rows.groupby("peptide").s.idxmax()  # each peptide's best row
    picks = [("T", p) for p in tg[tg.q <= 0.05].sort_values("s", ascending=False).index[:N_CAND]]
    picks += [("D", p) for p in pt[dec].sort_values("s", ascending=False).index[:N_CAND // 2]]
    picks += [("GT", p) for p in tg.index[tg.is_gt]]
    x = np.arange(-CEN_PPM, CEN_PPM + 1)
    for i, (kind, pep) in enumerate(picks):
        r = rows.loc[bi[pep]]
        qi = int(r.qi)
        fig, axs = plt.subplots(1, 6, figsize=(17, 2.8))
        for ax, c, name in zip(axs[:3], (0, 1, 2), ("M-1", "M", "M+1")):
            _image(ax, img[qi, c], layout, name)
        env = Q.loc[qi, ["env0", "env1", "env2"]].to_numpy(float)
        ref = averagine_dist3([Q.mz[qi]])[0]
        axs[3].bar(np.arange(3) - .2, env / np.nansum(env), .4, label="observed")
        axs[3].bar(np.arange(3) + .2, ref, .4, label="averagine")
        axs[3].set(xticks=range(3), xticklabels=["M", "M+1", "M+2"], title=f"KL {r.kl:.2g}")
        axs[3].legend(fontsize=6)
        axs[4].plot(x[prof[qi] > 0], prof[qi][prof[qi] > 0], ".-")  # sampled bins only
        if "dgrad" in Q:
            axs[4].axvline(Q.dgrad[qi] + offset, color="r", lw=.8)
            axs[4].set(xlabel="ppm from query", title=f"dgrad {Q.dgrad[qi]:+.2f}, crat {Q.crat[qi]:.2f}")
        else:
            axs[4].axvline(Q.cen[qi], color="r", lw=.8)
            axs[4].set(xlabel="ppm from query", title=f"centroid {Q.cen[qi]:+.2f} ppm")
        axs[5].barh(terms, Z[rows.index.get_loc(bi[pep])])
        axs[5].set(title=f"s {r.s:.2f}, q {pt.q[pep]:.3f}")
        fig.suptitle(f"{kind} {pep} ({r.protein}) m/z {r.feature_mz:.4f}", fontsize=8)
        fig.savefig(out / "candidates" / f"{kind}_{i:03d}_{pep[:40]}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
