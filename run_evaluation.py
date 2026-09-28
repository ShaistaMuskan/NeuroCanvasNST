import argparse
import time
from pathlib import Path

from PIL import Image

import engines as E
import library as L
import metrics as M


def log(frac, desc=""):
    print(f"  {frac:5.0%}  {desc}")


def md_table(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["Gatys (L-BFGS)", "AdaIN (feed-forward)", "Magenta (TF-Hub)"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[256, 512])
    ap.add_argument("--repeats", type=int, default=5, help="timing repeats for feed-forward methods")
    ap.add_argument("--gatys-iters", type=int, default=100)
    ap.add_argument("--loss-size", type=int, default=256)
    ap.add_argument("--n-content", type=int, default=3)
    ap.add_argument("--n-style", type=int, default=3)
    ap.add_argument("--conv-iters", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1e6)
    ap.add_argument("--out", default="evaluation_results")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    C, S = L.content_library(), L.style_library()
    if not C or not S:
        raise SystemExit("Put images in assets/content and assets/styles first.")
    cn, sn = list(C)[: a.n_content], list(S)[: a.n_style]
    c0, s0 = Image.open(C[cn[0]]), Image.open(S[sn[0]]["path"])
    report = [f"# NeuroCanvas quantitative evaluation\n\nDevice: `{E.DEVICE}`. Generated {time.strftime('%Y-%m-%d %H:%M')}.\n"]

    print("[1/3] speed")
    disp, raw, fig = M.speed_benchmark(c0, s0, a.methods, a.sizes, a.repeats, a.gatys_iters, a.gatys_iters, log)
    disp.to_csv(out / "speed.csv", index=False)
    fig.savefig(out / "speed.png", dpi=160)
    report += ["## 1. Stylisation speed (seconds per image)\n", md_table(disp), "\n![speed](speed.png)\n"]

    print("[2/3] loss comparison")
    pairs = [(c, Image.open(C[c]), s, Image.open(S[s]["path"])) for c in cn for s in sn]
    per, summ, fig = M.loss_comparison(pairs, a.methods, a.loss_size, a.alpha, a.beta, a.gatys_iters, a.gatys_iters, log)
    per.to_csv(out / "loss_per_image.csv", index=False)
    summ.to_csv(out / "loss_summary.csv", index=False)
    fig.savefig(out / "loss_comparison.png", dpi=160)
    report += [f"## 2. Loss comparison (α={a.alpha:g}, β={a.beta:g}; {len(pairs)} content×style pairs)\n",
               md_table(summ), "\n![loss](loss_comparison.png)\n"]

    print("[3/3] convergence")
    conv, fig, _ = M.convergence(c0, s0, list(M.CONV_PRESETS), a.conv_iters, a.loss_size, a.alpha, a.beta, progress=log)
    conv.to_csv(out / "convergence.csv", index=False)
    fig.savefig(out / "convergence.png", dpi=160)
    report += ["## 3. Loss variation during optimisation\n", md_table(conv), "\n![convergence](convergence.png)\n"]

    report += ["## 4. Method characteristics\n", md_table(M.capability_df()),
               "\nTraining time is not measured: all models are pre-trained.\n"]
    (out / "report.md").write_text("\n".join(report))
    print(f"Done -> {out}/report.md")


if __name__ == "__main__":
    main()
