from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import structural_similarity

import engines as E
from theme import PALETTE, SERIES, apply_mpl

apply_mpl()
from matplotlib.figure import Figure  # noqa: E402  (no pyplot: no global figure registry, thread-safe)


def _subplots(nrows=1, ncols=1, figsize=None):
    fig = Figure(figsize=figsize)
    return fig, fig.subplots(nrows, ncols)

NEVER = 10**9  


# Method registry
def _n_params(*mods) -> float:
    return sum(p.numel() for m in mods for p in m.parameters()) / 1e6


def method_registry(gatys_iters=100, adam_iters=100) -> dict[str, dict]:
    """name -> {kind, styles, fn(content, style, size) -> PIL, slow, params()}"""
    return {
        "Gatys (L-BFGS)": dict(
            kind="Optimisation (IOB-NST)", styles="1 per run", slow=True,
            fn=lambda c, s, size: E.last_of(
                E.gatys_stream(c, s, iterations=gatys_iters, size=size, preview_every=NEVER))[0],
            params=lambda: _n_params(E.get_loss_net()),
        ),
        "AdaIN (feed-forward)": dict(
            kind="Feed-forward (ASPM)", styles="any", slow=False,
            fn=lambda c, s, size: E.adain_stylize(c, s, size=size),
            params=lambda: _n_params(*E.get_adain()),
        ),
        "AdaIN + Adam": dict(
            kind="Optimisation (encoder space)", styles="1 per run", slow=True,
            fn=lambda c, s, size: E.last_of(
                E.adain_adam_stream(c, s, iterations=adam_iters, size=size, preview_every=NEVER))[0],
            params=lambda: _n_params(E.get_adain()[0]),
        ),
        "Magenta (TF-Hub)": dict(
            kind="Feed-forward (ASPM)", styles="any", slow=False,
            fn=lambda c, s, size: E.magenta_stylize(c, s, size=size),
            params=lambda: float("nan"),
        ),
    }


def _timed(fn, *args):
    E.sync()
    t = time.perf_counter()
    out = fn(*args)
    E.sync()
    return out, time.perf_counter() - t


def _is_oom(e: Exception) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


# 1) Stylisation speed
def speed_benchmark(content, style, methods, resolutions=(256, 512), repeats=3,
                    gatys_iters=100, adam_iters=100, progress=None):
    """Returns (display_df, raw_df, fig). Mean of `repeats` runs after one warm-up."""
    reg = method_registry(gatys_iters, adam_iters)
    rows_disp, rows_raw = [], []
    total = len(methods) * len(resolutions)
    k = 0
    for name in methods:
        m = reg[name]
        try:
            m["fn"](content, style, 64)  # warm-up: CUDA context, cudnn autotune, lazy model load
        except Exception as e:  # noqa: BLE001
            rows_disp.append({"Method": name, "Type": m["kind"], "Styles/model": m["styles"], "Error": str(e)[:80]})
            continue
        d = {"Method": name if not m["slow"] else f"{name} ({gatys_iters if 'Gatys' in name else adam_iters} it)",
             "Type": m["kind"], "Styles/model": m["styles"]}
        raw = {"method": name}
        peak = 0.0
        for res in resolutions:
            k += 1
            if progress:
                progress(k / max(total, 1), desc=f"{name} @ {res}px")
            times = []
            try:
                if E.DEVICE.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                for _ in range(1 if m["slow"] else repeats):
                    times.append(_timed(m["fn"], content, style, res)[1])
                if E.DEVICE.type == "cuda":
                    peak = max(peak, torch.cuda.max_memory_allocated() / 2**20)
                mu, sd = float(np.mean(times)), float(np.std(times))
                d[f"{res}x{res} (s)"] = f"{mu:.3f}" + (f" ± {sd:.3f}" if len(times) > 1 else "")
                raw[res] = mu
            except Exception as e:  # noqa: BLE001
                if not _is_oom(e):
                    raise
                d[f"{res}x{res} (s)"] = "OOM"
                raw[res] = math.nan
                if E.DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
        try:
            d["Params (M)"] = f"{m['params']():.1f}"
        except Exception:  # noqa: BLE001
            d["Params (M)"] = "-"
        if E.DEVICE.type == "cuda":
            d["Peak VRAM (MB)"] = f"{peak:.0f}"
        rows_disp.append(d)
        rows_raw.append(raw)
    raw_df = pd.DataFrame(rows_raw)
    return pd.DataFrame(rows_disp).fillna("-"), raw_df, plot_speed(raw_df, resolutions)


def plot_speed(raw_df, resolutions):
    fig, ax = _subplots(figsize=(7.2, 3.8))
    if raw_df.empty:
        return fig
    w = 0.8 / len(raw_df)
    x = np.arange(len(resolutions))
    for i, (_, r) in enumerate(raw_df.iterrows()):
        vals = [r.get(res, np.nan) for res in resolutions]
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w * 0.92, label=r["method"], color=SERIES[i % len(SERIES)])
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2g}", ha="center", va="bottom", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{r}x{r}" for r in resolutions])
    ax.set_ylabel("seconds / image (log)")
    ax.set_title("Stylisation speed")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


# 2) Loss comparison under ONE fixed objective (+ SSIM)
def ssim_gray(a: Image.Image, b: Image.Image) -> float:
    b = b.resize(a.size, Image.LANCZOS)
    return float(structural_similarity(np.asarray(a.convert("L")), np.asarray(b.convert("L")), data_range=255))


@torch.no_grad()
def score_output(content: Image.Image, style: Image.Image, output: Image.Image,
                 content_weight=1.0, style_weight=1e6) -> dict:
    """Score any stylised image with the Gatys objective (VGG-19 judge)."""
    ref_size = output.size
    c = E.to_tensor(content, 0, size_like=ref_size)
    s = E.to_tensor(style, max(ref_size))
    o = E.to_tensor(output, 0, size_like=ref_size)
    cl, sl, _ = E.GatysCriterion(c, s)(o)
    return {
        "content_loss": cl.item(), "style_loss": sl.item(),
        "total": content_weight * cl.item() + style_weight * sl.item(),
        "ssim": ssim_gray(content.resize(ref_size, Image.LANCZOS), output),
    }


def loss_comparison(pairs, methods, size=256, content_weight=1.0, style_weight=1e6,
                    gatys_iters=100, adam_iters=100, progress=None):
    """pairs: list of (content_name, content_pil, style_name, style_pil).
    Returns (per_sample_df, summary_df, fig)."""
    reg = method_registry(gatys_iters, adam_iters)
    rows = []
    total = len(pairs) * (len(methods) + 1)
    k = 0
    for cn, cimg, sn, simg in pairs:
        base = cimg.convert("RGB").resize(E.fit_size(*cimg.size, size)[:2], Image.LANCZOS)
        candidates = [("No stylisation (reference)", base, 0.0)]
        for name in methods:
            out, dt = _timed(reg[name]["fn"], cimg, simg, size)
            candidates.append((name, out, dt))
        for name, out, dt in candidates:
            k += 1
            if progress:
                progress(k / total, desc=f"{cn} + {sn}: {name}")
            sc = score_output(cimg, simg, out.convert("RGB"), content_weight, style_weight)
            rows.append({"method": name, "content": cn, "style": sn, "seconds": dt, **sc,
                         "weighted_content": content_weight * sc["content_loss"],
                         "weighted_style": style_weight * sc["style_loss"]})
    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df["method"]))
    g = df.groupby("method", sort=False)[["weighted_content", "weighted_style", "total", "ssim", "seconds"]]
    summ = g.mean().join(g.std(ddof=0).add_suffix(" (sd)")).reindex(order).reset_index()
    return df, summ, plot_losses(summ)


def plot_losses(summ):
    fig, axes = _subplots(1, 3, figsize=(11, 3.6))
    spec = [("total", "Total loss (α·C + β·S)"), ("weighted_style", "Style loss (β·S)"),
            ("weighted_content", "Content loss (α·C)")]
    for ax, (col, title) in zip(axes, spec):
        colors = [PALETTE["ash"] if m.startswith("No styl") else SERIES[i % len(SERIES)] for i, m in enumerate(summ["method"])]
        ax.bar(range(len(summ)), summ[col], yerr=summ[f"{col} (sd)"], color=colors, ecolor=PALETTE["ash"],
               capsize=3, width=0.7)
        ax.set_xticks(range(len(summ)), [m.replace(" (", "\n(") for m in summ["method"]], fontsize=7.5)
        ax.set_title(title)
    fig.suptitle("Same loss function, different algorithms (mean ± sd over content × style grid)",
                 fontsize=10, color=PALETTE["ash"])
    fig.tight_layout()
    return fig


# 3) Loss variation during optimisation (convergence)
CONV_PRESETS = {
    "Content init · lr 1.0": dict(init="content", lr=1.0),
    "Noise init · lr 1.0": dict(init="noise", lr=1.0),
    "Content init · lr 0.5": dict(init="content", lr=0.5),
}


def convergence(content, style, runs, iterations=200, size=256, content_weight=1.0,
                style_weight=1e6, tv_weight=1e-5, progress=None):
    """Runs Gatys L-BFGS under each preset with the SAME objective; returns (summary_df, fig, logs)."""
    logs = {}
    for i, label in enumerate(runs):
        if progress:
            progress(i / max(len(runs), 1), desc=f"Optimising: {label}")
        log = E.OptLog()
        E.last_of(E.gatys_stream(content, style, content_weight, style_weight, tv_weight,
                                 iterations=iterations, size=size, preview_every=NEVER, log=log,
                                 **CONV_PRESETS[label]))
        logs[label] = log
    rows = []
    for label, lg in logs.items():
        a = np.asarray(lg.total)
        target = a[0] - 0.9 * (a[0] - a.min())
        it90 = lg.step[int(np.argmax(a <= target))]
        rows.append({"run": label, "start loss": a[0], "final loss": a[-1], "iters to 90% drop": it90,
                     "evaluations": lg.step[-1], "seconds": lg.seconds,
                     "ms / evaluation": 1000 * lg.seconds / lg.step[-1]})
    return pd.DataFrame(rows), plot_convergence(logs), logs


def plot_convergence(logs: dict, height=3.4):
    fig, axes = _subplots(1, 3, figsize=(11, height))
    for ax, (attr, title) in zip(axes, [("total", "Total loss"), ("style", "Style loss (β·S)"),
                                        ("content", "Content loss (α·C)")]):
        for i, (label, lg) in enumerate(logs.items()):
            ax.plot(lg.step, getattr(lg, attr), label=label, color=SERIES[i % len(SERIES)], lw=1.6)
        ax.set_yscale("log")
        ax.set_xlabel("closure evaluation")
        ax.set_title(title)
    if logs:
        axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    return fig


def live_curve_fig(log: E.OptLog):
    """Small live loss plot for the Studio tab."""
    fig, ax = _subplots(figsize=(5.6, 2.6))
    if log is not None and log.step:
        ax.plot(log.step, log.total, color=PALETTE["bone"], lw=1.6, label="total")
        ax.plot(log.step, log.style, color=PALETTE["madder"], lw=1.2, label="style")
        ax.plot(log.step, log.content, color=PALETTE["verdigris"], lw=1.2, label="content")
        ax.set_yscale("log")
        ax.legend(fontsize=7.5, ncol=3, loc="upper right")
    ax.set_xlabel("closure evaluation")
    ax.set_title("Loss during optimisation", fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Style-scalability / capability table (static facts about these implementations)
# --------------------------------------------------------------------------- #
CAPABILITY_ROWS = [
    # method, efficient, arbitrary style, learning-free (no per-style training), family
    ("Gatys et al. (2016)", "No", "Yes", "Yes", "Image optimisation"),
    ("AdaIN (Huang & Belongie, 2017)", "Yes", "Yes", "No (trained once)", "Model optimisation, arbitrary style"),
    ("Magenta (Ghiasi et al., 2017)", "Yes", "Yes", "No (trained once)", "Model optimisation, arbitrary style"),
]


def capability_df():
    return pd.DataFrame(CAPABILITY_ROWS, columns=["Method", "Efficient", "Arbitrary style", "Learning-free", "Family"])


# --------------------------------------------------------------------------- #
# Studio "quick metrics" read-out
# --------------------------------------------------------------------------- #
def quick_metrics(content, style, output, seconds, content_weight=1.0, style_weight=1e6) -> dict:
    m = score_output(content, style, output, content_weight, style_weight)
    m["seconds"] = seconds
    return m
