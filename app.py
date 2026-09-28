from __future__ import annotations

import time

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

import engines as E
import library as L
import metrics as M
import theme as T

CONTENT = L.content_library()
STYLES = L.style_library()
ENGINE_CHOICES = [
    ("Gatys: optimisation", "gatys"),
    ("AdaIN: feed-forward", "adain"),
    ("Magenta: TF-Hub", "magenta"),
    ("AdaIN + Adam: optimisation", "adam"),
]
ENGINE_NAMES = {"gatys": "Gatys", "adain": "AdaIN", "magenta": "Magenta", "adam": "AdaIN+Adam"}
MAX_HISTORY = 12


# small helpers
def open_preset(path):
    return Image.open(path).convert("RGB") if path else None


def pick_content(name):
    return open_preset(CONTENT.get(name))


def pick_style(name):
    item = STYLES.get(name)
    return open_preset(item["path"]) if item else None, L.style_caption(name, STYLES)


def need(*imgs):
    if any(i is None for i in imgs):
        raise gr.Error("Please provide both a content image and a style image (upload one or pick a preset).")


def strength_from(cw, sw):
    """Feed-forward models have no loss to weight, so α (content) and β (style) set the
    mixing ratio instead: strength = β / (α + β). 1:1 -> 0.5, β >> α -> 1.0."""
    return float(sw) / (float(cw) + float(sw))


def metrics_html(m: dict, extra: dict | None = None) -> str:
    rows = [("Time", f"{m['seconds']:.2f} s"), ("Content loss (conv4_2 MSE)", f"{m['content_loss']:.4g}"),
            ("Style loss (mean Gram MSE)", f"{m['style_loss']:.4g}"), ("SSIM to content (grayscale)", f"{m['ssim']:.3f}")]
    rows += list((extra or {}).items())
    body = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<div class='nc-metrics'><table>{body}</table></div>"


# Studio
@E.gpu(duration=120)
def run_studio(engine, content, style, style2, size,
               g_cw, g_sw, g_tv, g_lr, g_iters, g_init,
               a_cw, a_sw, a_color, a_blend, m_cw, m_sw,
               d_cw, d_sw, d_tv, d_lr, d_iters,
               do_metrics, history, progress=gr.Progress()):
    need(content, style)
    size = int(size)
    history = list(history or [])
    t0 = time.perf_counter()
    log, final = None, None
    cw, sw = 1.0, 1e6
    label = ENGINE_NAMES[engine]

    if engine in ("gatys", "adam"):
        if engine == "gatys":
            cw, sw = g_cw, g_sw
            stream = E.gatys_stream(content, style, g_cw, g_sw, g_tv, g_lr, int(g_iters), g_init, size)
            total_it, settings = int(g_iters), f"β {g_sw:.0e}, {int(g_iters)} it, {g_init}"
        else:
            stream = E.adain_adam_stream(content, style, d_cw, d_sw, d_tv, d_lr, int(d_iters), size)
            total_it, settings = int(d_iters), f"Adam lr {d_lr}, {int(d_iters)} it"
        for pil, log, done in stream:
            final = pil
            n = log.step[-1] if log.step else 0
            status = (f"Step {min(n, total_it)} of {total_it}, {time.perf_counter() - t0:.1f} s elapsed"
                      + (f", loss {log.total[-1]:.1f}" if log.total else ""))
            yield pil, status, M.live_curve_fig(log), gr.skip(), history
            progress(min(n / total_it, 1.0))
    else:
        if engine == "adain":
            s2 = style2 if style2 is not None else None
            a_alpha = strength_from(a_cw, a_sw)
            final = E.adain_stylize(content, style, a_alpha, a_color, s2, a_blend, size)
            settings = f"α:β {a_cw:g}:{a_sw:g} (strength {a_alpha:.2f})" + (", 2-style blend" if s2 is not None else "") + (", CORAL" if a_color else "")
        else:
            m_strength = strength_from(m_cw, m_sw)
            final = E.magenta_stylize(content, style, size, m_strength)
            settings = f"α:β {m_cw:g}:{m_sw:g} (strength {m_strength:.2f})"

    secs = time.perf_counter() - t0
    fig = M.live_curve_fig(log) if log is not None else None
    m_html, m = "", None
    if do_metrics:
        m = M.quick_metrics(content, style, final, secs, cw, sw)
        m_html = metrics_html(m)
    history.append({"img": final, "title": f"{label} ({settings})", "engine": label, "settings": settings,
                    "seconds": secs, **({k: m[k] for k in ("content_loss", "style_loss", "ssim")} if m else {})})
    history = history[-MAX_HISTORY:]
    yield final, f"Finished in {secs:.1f} s", (fig if fig is not None else gr.skip()), m_html, history


def eff_text(cw, sw):
    return f"Effective style strength β/(α+β) = **{strength_from(cw, sw):.2f}**"


def render_history(history):
    if not history:
        return [], pd.DataFrame(columns=["Run", "Engine", "Settings", "Time (s)", "Content loss", "Style loss", "SSIM"])
    gallery = [(h["img"], f"Run {i + 1}: {h['title']}") for i, h in enumerate(history)]
    rows = [{"Run": i + 1, "Engine": h["engine"], "Settings": h["settings"], "Time (s)": round(h["seconds"], 2),
             "Content loss": h.get("content_loss"), "Style loss": h.get("style_loss"), "SSIM": h.get("ssim")}
            for i, h in enumerate(history)]
    return gallery, pd.DataFrame(rows)


# Object NST
def scan_objects(content):
    if content is None:
        return gr.CheckboxGroup(choices=[], value=[], label="Detected objects")
    found = E.detect_objects(content)
    if not found:
        return gr.CheckboxGroup(choices=[], value=[], label="Detected objects (none found; try another image)")
    return gr.CheckboxGroup(choices=found, value=[found[0]], label=f"Detected objects ({len(found)})")


@E.gpu(duration=90)
def run_object(content, style, classes, target, stylizer, strength, feather, keep_colors, size):
    need(content, style)
    if not classes:
        raise gr.Error("Select at least one detected object.")
    t0 = time.perf_counter()
    out, mask = E.object_nst(content, style, classes, "background" if target.startswith("Background") else "objects",
                             feather, keep_colors, stylizer, strength, int(size))
    return out, mask, f"Done in {time.perf_counter() - t0:.1f} s"


# Video NST
@E.gpu(duration=120)
def run_video(video, style, alpha, size, seconds, smooth, progress=gr.Progress()):
    if video is None or style is None:
        raise gr.Error("Please provide a video and a style image.")
    path, st = E.video_stylize(video, style, alpha, int(size), float(seconds), float(smooth), progress)
    rows = pd.DataFrame([{"Frames": st["frames"], "Resolution": st["resolution"],
                          "ms / frame": round(st["ms_per_frame"], 1), "Processing FPS": round(st["processing_fps"], 2),
                          "Flicker ratio": round(st["flicker_ratio"], 3)}])
    return path, rows


# Evaluation lab
METHOD_NAMES = list(M.method_registry().keys())


@E.gpu(duration=120)
def eval_speed(content, style, methods, resolutions, repeats, g_iters, progress=gr.Progress()):
    need(content, style)
    if not methods:
        raise gr.Error("Select at least one method.")
    disp, _raw, fig = M.speed_benchmark(content, style, methods, sorted(int(r) for r in resolutions),
                                        int(repeats), int(g_iters), int(g_iters), progress)
    return disp, fig


@E.gpu(duration=120)
def eval_loss(c_names, s_names, methods, size, alpha, beta, g_iters, progress=gr.Progress()):
    if not c_names or not s_names or not methods:
        raise gr.Error("Pick at least one content preset, one style preset and one method.")
    pairs = [(c, open_preset(CONTENT[c]), s, open_preset(STYLES[s]["path"])) for c in c_names for s in s_names]
    df, summ, fig = M.loss_comparison(pairs, methods, int(size), alpha, beta, int(g_iters), int(g_iters), progress)
    return summ.round(5), fig, df.round(5)


@E.gpu(duration=120)
def eval_convergence(content, style, runs, iters, size, progress=gr.Progress()):
    need(content, style)
    if not runs:
        raise gr.Error("Select at least one run configuration.")
    df, fig, _ = M.convergence(content, style, runs, int(iters), int(size), progress=progress)
    return df.round(3), fig


# Guide text
GUIDE = r"""
### Gatys optimisation
* **Content weight α** scales the MSE between `conv4_2` feature maps. Higher values keep structure: edges, faces, architecture.
* **Style weight β** scales the Gram-matrix mismatch across `conv1_1 … conv5_1`. Higher values bring in heavier brushwork and colour.
* **α/β ratio** is the real dial. Around 10⁻³ gives a light touch with a legible subject; 10⁻⁴ to 10⁻⁵ gives deep painterly fusion.
* **Total variation** penalises differences between neighbouring pixels. It removes salt-and-pepper noise at the cost of very fine texture.
* **L-BFGS learning rate** is the step scale of the quasi-Newton update. 1.0 is standard; lower is steadier, and slower.
* **Init from content** converges fastest and keeps geometry. **Init from noise** is the unbiased setup from the original paper and needs more iterations.
* **Iterations** count loss evaluations (not optimiser steps); one optimiser step can use several.

### AdaIN
* **Style strength α** interpolates in feature space: `T = α·AdaIN(f(c), f(s)) + (1−α)·f(c)`. 0 reconstructs the content; 1 is full stylisation.
* **Preserve colour (CORAL)** first matches the style image's colour statistics to the content, so brushwork transfers but the palette stays.
* **Second style** produces two AdaIN-stylised feature maps and mixes them before decoding (style interpolation).
* **AdaIN + Adam** optimises the image directly against the AdaIN encoder's relu4_1 statistics. It is an optimisation-based cousin of the paper's feed-forward network.

### Magenta (TF-Hub)
A single feed-forward pass conditioned on a 256 px style image, run on CPU by design so TensorFlow never competes with PyTorch for the GPU. **Strength** blends the result with the content image in pixel space.

### Object-aware NST
YOLOv8-seg finds COCO objects; the selected classes become a soft mask (feathered by a Gaussian) and the stylised image is alpha-composited only where the mask says so. *Preserve colours* keeps the stylised luminance but the original chroma.

### Video NST
AdaIN is applied frame by frame with the style encoded once. **Temporal smoothing** averages the AdaIN features with the previous frame's. The **flicker ratio** in the results is the mean frame-to-frame change of the output divided by that of the input; values well above 1 mean the stylisation adds jitter.

### Reading the evaluation
* **Speed** is wall-clock time per image after a warm-up. Optimisation methods are timed for a fixed iteration count, so compare per-iteration cost as well.
* **Loss comparison** scores every method's output with the *same* Gatys objective (a VGG-19 judge). Optimisation is expected to reach lower loss; feed-forward models trade some of that for speed.
* **Convergence** logs the loss on every closure evaluation so you can see how quickly each initialisation or learning rate gets there.
* **SSIM** on grayscale measures how much of the content's structure survives. It is a companion to the content loss, not a replacement for looking at the picture.
"""

# UI
def head(text, note=""):
    return gr.HTML(f"<div class='nc-h'>{text}</div>" + (f"<div class='nc-note'>{note}</div>" if note else ""))


def build_ui():
    c_names, s_names = list(CONTENT), list(STYLES)
    c0 = c_names[0] if c_names else None
    s0 = "The Starry Night" if "The Starry Night" in STYLES else (s_names[0] if s_names else None)

    with gr.Blocks(title="NeuroCanvas") as demo:
        history = gr.State([])
        gr.HTML(T.HERO_HTML)

        with gr.Tabs():
            with gr.Tab("Studio"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=300):
                        head("Inputs")
                        engine = gr.Radio(ENGINE_CHOICES, value="adain", label="Engine")
                        c_drop = gr.Dropdown(c_names, value=c0, label="Content preset")
                        content = gr.Image(value=pick_content(c0), type="pil", label="Content", height=210,
                                           sources=["upload", "clipboard"])
                        s_drop = gr.Dropdown(s_names, value=s0, label="Style preset")
                        s_info = gr.Markdown(L.style_caption(s0, STYLES) if s0 else "")
                        style = gr.Image(value=pick_style(s0)[0] if s0 else None, type="pil", label="Style",
                                         height=210, sources=["upload", "clipboard"])
                        size = gr.Radio([256, 384, 512, 768], value=E.DEFAULT_SIZE, label="Working size (long side, px)")

                    with gr.Column(scale=3, min_width=280):
                        head("Hyperparameters")
                        with gr.Group(visible=False) as grp_gatys:
                            g_cw = gr.Slider(0.1, 10, 1.0, step=0.1, label="Content weight α")
                            g_sw = gr.Slider(1e4, 1e7, 1e6, step=1e4, label="Style weight β")
                            g_tv = gr.Slider(1e-7, 1e-3, 1e-5, step=1e-6, label="Total variation weight")
                            g_lr = gr.Slider(0.1, 1.5, 1.0, step=0.1, label="L-BFGS learning rate")
                            g_iters = gr.Slider(50, 500, 200, step=25, label="Iterations")
                            g_init = gr.Radio([("Content image", "content"), ("White noise", "noise")], value="content",
                                              label="Initialise from")
                        with gr.Group(visible=True) as grp_adain:
                            a_cw = gr.Slider(0.1, 10, 1.0, step=0.1, label="Content weight α")
                            a_sw = gr.Slider(0.1, 10, 1.0, step=0.1, label="Style weight β")
                            a_eff = gr.Markdown(eff_text(1.0, 1.0))
                            a_color = gr.Checkbox(False, label="Preserve content colours (CORAL)")
                            with gr.Accordion("Blend a second style", open=False):
                                s2_drop = gr.Dropdown(s_names, value=None, label="Second style preset")
                                style2 = gr.Image(type="pil", label="Second style", height=140,
                                                  sources=["upload", "clipboard"])
                                a_blend = gr.Slider(0, 1, 0.5, step=0.05, label="Weight of first style")
                        with gr.Group(visible=False) as grp_magenta:
                            m_cw = gr.Slider(0.1, 10, 1.0, step=0.1, label="Content weight α")
                            m_sw = gr.Slider(0.1, 10, 1.0, step=0.1, label="Style weight β")
                            m_eff = gr.Markdown(eff_text(1.0, 1.0))
                        with gr.Group(visible=False) as grp_adam:
                            d_cw = gr.Slider(0.1, 10, 1.0, step=0.1, label="Content weight")
                            d_sw = gr.Slider(1, 50, 15, step=1, label="Style weight")
                            d_tv = gr.Slider(0, 1e-4, 1e-5, step=1e-6, label="Total variation weight")
                            d_lr = gr.Slider(0.005, 0.1, 0.03, step=0.005, label="Adam learning rate")
                            d_iters = gr.Slider(50, 400, 150, step=25, label="Iterations")
                        do_metrics = gr.Checkbox(True, label="Score the result (loss, SSIM) when finished")
                        run = gr.Button("Paint", variant="primary", size="lg")

                    with gr.Column(scale=5, min_width=340):
                        head("Canvas")
                        out = gr.Image(label=None, show_label=False, type="pil", elem_id="canvas-frame", height=430,
                                       interactive=False)
                        status = gr.Markdown("Choose an engine and press Paint.", elem_classes="nc-status")
                        loss_plot = gr.Plot(show_label=False)
                        m_out = gr.HTML()

            with gr.Tab("Compare runs"):
                head("Every run this session", "Each finished Studio run is filed here with its settings and scores.")
                gallery = gr.Gallery(columns=3, height="auto", object_fit="contain", show_label=False)
                table = gr.Dataframe(interactive=False, wrap=True, elem_classes="nc-table")
                clear = gr.Button("Clear history", variant="secondary")

            with gr.Tab("Object NST"):
                head("Stylise only what you choose",
                     "YOLOv8-seg finds the objects; a feathered mask decides where the style lands.")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4):
                        o_content = gr.Image(type="pil", label="Content", height=250, sources=["upload", "clipboard"])
                        o_style = gr.Image(type="pil", label="Style", height=200, sources=["upload", "clipboard"],
                                           value=pick_style(s0)[0] if s0 else None)
                        o_load = gr.Button("Use Studio content", variant="secondary", size="sm")
                    with gr.Column(scale=3):
                        o_objs = gr.CheckboxGroup([], label="Detected objects")
                        o_target = gr.Radio(["Selected objects", "Background (keep objects natural)"],
                                            value="Selected objects", label="Stylise")
                        o_stylizer = gr.Radio(["Magenta (TF-Hub)", "AdaIN"], value="AdaIN", label="Stylizer")
                        o_strength = gr.Slider(0, 1, 1.0, step=0.05, label="Style strength")
                        o_feather = gr.Slider(1, 61, 15, step=2, label="Edge feathering (px)")
                        o_keep = gr.Checkbox(False, label="Preserve original colours")
                        o_size = gr.Radio([256, 384, 512, 768], value=512, label="Working size")
                        o_run = gr.Button("Stylise", variant="primary")
                    with gr.Column(scale=5):
                        o_out = gr.Image(label=None, show_label=False, elem_id="canvas-frame-2", height=380,
                                         interactive=False)
                        o_status = gr.Markdown(elem_classes="nc-status")
                        o_mask = gr.Image(label="Blend mask", height=160, interactive=False)

            with gr.Tab("Video NST"):
                head("Stylise a clip", "Frame-by-frame AdaIN with the style encoded once. Clips are capped for shared hardware.")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4):
                        v_in = gr.Video(label="Input video", format="mp4")
                        v_style = gr.Image(type="pil", label="Style", height=200, sources=["upload", "clipboard"],
                                           value=pick_style(s0)[0] if s0 else None)
                        v_drop = gr.Dropdown(s_names, value=s0, label="Style preset")
                    with gr.Column(scale=3):
                        v_alpha = gr.Slider(0, 1, 1.0, step=0.05, label="Style strength α")
                        v_size = gr.Radio([256, 384, 512], value=256 if not E.DEVICE.type == "cuda" else 384,
                                          label="Frame size (long side)")
                        v_secs = gr.Slider(1, 10, 4, step=1, label="Max seconds processed")
                        v_smooth = gr.Slider(0, 0.9, 0.3, step=0.05, label="Temporal smoothing")
                        v_run = gr.Button("Stylise video", variant="primary")
                    with gr.Column(scale=5):
                        v_out = gr.Video(label="Stylised", format="mp4")
                        v_stats = gr.Dataframe(headers=["Frames", "Resolution", "ms / frame", "Processing FPS", "Flicker ratio"], row_count=1, interactive=False, elem_classes="nc-table")

            with gr.Tab("Evaluation lab"):
                head("Quantitative evaluation",
                     "Follows the survey protocol: speed, loss under one fixed objective, and convergence. "
                     "Speed and convergence use the Studio content and style images; on a CPU-only host keep sizes small.")
                with gr.Tabs():
                    with gr.Tab("Speed"):
                        with gr.Row():
                            with gr.Column(scale=3):
                                e_methods = gr.CheckboxGroup(METHOD_NAMES, value=["AdaIN (feed-forward)", "Magenta (TF-Hub)", "Gatys (L-BFGS)"],
                                                             label="Methods")
                                e_res = gr.CheckboxGroup([256, 512, 1024], value=[256, 512], label="Resolutions")
                                e_rep = gr.Slider(1, 10, 3, step=1, label="Repeats (feed-forward methods)")
                                e_git = gr.Slider(10, 300, 50, step=10, label="Iterations for optimisation methods")
                                e_run = gr.Button("Run speed benchmark", variant="primary")
                            with gr.Column(scale=5):
                                e_tab = gr.Dataframe(interactive=False, elem_classes="nc-table")
                                e_fig = gr.Plot(show_label=False)
                    with gr.Tab("Loss comparison"):
                        with gr.Row():
                            with gr.Column(scale=3):
                                l_c = gr.CheckboxGroup(c_names, value=c_names[:2], label="Content images")
                                l_s = gr.CheckboxGroup(s_names, value=s_names[:2], label="Style images")
                                l_m = gr.CheckboxGroup(METHOD_NAMES, value=["AdaIN (feed-forward)", "Magenta (TF-Hub)", "Gatys (L-BFGS)"],
                                                       label="Methods")
                                l_size = gr.Radio([256, 384, 512], value=256, label="Evaluation size")
                                with gr.Row():
                                    l_a = gr.Number(1.0, label="α")
                                    l_b = gr.Number(1e6, label="β")
                                l_git = gr.Slider(10, 300, 100, step=10, label="Iterations for optimisation methods")
                                l_run = gr.Button("Run loss comparison", variant="primary")
                            with gr.Column(scale=5):
                                l_summ = gr.Dataframe(interactive=False, elem_classes="nc-table")
                                l_fig = gr.Plot(show_label=False)
                                with gr.Accordion("Per-image scores", open=False):
                                    l_raw = gr.Dataframe(interactive=False, elem_classes="nc-table")
                    with gr.Tab("Convergence"):
                        with gr.Row():
                            with gr.Column(scale=3):
                                k_runs = gr.CheckboxGroup(list(M.CONV_PRESETS), value=list(M.CONV_PRESETS)[:2], label="Runs to compare")
                                k_it = gr.Slider(50, 500, 150, step=25, label="Closure evaluations")
                                k_size = gr.Radio([256, 384, 512], value=256, label="Working size")
                                k_run = gr.Button("Run convergence study", variant="primary")
                            with gr.Column(scale=5):
                                k_tab = gr.Dataframe(interactive=False, elem_classes="nc-table")
                                k_fig = gr.Plot(show_label=False)
                    with gr.Tab("Method summary"):
                        gr.Dataframe(M.capability_df(), interactive=False, elem_classes="nc-table")
                        gr.Markdown(
                            "All models here are used **pre-trained**, so training time is not measured. "
                            "Style scalability is 'any style' for AdaIN and Magenta and one style per run for the "
                            "optimisation methods, which is what the *Styles/model* column in the speed table reports.")

            with gr.Tab("Parameter guide"):
                gr.Markdown(GUIDE)

        c_drop.change(pick_content, c_drop, content)
        s_drop.change(pick_style, s_drop, [style, s_info])
        s2_drop.change(lambda n: pick_style(n)[0], s2_drop, style2)
        v_drop.change(lambda n: pick_style(n)[0], v_drop, v_style)

        def toggle(e):
            return [gr.update(visible=e == k) for k in ("gatys", "adain", "magenta", "adam")]

        for cw_, sw_, eff_ in ((a_cw, a_sw, a_eff), (m_cw, m_sw, m_eff)):
            cw_.change(eff_text, [cw_, sw_], eff_)
            sw_.change(eff_text, [cw_, sw_], eff_)

        engine.change(toggle, engine, [grp_gatys, grp_adain, grp_magenta, grp_adam])

        run.click(
            run_studio,
            [engine, content, style, style2, size, g_cw, g_sw, g_tv, g_lr, g_iters, g_init,
             a_cw, a_sw, a_color, a_blend, m_cw, m_sw, d_cw, d_sw, d_tv, d_lr, d_iters, do_metrics, history],
            [out, status, loss_plot, m_out, history],
        ).then(render_history, history, [gallery, table])
        clear.click(lambda: ([], [], render_history([])[1]), None, [history, gallery, table])

        o_load.click(lambda x: x, content, o_content)
        o_content.change(scan_objects, o_content, o_objs)
        o_run.click(run_object, [o_content, o_style, o_objs, o_target, o_stylizer, o_strength, o_feather, o_keep, o_size],
                    [o_out, o_mask, o_status])

        v_run.click(run_video, [v_in, v_style, v_alpha, v_size, v_secs, v_smooth], [v_out, v_stats])

        e_run.click(eval_speed, [content, style, e_methods, e_res, e_rep, e_git], [e_tab, e_fig])
        l_run.click(eval_loss, [l_c, l_s, l_m, l_size, l_a, l_b, l_git], [l_summ, l_fig, l_raw])
        k_run.click(eval_convergence, [content, style, k_runs, k_it, k_size], [k_tab, k_fig])
    return demo


if __name__ == "__main__":
    build_ui().queue(max_size=16).launch(share=True, theme=T.build_theme(), css=T.CSS, js=T.FORCE_DARK_JS, ssr_mode=False)
