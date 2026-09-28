import gradio as gr

PALETTE = {
    "ink": "#0f0e13", "panel": "#17151d", "raised": "#201d28", "line": "#2e2a38",
    "bone": "#eae5db", "ash": "#9b95a8",
    "madder": "#d0507a", "verdigris": "#58b5a4", "ochre": "#d9a441", "smalt": "#7d93e0",
}
SERIES = [PALETTE["madder"], PALETTE["verdigris"], PALETTE["ochre"], PALETTE["smalt"], "#b48ce0"]


def apply_mpl():
    import matplotlib as mpl

    mpl.use("Agg")
    p = PALETTE
    mpl.rcParams.update({
        "figure.facecolor": p["panel"], "axes.facecolor": p["panel"], "savefig.facecolor": p["panel"],
        "axes.edgecolor": p["line"], "axes.labelcolor": p["ash"], "axes.titlecolor": p["bone"],
        "xtick.color": p["ash"], "ytick.color": p["ash"], "text.color": p["bone"],
        "grid.color": p["line"], "grid.alpha": 0.6, "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "legend.facecolor": p["raised"], "legend.edgecolor": p["line"], "legend.labelcolor": p["bone"],
        "font.size": 10, "axes.titlesize": 11, "figure.dpi": 110,
    })


def _ramp(*vals):
    keys = ["c50", "c100", "c200", "c300", "c400", "c500", "c600", "c700", "c800", "c900", "c950"]
    return gr.themes.Color(**dict(zip(keys, vals)), name="custom")


MADDER = _ramp("#fdf0f4", "#fbdde6", "#f6bccd", "#ee90ab", "#e2607f", "#d0507a",
               "#b93f68", "#983355", "#7b2c48", "#652840", "#3a1224")
INK = _ramp("#f3f0f6", "#e6e1ec", "#cdc6d7", "#b0a8bd", "#8f879e", "#726a82",
            "#59526a", "#423d52", "#2e2a38", "#1d1a25", "#121016")


def build_theme():
    p = PALETTE
    t = gr.themes.Base(
        primary_hue=MADDER, secondary_hue=INK, neutral_hue=INK,
        font=[gr.themes.GoogleFont("Hanken Grotesk"), "system-ui", "sans-serif"],
        radius_size=gr.themes.sizes.radius_sm,
    )
    dark = dict(
        body_background_fill=p["ink"], body_text_color=p["bone"], body_text_color_subdued=p["ash"],
        background_fill_primary=p["panel"], background_fill_secondary=p["ink"],
        block_background_fill=p["panel"], block_border_color=p["line"], block_border_width="1px",
        block_label_background_fill=p["panel"], block_label_text_color=p["ash"],
        block_title_text_color=p["bone"], block_info_text_color=p["ash"],
        input_background_fill=p["raised"], input_border_color=p["line"], input_border_color_focus=p["madder"],
        border_color_primary=p["line"], panel_background_fill=p["panel"],
        button_primary_background_fill=p["madder"], button_primary_background_fill_hover="#e0608a",
        button_primary_text_color="#fff", button_primary_border_color=p["madder"],
        button_secondary_background_fill=p["raised"], button_secondary_text_color=p["bone"],
        button_secondary_border_color=p["line"], button_secondary_background_fill_hover=p["line"],
        slider_color=p["madder"], checkbox_background_color=p["raised"],
        checkbox_background_color_selected=p["madder"], checkbox_border_color=p["line"],
        table_even_background_fill=p["panel"], table_odd_background_fill=p["raised"],
        table_border_color=p["line"], color_accent_soft=p["raised"],
    )
    kwargs = {}
    for k, v in dark.items():  
        kwargs[k] = v
        kwargs[k + "_dark"] = v
    return t.set(**kwargs)


FORCE_DARK_JS = "() => { document.body.classList.add('dark'); document.documentElement.style.colorScheme = 'dark'; }"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&display=swap');
:root { --nc-madder:#d0507a; --nc-ochre:#d9a441; --nc-line:#2e2a38; --nc-ash:#9b95a8; --nc-bone:#eae5db; }
.gradio-container { max-width: 1320px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* ---- hero ---- */
.nc-hero { padding: 34px 4px 22px; border-bottom: 1px solid var(--nc-line); margin-bottom: 18px; }
.nc-title { font-family: 'Instrument Serif', Georgia, serif; font-size: clamp(2.6rem, 6vw, 4.4rem);
            line-height: 1; letter-spacing: -0.01em; color: var(--nc-bone); }
.nc-title i { color: var(--nc-madder); font-style: italic; }
.nc-sub { max-width: 62ch; margin-top: 12px; color: var(--nc-ash); font-size: 1.02rem; line-height: 1.55; }
.nc-engines { display: flex; flex-wrap: wrap; gap: 0; margin-top: 18px; font-size: .86rem; color: var(--nc-ash); }
.nc-engines span { padding: 2px 14px; border-left: 1px solid var(--nc-line); }
.nc-engines span:first-child { padding-left: 0; border-left: none; }

/* ---- tabs ---- */
button[role="tab"] { font-weight: 500; letter-spacing: .01em; }
button[role="tab"][aria-selected="true"] { color: var(--nc-bone) !important; border-color: var(--nc-madder) !important; }

/* ---- section headings inside tabs ---- */
.nc-h { font-family: 'Instrument Serif', Georgia, serif; font-size: 1.55rem; color: var(--nc-bone);
        margin: 4px 0 2px; font-weight: 400; }
.nc-note { color: var(--nc-ash); font-size: .9rem; line-height: 1.5; }

/* ---- the framed canvas: the one loud element ---- */
#canvas-frame, #canvas-frame-2 { background: #0a090d !important; border: 1px solid #5a4a2a !important; border-radius: 2px !important;
                padding: 16px !important;
                box-shadow: inset 0 0 0 1px #0a090d, inset 0 0 0 2px #3a3020, 0 0 0 7px #17151d, 0 0 0 8px #3a3020,
                            0 24px 60px rgba(0,0,0,.55); }
#canvas-frame img, #canvas-frame-2 img { border-radius: 0 !important; }

/* ---- metric read-out ---- */
.nc-metrics table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
.nc-metrics th, .nc-metrics td { padding: 6px 10px; border-bottom: 1px solid var(--nc-line); text-align: left; }
.nc-metrics th { color: var(--nc-ash); font-weight: 500; }
.nc-metrics td:last-child { color: var(--nc-ochre); }
.nc-status { font-variant-numeric: tabular-nums; color: var(--nc-ash); min-height: 1.6em; }

/* ---- data frames ---- */
.nc-table { font-variant-numeric: tabular-nums; }
@media (prefers-reduced-motion: no-preference) { button { transition: background-color .15s, border-color .15s; } }
"""

HERO_HTML = """
<div class="nc-hero">
  <div class="nc-title">Neuro<i>Canvas</i></div>
  <div class="nc-sub">An explainable laboratory for neural style transfer: run multiple different
  engines on the same image, watch the optimisation unfold, and measure what each one costs
  and what it delivers.</div>
  <div class="nc-engines">
    <span>Gatys optimisation</span><span>AdaIN</span><span>Magenta TF-Hub</span>
    <span>Object-aware</span><span>Video</span>
  </div>
</div>
"""
