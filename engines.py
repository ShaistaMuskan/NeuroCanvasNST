from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import models

# Hugging Face 
try:
    import spaces  # type: ignore

    gpu = spaces.GPU
except Exception:  # pragma: no cover

    def gpu(fn=None, **_kw):
        if callable(fn):
            return fn
        return lambda f: f


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SIZE = 512 if torch.cuda.is_available() else 256
MODELS_DIR = Path(__file__).parent / "models"


# Image helpers
def fit_size(w: int, h: int, long_side: int, mult: int = 8) -> tuple[int, int]:
    """Scale (w, h) so the long side == long_side, snapped to a multiple of `mult`."""
    s = long_side / max(w, h)
    return max(mult, round(w * s / mult) * mult), max(mult, round(h * s / mult) * mult)


def to_tensor(img: Image.Image, long_side: int, size_like: tuple[int, int] | None = None):
    img = img.convert("RGB")
    w, h = size_like or fit_size(*img.size, long_side)
    img = img.resize((w, h), Image.LANCZOS)
    return TF.to_tensor(img).unsqueeze(0).to(DEVICE)


def to_pil(t: torch.Tensor) -> Image.Image:
    return TF.to_pil_image(t.detach().float().cpu().squeeze(0).clamp(0, 1))


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


# Optimisation log (used for live curves + the convergence analysis)
@dataclass
class OptLog:
    step: list = field(default_factory=list)
    total: list = field(default_factory=list)
    content: list = field(default_factory=list)
    style: list = field(default_factory=list)
    tv: list = field(default_factory=list)
    t: list = field(default_factory=list)
    _t0: float = field(default_factory=time.perf_counter)

    def add(self, n, total, content, style, tv):
        self.step.append(n)
        self.total.append(total)
        self.content.append(content)
        self.style.append(style)
        self.tv.append(tv)
        self.t.append(time.perf_counter() - self._t0)

    @property
    def seconds(self):
        return self.t[-1] if self.t else 0.0


# Gatys et al. (2016) -- VGG-19, Gram-matrix style loss, L-BFGS
CONTENT_LAYER = "relu4_2"
STYLE_LAYERS = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]


class VGG19Features(nn.Module):
    """torchvision VGG-19 with max-pool -> avg-pool (as in Gatys et al.)."""

    def __init__(self, wanted):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
        layers, names = [], []
        block, conv = 1, 0
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                conv += 1
                name = f"conv{block}_{conv}"
            elif isinstance(layer, nn.ReLU):
                name = f"relu{block}_{conv}"
                layer = nn.ReLU(inplace=False)
            else:
                name = f"pool_{block}"
                layer = nn.AvgPool2d(2, 2)
                block += 1
                conv = 0
            names.append(name)
            layers.append(layer)
        last = max(names.index(n) for n in wanted)
        self.layers = nn.ModuleList(layers[: last + 1])
        self.names = names[: last + 1]
        self.wanted = set(wanted)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        x = (x - self.mean) / self.std
        out = {}
        for n, l in zip(self.names, self.layers):
            x = l(x)
            if n in self.wanted:
                out[n] = x
        return out


_LOSS_NET = None


def get_loss_net() -> VGG19Features:
    """One shared VGG-19: drives Gatys optimisation AND is the fixed 'judge' for evaluation."""
    global _LOSS_NET
    if _LOSS_NET is None:
        _LOSS_NET = VGG19Features([CONTENT_LAYER] + STYLE_LAYERS).to(DEVICE)
    return _LOSS_NET


def gram_matrix(f: torch.Tensor) -> torch.Tensor:
    b, c, h, w = f.shape
    x = f.reshape(b * c, h * w)
    return (x @ x.t()) / (b * c * h * w)


def tv_loss(img: torch.Tensor) -> torch.Tensor:
    return (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().sum() + (
        img[:, :, :, 1:] - img[:, :, :, :-1]
    ).abs().sum()


class GatysCriterion:
    """Content (conv4_2 MSE) + style (mean Gram MSE over 5 layers) + TV.

    Used by the optimiser *and* by the evaluation suite to score the output of any method.
    """

    def __init__(self, content: torch.Tensor, style: torch.Tensor):
        net = get_loss_net()
        with torch.no_grad():
            self.c_target = net(content)[CONTENT_LAYER]
            sf = net(style)
            self.s_targets = {l: gram_matrix(sf[l]) for l in STYLE_LAYERS}

    def __call__(self, x: torch.Tensor):
        f = get_loss_net()(x)
        c = F.mse_loss(f[CONTENT_LAYER], self.c_target)
        s = sum(F.mse_loss(gram_matrix(f[l]), self.s_targets[l]) for l in STYLE_LAYERS) / len(STYLE_LAYERS)
        return c, s, tv_loss(x)


def gatys_stream(
    content: Image.Image,
    style: Image.Image,
    content_weight=1.0,
    style_weight=1e6,
    tv_weight=1e-5,
    lr=1.0,
    iterations=200,
    init="content",
    size=None,
    preview_every=20,
    log: OptLog | None = None,
):
    """Generator -> (PIL preview, OptLog, done). `iterations` counts closure evaluations."""
    size = size or DEFAULT_SIZE
    c = to_tensor(content, size)
    s = to_tensor(style, size)
    crit = GatysCriterion(c, s)
    img = (c.clone() if init == "content" else torch.rand_like(c)).requires_grad_(True)
    opt = optim.LBFGS([img], lr=lr, max_iter=10)
    log = log if log is not None else OptLog()
    n = [0]

    def closure():
        with torch.no_grad():
            img.clamp_(0, 1)
        opt.zero_grad()
        cl, sl, tv = crit(img)
        wc, ws, wt = content_weight * cl, style_weight * sl, tv_weight * tv
        total = wc + ws + wt
        total.backward()
        n[0] += 1
        log.add(n[0], total.item(), wc.item(), ws.item(), wt.item())
        return total

    yield to_pil(img), log, False
    last = 0
    while n[0] < iterations:
        opt.step(closure)
        if n[0] - last >= preview_every:
            last = n[0]
            yield to_pil(img), log, False
    with torch.no_grad():
        img.clamp_(0, 1)
    yield to_pil(img), log, True


# AdaIN (Huang & Belongie, 2017)
ADAIN_URLS = {
    "vgg_normalised.pth": "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth",
    "decoder.pth": "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth",
}
_ADAIN = None


def get_adain():
    global _ADAIN
    if _ADAIN is None:
        import adain_net as net

        MODELS_DIR.mkdir(exist_ok=True)
        for fname, url in ADAIN_URLS.items():
            dst = MODELS_DIR / fname
            if not dst.exists():
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 NeuroCanvas"})
                with urllib.request.urlopen(req) as r, open(str(dst) + ".part", "wb") as f:
                    f.write(r.read())
                os.replace(str(dst) + ".part", dst)
        dec, vgg = net.decoder, net.vgg
        dec.load_state_dict(torch.load(MODELS_DIR / "decoder.pth", map_location="cpu"))
        vgg.load_state_dict(torch.load(MODELS_DIR / "vgg_normalised.pth", map_location="cpu"))
        vgg = nn.Sequential(*list(vgg.children())[:31])  # up to relu4_1
        for m in (vgg, dec):
            m.eval().to(DEVICE)
            for p in m.parameters():
                p.requires_grad_(False)
        _ADAIN = (vgg, dec)
    return _ADAIN


def _coral_to_content(style_t: torch.Tensor, content_t: torch.Tensor) -> torch.Tensor:
    """Colour-preserving pre-processing (CORAL). The reference implementation
    needs 3-D CPU float tensors, so we move data across and back."""
    from adain_function import coral

    out = coral(style_t.squeeze(0).float().cpu(), content_t.squeeze(0).float().cpu())
    return out.unsqueeze(0).to(DEVICE)


@torch.no_grad()
def adain_stylize(
    content: Image.Image,
    style: Image.Image,
    alpha=1.0,
    preserve_color=False,
    style2: Image.Image | None = None,
    blend=0.5,
    size=None,
) -> Image.Image:
    """Feed-forward AdaIN; optional 2-style interpolation (blend = weight of style 1)."""
    from adain_function import adaptive_instance_normalization as adain

    vgg, dec = get_adain()
    size = size or DEFAULT_SIZE
    c = to_tensor(content, size)
    styles = [(style, blend)] + ([(style2, 1 - blend)] if style2 is not None else [])
    if style2 is None:
        styles = [(style, 1.0)]
    cf = vgg(c)
    t = 0
    for simg, w in styles:
        s = to_tensor(simg, size)
        if preserve_color:
            s = _coral_to_content(s, c)
        t = t + w * adain(cf, vgg(s))
    t = alpha * t + (1 - alpha) * cf
    return to_pil(dec(t))


def adain_adam_stream(
    content, style, content_weight=1.0, style_weight=15.0, tv_weight=1e-5,
    lr=0.03, iterations=150, size=None, preview_every=10, log: OptLog | None = None,
):
    """AdaIN-encoder-space optimisation with Adam (from the AdaIN notebook)."""
    from adain_function import adaptive_instance_normalization as adain

    vgg, _ = get_adain()
    size = size or DEFAULT_SIZE
    c, s = to_tensor(content, size), to_tensor(style, size)
    with torch.no_grad():
        cf, sf = vgg(c), vgg(s)
        target = adain(cf, sf)
        s_mean, s_std = sf.mean((2, 3)), sf.var((2, 3)).sqrt()
    img = c.clone().requires_grad_(True)
    opt = optim.Adam([img], lr=lr)
    log = log if log is not None else OptLog()
    yield to_pil(img), log, False
    for i in range(1, iterations + 1):
        opt.zero_grad()
        g = vgg(img)
        cl = F.mse_loss(g, target)
        sl = F.mse_loss(g.mean((2, 3)), s_mean) + F.mse_loss(g.var((2, 3)).sqrt(), s_std)
        tv = (img[:, :, 1:] - img[:, :, :-1]).abs().mean() + (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()
        wc, ws, wt = content_weight * cl, style_weight * sl, tv_weight * tv
        total = wc + ws + wt
        total.backward()
        opt.step()
        with torch.no_grad():
            img.clamp_(0, 1)
        log.add(i, total.item(), wc.item(), ws.item(), wt.item())
        if i % preview_every == 0 and i < iterations:
            yield to_pil(img), log, False
    yield to_pil(img), log, True


# Magenta arbitrary-image-stylization (TF-Hub) - TensorFlow
MAGENTA_URLS = [
    "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2",
    "https://www.kaggle.com/models/google/arbitrary-image-stylization-v1/tensorFlow1/256/2",
]
_MAGENTA = None


def get_magenta():
    global _MAGENTA
    if _MAGENTA is None:
        os.environ.setdefault("TFHUB_MODEL_LOAD_FORMAT", "COMPRESSED")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError as e:
            raise RuntimeError(
                "TensorFlow / tensorflow-hub are not installed, so the Magenta engine is unavailable."
            ) from e
        try:  # keep the GPU for PyTorch
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass
        last = None
        for url in MAGENTA_URLS:
            try:
                _MAGENTA = hub.load(url)
                break
            except Exception as e:  # noqa: BLE001
                last = e
        if _MAGENTA is None:
            raise RuntimeError(f"Could not load the Magenta model: {last}")
    return _MAGENTA


def magenta_stylize(content: Image.Image, style: Image.Image, size=None, blend=1.0) -> Image.Image:
    import tensorflow as tf

    model = get_magenta()
    size = size or DEFAULT_SIZE
    c = content.convert("RGB")
    c = c.resize(fit_size(*c.size, size, mult=1), Image.LANCZOS)
    s = style.convert("RGB").resize((256, 256), Image.LANCZOS)  # model was trained on 256px styles
    ca = np.asarray(c, dtype=np.float32)[None] / 255.0
    sa = np.asarray(s, dtype=np.float32)[None] / 255.0
    out = model(tf.constant(ca), tf.constant(sa))[0].numpy()[0]
    out = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).resize(c.size, Image.LANCZOS)
    return out if blend >= 1.0 else Image.blend(c, out, blend)


# Object-aware NST (YOLOv8-seg mask + any stylizer)
_YOLO = None


def get_yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO

        _YOLO = YOLO("yolov8n-seg.pt")
    return _YOLO


def _yolo_results(pil: Image.Image):
    bgr = np.asarray(pil.convert("RGB"))[..., ::-1].copy()  # ultralytics expects BGR ndarrays
    return get_yolo()(bgr, verbose=False)[0]


def detect_objects(pil: Image.Image | None) -> list[str]:
    if pil is None:
        return []
    r = _yolo_results(pil)
    if r.boxes is None or len(r.boxes) == 0:
        return []
    return sorted({r.names[int(c)] for c in r.boxes.cls.tolist()})


def object_mask(pil: Image.Image, classes: list[str]) -> np.ndarray:
    """Binary mask (H, W) float32 for the requested COCO classes, built from
    the segmentation polygons so it is exactly aligned with the input image."""
    import cv2

    w, h = pil.size
    mask = np.zeros((h, w), np.float32)
    r = _yolo_results(pil)
    if r.masks is None:
        return mask
    wanted = {c.lower() for c in classes}
    for poly, cls in zip(r.masks.xy, r.boxes.cls.tolist()):
        if r.names[int(cls)].lower() in wanted and len(poly) >= 3:
            cv2.fillPoly(mask, [poly.astype(np.int32)], 1.0)
    return mask


def keep_luminance_only(content: np.ndarray, stylised: np.ndarray) -> np.ndarray:
    """Stylised L channel + original a/b chroma (texture transfer without recolouring). Inputs float [0,1]."""
    import cv2

    c = cv2.cvtColor((content * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    s = cv2.cvtColor((stylised * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    merged = cv2.merge([s[..., 0], c[..., 1], c[..., 2]])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def object_nst(
    content: Image.Image, style: Image.Image, classes: list[str], target="objects",
    feather=15, keep_colors=False, stylizer="Magenta (TF-Hub)", strength=1.0, size=512,
):
    import cv2

    content = content.convert("RGB")
    content = content.resize(fit_size(*content.size, size, mult=1), Image.LANCZOS)
    if stylizer.startswith("AdaIN"):
        styl = adain_stylize(content, style, alpha=strength, size=size).resize(content.size)
    else:
        styl = magenta_stylize(content, style, size=size, blend=strength)
    c_np = np.asarray(content, np.float32) / 255.0
    s_np = np.asarray(styl.resize(content.size), np.float32) / 255.0
    if keep_colors:
        s_np = keep_luminance_only(c_np, s_np)
    m = object_mask(content, classes)
    k = int(feather) | 1
    m = cv2.GaussianBlur(m, (k, k), 0)[..., None]
    alpha = 1 - m if target == "background" else m
    out = np.clip(alpha * s_np + (1 - alpha) * c_np, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8)), Image.fromarray((alpha[..., 0] * 255).astype(np.uint8))


# Video NST (frame-by-frame AdaIN, style encoded once, optional feature-space smoothing)
@torch.no_grad()
def video_stylize(
    video_path: str, style: Image.Image, alpha=1.0, size=384,
    max_seconds=6.0, smooth=0.0, progress=None,
):
    """Returns (out_path, stats dict). `smooth` in [0,1): EMA of AdaIN features across
    frames to damp flicker (0 = independent frames)."""
    import cv2
    import imageio.v2 as imageio
    from adain_function import adaptive_instance_normalization as adain

    vgg, dec = get_adain()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10**9
    n_frames = min(total, int(fps * max_seconds))
    W, H = fit_size(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), size)
    sf = vgg(to_tensor(style, size))  # encoded once
    out_path = str(Path(video_path).with_name(Path(video_path).stem + "_neurocanvas.mp4"))
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)

    prev_t, prev_in, prev_out = None, None, None
    d_in, d_out = [], []
    t0 = time.perf_counter()
    done = 0
    while done < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(DEVICE)
        cf = vgg(x)
        t = alpha * adain(cf, sf) + (1 - alpha) * cf
        if smooth > 0 and prev_t is not None:
            t = (1 - smooth) * t + smooth * prev_t
        prev_t = t
        y = dec(t).clamp(0, 1)[0].permute(1, 2, 0).mul(255).byte().cpu().numpy()
        y = cv2.resize(y, (W, H)) if y.shape[:2] != (H, W) else y
        writer.append_data(y)
        if prev_in is not None:
            d_in.append(np.abs(rgb.astype(np.float32) - prev_in).mean())
            d_out.append(np.abs(y.astype(np.float32) - prev_out).mean())
        prev_in, prev_out = rgb.astype(np.float32), y.astype(np.float32)
        done += 1
        if progress:
            progress(done / n_frames, desc=f"Frame {done}/{n_frames}")
    sync()
    elapsed = time.perf_counter() - t0
    cap.release()
    writer.close()
    stats = {
        "frames": done,
        "resolution": f"{W}x{H}",
        "ms_per_frame": 1000 * elapsed / max(done, 1),
        "processing_fps": done / max(elapsed, 1e-9),
        # >1 => the stylisation adds frame-to-frame change (flicker) beyond the source's own motion
        "flicker_ratio": (np.mean(d_out) / max(np.mean(d_in), 1e-6)) if d_in else float("nan"),
    }
    return out_path, stats


# Unified single-image entry point (used by the Studio tab and the benchmarks)
def last_of(stream):
    out = None
    for out in stream:
        pass
    return out
