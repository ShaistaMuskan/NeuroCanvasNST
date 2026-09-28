# NeuroCanvas

An explainable laboratory for neural style transfer analysis and hyperparameter exploration.

| Tab | What it does |
|---|---|
| Studio | Gatys (L-BFGS), AdaIN (feed-forward, CORAL colour, 2-style blending), Magenta (TF-Hub), AdaIN + Adam. Live loss curve and per-run scores. |
| Compare runs | Gallery + table of every run this session (settings, time, content loss, style loss, SSIM). |
| Object NST | YOLOv8-seg picks objects; only those (or the background) get stylised. |
| Video NST | Frame-by-frame AdaIN with temporal smoothing and a flicker metric. |
| Evaluation lab | Speed (256/512/1024), loss comparison under one fixed objective, convergence curves. |
| Parameter guide | What every knob does. |

## Setup

1. Copy your images into `assets/content/` and `assets/styles/` (same file names as in the notebooks:
   `car.jpg`, `starry_night.jpg`, ...). Any other .jpg/.png you add shows up automatically.
2. `pip install -r requirements.txt` then `python app.py`.

Model weights (AdaIN, VGG-19, YOLOv8n-seg, Magenta) download on first use.

## Quantitative evaluation from the command line

```bash
python run_evaluation.py --sizes 256 512 1024 --repeats 10 --gatys-iters 300
```
Writes CSVs, PNGs and `report.md` to `evaluation_results/`.

## Credits

AdaIN code vendored from naoto0804/pytorch-AdaIN (MIT, see `LICENSE_AdaIN`).
