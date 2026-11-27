"""One-off: make the studio car photos transparent (Pillow + numpy only; no rembg).

    python tools/cutout_cars.py <input_dir> <output_dir>        # e.g. the originals -> assets/images/cars

How:
  1. background model: a smooth colour surface (quadratic in x, y) fitted to the image border, so light-grey
     gradients and vignettes are handled, not only pure white;
  2. mark what is clearly not background or floor shadow (outline, windows, wheels, colour), close small gaps
     in that outline, then flood from the image edges: whatever the flood cannot reach is car, so white or
     silver paint inside the outline is kept even when it matches the background colour;
  3. alpha: background and floor shadow 0 (the app adds a soft CSS drop shadow), a 1-2 px soft edge feathered
     inwards along the car outline;
  4. crop to the car with a small margin.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

BRIGHT_TOL = 10      # upper limit for the per-image background tolerance (set from the border noise)
DARK_TOL = 120       # floor shadow (only in the zone under the car): up to this much darker than the background
SAT_MAX = 16         # greyish: max - min channel
MARGIN = 0.03        # crop margin, share of the car's size
CLOSE_PX = 15        # gaps in the car outline up to about this size are bridged
SHADOW_RISE = 0.10   # floor shadow may start this share of the height above the lowest dark car part


def background_model(rgb: np.ndarray) -> np.ndarray:
    """Quadratic colour surface fitted to a 6 px border (all four sides)."""
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[0:h, 0:w]
    b = np.zeros((h, w), bool)
    b[:6], b[-6:], b[:, :6], b[:, -6:] = True, True, True, True
    x, y = xx[b] / w, yy[b] / h
    A = np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], 1)
    X, Y = xx / w, yy / h
    full = np.stack([np.ones_like(X), X, Y, X * X, Y * Y, X * Y], -1)
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(A, rgb[..., c][b].astype(np.float64), rcond=None)
        out[..., c] = full @ coef
    return out


def flood_from_edges(cand: np.ndarray) -> np.ndarray:
    """Pixels of `cand` connected (4-neighbour) to the image border."""
    reach = np.zeros_like(cand)
    reach[0], reach[-1], reach[:, 0], reach[:, -1] = cand[0], cand[-1], cand[:, 0], cand[:, -1]
    while True:
        grow = reach.copy()
        grow[1:] |= reach[:-1]
        grow[:-1] |= reach[1:]
        grow[:, 1:] |= reach[:, :-1]
        grow[:, :-1] |= reach[:, 1:]
        grow &= cand
        if (grow == reach).all():
            return reach
        reach = grow


def _close(mask: np.ndarray, size: int) -> np.ndarray:
    """Morphological closing (dilate, then erode) with a size x size square: bridges small gaps in the outline."""
    im = Image.fromarray((mask * 255).astype(np.uint8))
    im = im.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))
    return np.asarray(im) > 127


def cutout(img: Image.Image) -> Image.Image:
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    bg = background_model(rgb)
    lum, bg_lum = rgb.mean(-1), bg.mean(-1)
    sat = rgb.max(-1) - rgb.min(-1)
    diff_bright = (rgb - bg).max(-1)            # > 0: brighter than the background
    darker = bg_lum - lum                       # > 0: darker than the background
    greyish = sat <= SAT_MAX
    # tolerance from this image's own background noise: a clean pure-white studio (residual ~1) gets a tight
    # tolerance, so off-white paint still counts as car; a noisy grey gradient gets a looser one
    border = np.zeros(lum.shape, bool)
    border[:6], border[-6:], border[:, :6], border[:, -6:] = True, True, True, True
    tol = float(np.clip(np.percentile(np.abs(rgb - bg).max(-1)[border], 99) + 1.0, 2.5, BRIGHT_TOL))
    bg_like = greyish & (diff_bright <= tol) & (darker <= tol)
    shadow_like = greyish & (diff_bright <= tol) & (darker <= DARK_TOL)
    # floor shadow only under / beside the car: below the lowest clearly dark car part (tyres, lower grille) of
    # the neighbourhood. Slightly grey white paint higher up looks like shadow by colour, but is car.
    dark = np.asarray(Image.fromarray(((~shadow_like) * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(5))) > 127
    h, w = lum.shape
    rows = np.arange(h)[:, None]
    lowest = np.where(dark.any(0), np.where(dark, rows, -1).max(0), -1)      # per column, -1: nothing dark
    win = max(3, w // 12)
    pad = np.pad(lowest, win // 2, mode="edge")
    lowest = np.lib.stride_tricks.sliding_window_view(pad, win // 2 * 2 + 1).max(1)   # max over neighbouring columns
    shadow_zone = rows > (lowest[None, :] - SHADOW_RISE * h)
    shadow_like &= shadow_zone | bg_like
    # 1. what is clearly not background or floor shadow: outline, windows, wheels, lights, colour, white paint
    seed = ~shadow_like
    despeck = np.asarray(Image.fromarray((seed * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3))) > 127
    seed = seed & (despeck | ~bg_like)            # single-pixel noise on the background is not car
    # 2. close small gaps in the outline; everything the edge flood cannot reach is car
    closed = _close(seed, CLOSE_PX)
    region = flood_from_edges(~closed)          # background + floor shadow
    car = ~region
    background = region & bg_like
    shadow = region & ~bg_like

    alpha = np.zeros(lum.shape, np.float32)
    # the floor shadow is removed with the background (the app adds a soft CSS drop shadow instead)
    # 3. car: opaque, with a 1-2 px soft edge feathered inwards (no light halo from the old background)
    inner = np.asarray(Image.fromarray((car * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3))
                       .filter(ImageFilter.GaussianBlur(1.0)), np.float32) / 255
    alpha[car] = np.maximum(alpha[car], inner[car])
    out_rgb = rgb.copy()
    out_rgb[~car] = 0.0
    rgba = np.dstack([out_rgb, alpha * 255]).clip(0, 255).astype(np.uint8)
    res = Image.fromarray(rgba, "RGBA")

    # 4. crop to the car with a small margin
    ys, xs = np.where(alpha > 0.5)            # crop to the car itself
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    m = int(max(y1 - y0, x1 - x0) * MARGIN)
    return res.crop((max(0, x0 - m), max(0, y0 - m), min(res.width, x1 + m + 1), min(res.height, y1 + m + 1)))


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.glob("*.png")):
        out = cutout(Image.open(p))
        out.save(dst / p.name, optimize=True)
        print(f"{p.name}: {out.size[0]}x{out.size[1]}")
