import io
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# --- Part 1 : to jpg


def safe_name(s: str) -> str:
    """Keep only [A-Za-z0-9._-]; drop everything else."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return s or "file"


def triage_to_jpg(
    src_dir: str | Path, dst_dir: str | Path, size: Tuple[int, int]
) -> None:
    """Rasterize images/PDFs from src_dir into dst_dir as same-sized JPEGs."""
    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}

    def save(im: Image.Image, out: Path) -> None:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA"):  # remove alpha over white
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im.convert("RGBA"), mask=im.getchannel("A"))
            im = bg
        else:
            im = im.convert("RGB")
        im = ImageOps.pad(im, size, method=Image.BICUBIC, color=(0, 0, 0))
        im.save(out, format="JPEG", quality=90, optimize=True)

    for p in sorted(src.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        stem = safe_name(p.stem)
        if p.suffix.lower() == ".pdf":
            if fitz is None:
                raise RuntimeError("PyMuPDF (fitz) is required to rasterize PDFs.")
            doc = fitz.open(p)
            try:
                for i, page in enumerate(doc):
                    mat = fitz.Matrix(200 / 72, 200 / 72)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    with Image.open(io.BytesIO(pix.tobytes("png"))) as im:
                        name = (
                            f"{stem}_p{i+1}.jpg"
                            if doc.page_count > 1
                            else f"{stem}.jpg"
                        )
                        save(im, dst / name)
            finally:
                doc.close()
        else:
            with Image.open(p) as im:
                save(im, dst / f"{stem}.jpg")


# --- Part 2 : classify ---


def _black_ratio_on_canvas(
    img: Image.Image, eval_size: Tuple[int, int], border: int = 0
) -> float:
    """Letterbox to eval_size on black, then return fraction of pure-black pixels."""
    canvas = ImageOps.pad(
        img.convert("RGB"), eval_size, method=Image.BICUBIC, color=(0, 0, 0)
    )
    if border > 0:
        w, h = canvas.size
        canvas = canvas.crop((border, border, w - border, h - border))
    a = np.asarray(canvas, dtype=np.uint8)
    return float(np.mean((a[..., 0] == 0) & (a[..., 1] == 0) & (a[..., 2] == 0)))


def scan_black_ratios_ar(
    src_jpg_dir: str | Path, eval_size: Tuple[int, int], border: int = 0
) -> List[tuple[str, float]]:
    """[(filename, ratio)] using a fixed evaluation aspect ratio (does not save images)."""
    src = Path(src_jpg_dir)
    out: List[tuple[str, float]] = []
    for p in sorted(src.glob("*.jpg")):
        with Image.open(p) as im:
            r = _black_ratio_on_canvas(im, eval_size, border)
        out.append((p.name, r))
    return out


def classify_by_black(
    src_jpg_dir: str | Path,
    out_root: str | Path,
    thresholds: List[float],
    labels: List[str],
    eval_size: Tuple[int, int],
    border: int = 0,
) -> None:
    """
    Bin rule (ascending): r <= t0 -> labels[0]; t0 < r <= t1 -> labels[1]; ...; r > t_last -> labels[-1].
    Evaluation is done on a black-letterboxed canvas of eval_size, but originals are copied.
    """
    assert len(thresholds) + 1 == len(labels)
    thresholds = sorted(thresholds)

    src, out = Path(src_jpg_dir), Path(out_root)
    for lab in labels:
        (out / lab).mkdir(parents=True, exist_ok=True)

    for p in sorted(src.glob("*.jpg")):
        with Image.open(p) as im:
            r = _black_ratio_on_canvas(im, eval_size, border)
        idx = int(np.searchsorted(np.asarray(thresholds, float), r, side="right"))
        shutil.copy2(p, out / labels[idx] / p.name)


# --- Part 3 : crop ---


def _zoom_and_resize(
    im: Image.Image, out_size: Tuple[int, int], tol: int = 3, margin_frac: float = 0.01
) -> Image.Image:
    """Crop to non-black content and letterbox to out_size."""
    im = ImageOps.exif_transpose(im)
    a = np.asarray(im.convert("RGB"), dtype=np.uint8)
    mask = (a[..., 0] > tol) | (a[..., 1] > tol) | (a[..., 2] > tol)
    if mask.any():
        ys, xs = np.where(mask)
        L, T, R, B = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        m = int(min(im.size) * margin_frac)
        L, T = max(L - m, 0), max(T - m, 0)
        R, B = min(R + m, im.size[0] - 1), min(B + m, im.size[1] - 1)
        im = im.crop((L, T, R + 1, B + 1))
    W, H = out_size
    iw, ih = im.size
    s = min(W / iw, H / ih)
    nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
    imr = im.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(imr, ((W - nw) // 2, (H - nh) // 2))

    pad_frac = 1.0 - (nw * nh) / float(W * H)
    return canvas, pad_frac


def crop_classified(
    src_root: str | Path,
    out_dir: str | Path,
    out_size: Tuple[int, int] = (1024, 1448),
    tol: int = 3,
    center_gutter_frac: float = 0.15,
    max_pad_frac: float = 0.15,
) -> None:
    """Classes: single → one output; side_by_side → split vertical; top_bottom → split horizontal."""
    src = Path(src_root)
    out_main = Path(out_dir)
    out_extra = out_main / "extras"
    out_errors = out_main / "errors"
    out_main.mkdir(parents=True, exist_ok=True)
    out_extra.mkdir(parents=True, exist_ok=True)
    out_errors.mkdir(parents=True, exist_ok=True)

    def _save(img: Image.Image, dst: Path, stem: str, suffix: str = "") -> None:
        name = f"{stem}{suffix}.jpg"
        img.save(dst / name, "JPEG", quality=95)

    for cls in ("single", "side_by_side", "top_bottom"):
        for p in sorted((src / cls).glob("*.jpg")):
            stem = p.stem
            with Image.open(p) as im:
                W, H = im.size
                if cls == "single":
                    img, pad = _zoom_and_resize(im, out_size, tol=tol)
                    dst = out_main if pad <= max_pad_frac else out_errors
                    _save(img, out_main, stem)

                elif cls == "side_by_side":
                    mid = W // 2
                    left = im.crop((0, 0, mid, H))
                    right = im.crop((mid, 0, W, H))

                    img1, pad1 = _zoom_and_resize(left, out_size, tol=tol)
                    img2, pad2 = _zoom_and_resize(right, out_size, tol=tol)

                    dst1 = out_main if pad1 <= max_pad_frac else out_errors
                    dst2 = out_extra if pad2 <= max_pad_frac else out_errors

                    _save(img1, out_main, stem, "_p1")
                    _save(img2, out_extra, stem, "_p2")
                else:  # top_bottom
                    mid = H // 2
                    g = max(0, int(round(H * center_gutter_frac / 2)))
                    T1, B1 = 0, max(1, mid - g)
                    T2, B2 = min(H - 1, mid + g), H
                    top = im.crop((0, T1, W, B1))
                    bottom = im.crop((0, T2, W, B2))
                    # top = im.crop((0, 0, W, mid))
                    # bottom = im.crop((0, mid, W, H))

                    img1, pad1 = _zoom_and_resize(top, out_size, tol=tol)
                    img2, pad2 = _zoom_and_resize(bottom, out_size, tol=tol)

                    dst1 = out_main if pad1 <= max_pad_frac else out_errors
                    dst2 = out_extra if pad2 <= max_pad_frac else out_errors

                    _save(img1, dst1, stem, "_p1")
                    _save(img2, dst2, stem, "_p2")


# --- Part 4 : Find first page


EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

RMIN_FRAC: float = 0.002
RMAX_FRAC: float = 0.035
MIN_BUBBLES_FIRST: int = 10


def _read_gray(p: str | Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {p}")
    return img


def _enhance_filled(gray: np.ndarray) -> np.ndarray:
    """Emphasize dark filled disks; keep it minimal."""
    h = gray.shape[0]
    k = max(3, int(0.018 * h)) | 1
    g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    bh = cv2.morphologyEx(
        g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    return cv2.GaussianBlur(bh, (5, 5), 0)


def _circle_annulus_contrast(
    gray: np.ndarray, x: int, y: int, r: int
) -> tuple[float, float]:
    """Return (contrast, fill_frac) using inner disk vs. outer ring."""
    H, W = gray.shape
    yy, xx = np.ogrid[:H, :W]
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    inner = d2 <= (0.95 * r) ** 2
    ring = (d2 >= (1.25 * r) ** 2) & (d2 <= (1.65 * r) ** 2)

    if not inner.any() or not ring.any():
        return 0.0, 0.0
    inner_vals = gray[inner].astype(np.float32)
    ring_vals = gray[ring].astype(np.float32)
    contrast = float(ring_vals.mean() - inner_vals.mean())
    thresh = ring_vals.mean() - 0.4 * max(10.0, contrast)
    fill_frac = float((inner_vals < thresh).mean())
    return contrast, fill_frac


def detect_filled_bubbles(
    gray: np.ndarray,
) -> tuple[np.ndarray, List[tuple[int, int, int]]]:
    """Kernel + Hough, pruned by annulus contrast and size/spacing sanity."""
    H, W = gray.shape
    rmin, rmax = int(RMIN_FRAC * H), int(RMAX_FRAC * H)
    enh = _enhance_filled(gray)
    edges = cv2.Canny(enh, 50, 140)

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(6, rmin),
        param1=160,
        param2=18,
        minRadius=rmin,
        maxRadius=rmax,
    )
    if circles is None:
        return np.zeros_like(gray), []

    raw = np.round(circles[0]).astype(int).tolist()
    rs = np.array([r for _, _, r in raw])
    r_med = float(np.median(rs))
    raw = [(x, y, r) for (x, y, r) in raw if 0.7 * r_med <= r <= 1.3 * r_med]

    min_dist = max(int(1.2 * r_med), rmin)
    circles2 = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=160,
        param2=20,
        minRadius=int(0.8 * r_med),
        maxRadius=int(1.2 * r_med),
    )
    cand = raw if circles2 is None else np.round(circles2[0]).astype(int).tolist()

    kept: List[tuple[int, int, int]] = []
    for x, y, r in cand:
        if x - r < 0 or y - r < 0 or x + r >= W or y + r >= H:
            continue
        contrast, fill_frac = _circle_annulus_contrast(gray, x, y, r)
        if contrast >= 18.0 and fill_frac >= 0.45:
            kept.append((x, y, r))

    mask = np.zeros_like(gray, dtype=np.uint8)
    for x, y, r in kept:
        cv2.circle(mask, (x, y), r, 255, -1)
    return mask, kept


def is_first_page(gray: np.ndarray) -> bool:
    _, kept = detect_filled_bubbles(gray)
    return len(kept) > MIN_BUBBLES_FIRST


def _iter_images(src: Path) -> Iterable[Path]:
    for p in src.iterdir():
        if p.is_file() and p.suffix.lower() in EXTS:
            yield p


def classify_dir(
    src_dir: str | Path, out_root: str | Path, save_masks: bool = False
) -> None:
    src, out = Path(src_dir), Path(out_root)
    (out).mkdir(parents=True, exist_ok=True)
    (out / "extras").mkdir(parents=True, exist_ok=True)
    (out / "errors").mkdir(parents=True, exist_ok=True)

    shutil.copytree(src / "extras", out / "extras", dirs_exist_ok=True)
    shutil.copytree(src / "errors", out / "errors", dirs_exist_ok=True)

    for img_path in _iter_images(src):
        gray = _read_gray(img_path)
        mask, kept = detect_filled_bubbles(gray)
        dst = (out if len(kept) > MIN_BUBBLES_FIRST else out / "extras") / img_path.name
        cv2.imwrite(str(dst), cv2.imread(str(img_path)))
        if save_masks:
            cv2.imwrite(str(dst.with_stem(dst.stem + "_mask")), mask)


def debug_show(path: str | Path) -> None:
    gray = _read_gray(path)
    mask, kept = detect_filled_bubbles(gray)
    print(
        ("IS FIRST PAGE" if len(kept) > MIN_BUBBLES_FIRST else "NOT FIRST PAGE")
        + f" | bubbles={len(kept)}"
    )
    plt.imshow(np.concatenate([gray, mask], axis=1), cmap="gray")
    plt.axis("off")
    plt.show()

