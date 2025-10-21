import csv
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable, List, Literal, Sequence, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Quad:
    """Ordered TL, TR, BR, BL."""

    tl: Tuple[float, float]
    tr: Tuple[float, float]
    br: Tuple[float, float]
    bl: Tuple[float, float]


@dataclass(frozen=True)
class WarpConfig:
    """Target canvas for the perspective warp."""

    width: int
    height: int


@dataclass(frozen=True)
class BlockSpec:
    """Rectangular block and its grid."""

    x: int
    y: int
    w: int
    h: int
    rows: int
    cols: int
    pad: int = 0


@dataclass(frozen=True)
class DecodeThresholds:
    """Binary and selection thresholds."""

    binarize: int | None = None
    fill_frac: float = 0.18


@dataclass(frozen=True)
class SelectDet:
    """Which detection to use from a YOLO .txt."""

    mode: Literal["max_conf", "index"] = "max_conf"
    index: int = 0


def _order_quad(points: Iterable[Tuple[float, float]]) -> Quad:
    """Order 4 points to TL, TR, BR, BL."""
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return Quad(tuple(tl), tuple(tr), tuple(br), tuple(bl))


# --- Part 1: IO ---


def load_yolo_pose_keypoints(
    img_path: str | Path, txt_path: str | Path, select: SelectDet = SelectDet()
) -> List[Tuple[float, float]]:
    """Parse YOLO pose .txt and return pixel (x,y) keypoints."""
    img_path, txt_path = Path(img_path), Path(txt_path)
    im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    H, W = im.shape[:2]
    if not txt_path.exists():
        raise FileNotFoundError(f"Label not found: {txt_path}")

    dets: list[tuple[float, list[tuple[float, float]]]] = []
    for line in txt_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts or len(parts) < 8:
            continue
        _cls = parts[0]
        cx, cy, bw, bh = map(float, parts[1:5])

        rest_no_conf = len(parts) - 5
        rest_with_conf = len(parts) - 6
        if rest_no_conf % 3 == 0:
            obj_conf = None
            rest = list(map(float, parts[5:]))
        elif rest_with_conf % 3 == 0:
            obj_conf = float(parts[5])
            rest = list(map(float, parts[6:]))
        else:
            continue

        K = len(rest) // 3
        kps_norm: list[tuple[float, float]] = []
        kp_confs: list[float] = []
        for k in range(K):
            xi, yi, ci = rest[3 * k : 3 * k + 3]
            kps_norm.append((xi, yi))
            kp_confs.append(ci)

        kps_px = [(xi * W, yi * H) for (xi, yi) in kps_norm]
        score = (
            obj_conf
            if obj_conf is not None
            else (float(np.mean(kp_confs)) if kp_confs else 0.0)
        )
        dets.append((score, kps_px))

    if not dets:
        raise ValueError(f"No detections parsed from {txt_path}")

    if select.mode == "index":
        _, kps = dets[min(max(select.index, 0), len(dets) - 1)]
    else:
        _, kps = max(dets, key=lambda t: t[0])
    return kps


# --- Part 2 : identify the blocks ---


def warp_by_keypoints(
    img: np.ndarray, keypoints_xy: Iterable[Tuple[float, float]], cfg: WarpConfig
) -> np.ndarray:
    """Warp image to a fixed canvas using 4 extreme keypoints."""
    q = _order_quad(keypoints_xy)
    src = np.array([q.tl, q.tr, q.br, q.bl], dtype=np.float32)
    dst = np.array(
        [
            (0, 0),
            (cfg.width - 1, 0),
            (cfg.width - 1, cfg.height - 1),
            (0, cfg.height - 1),
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (cfg.width, cfg.height), flags=cv2.INTER_LINEAR)


def extract_block_cells(warped: np.ndarray, spec: BlockSpec) -> List[List[np.ndarray]]:
    """Split a rectangular block into (rows×cols) cell images."""
    x0, y0, w, h, r, c, p = (
        spec.x,
        spec.y,
        spec.w,
        spec.h,
        spec.rows,
        spec.cols,
        spec.pad,
    )
    roi = warped[max(0, y0) : y0 + h, max(0, x0) : x0 + w]
    cell_w = (roi.shape[1] - 2 * p) // c
    cell_h = (roi.shape[0] - 2 * p) // r
    cells: List[List[np.ndarray]] = []
    for i in range(r):
        row: List[np.ndarray] = []
        for j in range(c):
            xs = p + j * cell_w
            ys = p + i * cell_h
            row.append(roi[ys : ys + cell_h, xs : xs + cell_w])
        cells.append(row)
    return cells


# --- Part 3 : decode the blocks ---


def _block_binarize(roi: np.ndarray, thr: int | None) -> np.ndarray:
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    if thr is None:
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY_INV)
    bw = cv2.medianBlur(bw, 3)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return bw


def _cell_mask(h: int, w: int, shrink: float = 0.78) -> np.ndarray:
    r = int(0.5 * shrink * min(h, w))
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).astype(np.uint8)


def _scores_from_cells(
    cells: List[List[np.ndarray]], thr: DecodeThresholds
) -> np.ndarray:
    roi = np.vstack([np.hstack(row) for row in cells])
    bw = _block_binarize(roi, thr.binarize)
    h, w = cells[0][0].shape[:2]
    mask = _cell_mask(h, w)
    scores = np.zeros((len(cells), len(cells[0])), dtype=np.float32)
    for i in range(len(cells)):
        for j in range(len(cells[0])):
            y0, y1 = i * h, (i + 1) * h
            x0, x1 = j * w, (j + 1) * w
            inside = cv2.bitwise_and(bw[y0:y1, x0:x1], bw[y0:y1, x0:x1], mask=mask)
            scores[i, j] = float(cv2.countNonZero(inside)) / float(mask.sum() + 1e-6)
    return scores


def decode_student_code(
    cells: List[List[np.ndarray]], thr: DecodeThresholds
) -> List[int]:
    """10×N grid; row 0..8→1..9, row 9→0; pick strongest per column with margin."""
    rows, cols = len(cells), len(cells[0])
    assert rows == 10
    S = _scores_from_cells(cells, thr)
    out: List[int] = []
    for j in range(cols):
        col = S[:, j]
        i_best = int(np.argmax(col))
        best = float(col[i_best])
        second = float(np.partition(col, -2)[-2]) if rows > 1 else 0.0
        if best >= thr.fill_frac and best - second >= 0.08:
            out.append(0 if i_best == 9 else (i_best + 1))
        else:
            out.append(-1)
    return out


def decode_answers_mcq(
    cells: List[List[np.ndarray]], thr: DecodeThresholds
) -> List[int]:
    """Q×4 grid; pick strongest per row with margin; returns 0..3 or -1."""
    rows, cols = len(cells), len(cells[0])
    assert cols == 4
    S = _scores_from_cells(cells, thr)
    ans: List[int] = []
    for i in range(rows):
        row = S[i]
        j_best = int(np.argmax(row))
        best = float(row[j_best])
        second = float(np.partition(row, -2)[-2]) if cols > 1 else 0.0
        ans.append(j_best if best >= thr.fill_frac and best - second >= 0.08 else -1)
    return ans


def print_answers(
    indices: Sequence[int],
    labels: Tuple[str, str, str, str] = ("a", "b", "c", "d"),
    missing: str = "-",
    cols: int = 2,
) -> str:
    """Return a two-column text table like '1: a)  |  6: c)'."""

    def lab(i: int) -> str:
        return labels[i] if 0 <= i < len(labels) else missing

    n = len(indices)
    rows = ceil(n / cols)
    cells: List[str] = [f"{k+1:2d}: {lab(idx)})" for k, idx in enumerate(indices)]
    col_chunks = [
        cells[r * 1 + c * rows : r * 1 + c * rows + rows]
        for c in range(cols)
        for r in (0,)
    ]
    for chunk in col_chunks:
        while len(chunk) < rows:
            chunk.append("")
    widths = [max((len(s) for s in chunk), default=0) for chunk in col_chunks]
    lines = []
    for r in range(rows):
        parts = [(col_chunks[c][r].ljust(widths[c])) for c in range(cols)]
        lines.append("  |  ".join(parts).rstrip())
    return "\n".join(lines)


# --- plots and reports ---


def plot_results(
    orig_bgr,
    warped_bgr,
    keypoints,
    id_cells,
    resp_left_cells,
    resp_right_cells,
    id_block,
    resp_left_block,
    resp_right_block,
    figsize=(20, 6),
    border_color=(255, 0, 0),
    border_thickness=6,
    savefig=None,
    # --- page overlays ---
    show_grids: bool = False,
    show_painted: bool = False,
    thr: "DecodeThresholds|None" = None,
    grid_color: tuple[int, int, int] = (0, 255, 255),
    grid_thickness: int = 2,
    painted_color: tuple[int, int, int] = (0, 0, 255),
    painted_alpha: float = 0.35,
    mask_shrink: float = 0.78,
    # --- zoomed panels overlays ---
    zoom_show_painted: bool = False,
    zoom_tile: int = 40,
):
    """Show side-by-side debug plot; optionally overlay per-cell grid on page and painted masks on page and zoomed cells."""

    # ---------- small utils ----------
    def as_bgr(img):
        return (
            img
            if (img.ndim == 3 and img.shape[2] == 3)
            else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        )

    def draw_kps(img, kps):
        vis = img.copy()
        for x, y in kps:
            cv2.circle(vis, (int(round(x)), int(round(y))), 5, (0, 255, 0), 8)
        return vis

    def add_border(img):
        return cv2.copyMakeBorder(
            img,
            border_thickness,
            border_thickness,
            border_thickness,
            border_thickness,
            cv2.BORDER_CONSTANT,
            value=border_color,
        )

    def draw_block_rects(img, blocks, color=(0, 0, 255), thickness=4):
        vis = img.copy()
        for spec in blocks:
            x0, y0, w, h = spec.x, spec.y, spec.w, spec.h
            cv2.rectangle(vis, (x0, y0), (x0 + w, y0 + h), color, thickness)
        return vis

    # ---------- page overlay helpers ----------
    def _iter_cell_slices(spec: BlockSpec, roi_w: int, roi_h: int):
        _, _, _, _, r, c, p = (
            spec.x,
            spec.y,
            spec.w,
            spec.h,
            spec.rows,
            spec.cols,
            spec.pad,
        )
        inner_w = roi_w - 2 * p
        inner_h = roi_h - 2 * p
        cell_w = max(1, inner_w // c)
        cell_h = max(1, inner_h // r)
        for i in range(r):
            ys = p + i * cell_h
            ye = p + (i + 1) * cell_h if i < r - 1 else roi_h - p
            for j in range(c):
                xs = p + j * cell_w
                xe = p + (j + 1) * cell_w if j < c - 1 else roi_w - p
                ys_c = max(0, min(ys, roi_h))
                ye_c = max(ys_c, min(ye, roi_h))
                xs_c = max(0, min(xs, roi_w))
                xe_c = max(xs_c, min(xe, roi_w))
                yield i, j, xs_c, xe_c, ys_c, ye_c

    def _grid_overlay_inplace(vis, spec: BlockSpec):
        x0, y0, w, h, r, c, p = (
            spec.x,
            spec.y,
            spec.w,
            spec.h,
            spec.rows,
            spec.cols,
            spec.pad,
        )
        cell_w = (w - 2 * p) // c
        cell_h = (h - 2 * p) // r
        for i in range(1, r):
            y = y0 + p + i * cell_h
            cv2.line(vis, (x0 + p, y), (x0 + w - p - 1, y), grid_color, grid_thickness)
        for j in range(1, c):
            x = x0 + p + j * cell_w
            cv2.line(vis, (x, y0 + p), (x, y0 + h - p - 1), grid_color, grid_thickness)
        for i in range(r):
            for j in range(c):
                cx = x0 + p + j * cell_w + cell_w // 2
                cy = y0 + p + i * cell_h + cell_h // 2
                cv2.circle(vis, (cx, cy), 2, grid_color, -1)

    def _painted_overlay_inplace(vis, spec: BlockSpec):
        if thr is None:
            return
        x0, y0, w, h = spec.x, spec.y, spec.w, spec.h
        roi = vis[y0 : y0 + h, x0 : x0 + w]
        bw = _block_binarize(roi, thr.binarize)  # 255 where ink after INV
        roi_h, roi_w = bw.shape[:2]
        mask_full = np.zeros((roi_h, roi_w), dtype=np.uint8)
        for _, _, xs, xe, ys, ye in _iter_cell_slices(spec, roi_w, roi_h):
            ch, cw = ye - ys, xe - xs
            if ch <= 0 or cw <= 0:
                continue
            cell_mask = _cell_mask(ch, cw, shrink=mask_shrink) * 255
            cur = mask_full[ys:ye, xs:xe]
            mask_full[ys:ye, xs:xe] = np.maximum(cur, cell_mask)
        painted = cv2.bitwise_and(bw, mask_full)
        paint_rgb = np.zeros_like(roi)
        paint_rgb[:] = painted_color
        mix = cv2.addWeighted(roi, 1.0, paint_rgb, painted_alpha, 0.0)
        roi[painted > 0] = mix[painted > 0]

    def cells_to_grid_debug(cells, do_painted=False, tile=40):
        if not cells:
            return np.zeros((tile, tile, 3), dtype=np.uint8)
        rows_tiles = []
        for row in cells:
            row_tiles = []
            for cell in row:
                cell_bgr = as_bgr(cell)
                if do_painted and thr is not None:
                    # binarize + mask per cell (local coords)
                    gmask = _block_binarize(cell_bgr, thr.binarize)  # on the small cell
                    ch, cw = gmask.shape[:2]
                    cmask = _cell_mask(ch, cw, shrink=mask_shrink) * 255
                    painted = cv2.bitwise_and(gmask, cmask)
                    overlay = cell_bgr.copy()
                    color = np.zeros_like(cell_bgr)
                    color[:] = painted_color
                    mixed = cv2.addWeighted(overlay, 1.0, color, painted_alpha, 0.0)
                    cell_bgr[painted > 0] = mixed[painted > 0]
                    rad = int(0.5 * mask_shrink * min(ch, cw))
                    cv2.circle(cell_bgr, (cw // 2, ch // 2), rad, grid_color, 1)
                row_tiles.append(cv2.resize(cell_bgr, (tile, tile)))
            rows_tiles.append(np.hstack(row_tiles))
        return np.vstack(rows_tiles)

    # ---------- assemble panels ----------
    q1 = add_border(draw_kps(as_bgr(orig_bgr), keypoints))

    warped_vis = as_bgr(warped_bgr).copy()
    warped_vis = draw_block_rects(
        warped_vis, [id_block, resp_left_block, resp_right_block]
    )
    if show_grids:
        for b in (id_block, resp_left_block, resp_right_block):
            _grid_overlay_inplace(warped_vis, b)
    if show_painted:
        for b in (id_block, resp_left_block, resp_right_block):
            _painted_overlay_inplace(warped_vis, b)
    q2 = add_border(warped_vis)

    q3 = add_border(
        cells_to_grid_debug(id_cells, do_painted=zoom_show_painted, tile=zoom_tile)
    )
    q4 = add_border(
        cells_to_grid_debug(
            resp_left_cells, do_painted=zoom_show_painted, tile=zoom_tile
        )
    )
    q5 = add_border(
        cells_to_grid_debug(
            resp_right_cells, do_painted=zoom_show_painted, tile=zoom_tile
        )
    )

    imgs = [q1, q2, q3, q4, q5]
    H = max(im.shape[0] for im in imgs)
    imgs = [
        (
            cv2.resize(im, (int(im.shape[1] * H / im.shape[0]), H))
            if im.shape[0] != H
            else im
        )
        for im in imgs
    ]
    out = np.hstack(imgs)

    fig = plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    if savefig is not None:
        Path(savefig).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(savefig), bbox_inches="tight", dpi=200)
    else:
        plt.show()
    plt.close(fig)


def iter_labeled_images(
    img_dir: Path,
    lbl_dir: Path,
    exts={".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"},
):
    """Yield (img_path, txt_path) only when the label exists."""
    for img_path in img_dir.rglob("*"):
        if not (img_path.is_file() and img_path.suffix.lower() in exts):
            continue
        txt_path = (lbl_dir / img_path.stem).with_suffix(".txt")
        if txt_path.exists():
            yield img_path, txt_path


# --- CSV save ---


@dataclass(frozen=True)
class CsvRow:
    """Single report row."""

    student_id: str
    answers: str
    img_path: str
    report_path: str


def encode_csv_list(ans: Sequence[int], valid_max: int = 3) -> str:
    """Join answers as comma-separated ints; invalids become -1."""

    def norm(x: int) -> int:
        return x if isinstance(x, int) and -1 <= x <= valid_max else -1

    return ",".join(str(norm(x)) for x in ans)


def append_rows_csv(csv_path: Path, rows: Iterable[CsvRow]) -> None:
    """Append rows; create with header if missing."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["student_id", "answers", "img_path", "report_path"])
        for r in rows:
            w.writerow([r.student_id, r.answers, r.img_path, r.report_path])


# --- MAIN LOOP ---


def process_dataset(
    root: str,
    report_dir: str,
    model_name: str,
    warp: WarpConfig,
    id_block: BlockSpec,
    resp_left_block: BlockSpec,
    resp_right_block: BlockSpec,
    thr: DecodeThresholds,
):
    matplotlib.use("Agg")

    root = Path(root)
    report_dir = Path(report_dir)

    img_dir = root / "images"
    lbl_dir = root / "labels"
    csv_out = root / "result.csv"

    report_dir.mkdir(parents=True, exist_ok=True)

    rows_buf: List[CsvRow] = []
    CHUNK_SIZE = 64
    processed = 0
    errors = 0

    for i, (img_path, txt_path) in enumerate(
        iter_labeled_images(img_dir, lbl_dir), start=1
    ):
        img = warped = None
        id_cells = respL_cells = respR_cells = None
        kps = None
        try:
            kps = load_yolo_pose_keypoints(str(img_path), str(txt_path))
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"cv2.imread failed: {img_path}")

            warped = warp_by_keypoints(img, kps, warp)
            id_cells = extract_block_cells(warped, id_block)
            respL_cells = extract_block_cells(warped, resp_left_block)
            respR_cells = extract_block_cells(warped, resp_right_block)

            student_id_ls = decode_student_code(id_cells, thr)
            answers_int = decode_answers_mcq(respL_cells, thr) + decode_answers_mcq(
                respR_cells, thr
            )

            student_id_csv = encode_csv_list(student_id_ls, valid_max=9)
            answers_csv = encode_csv_list(answers_int, valid_max=3)

            savefig = report_dir / img_path.name
            try:
                with matplotlib.pyplot.ioff():
                    plot_results(
                        img,
                        warped,
                        kps,
                        id_cells,
                        respL_cells,
                        respR_cells,
                        id_block,
                        resp_left_block,
                        resp_right_block,
                        figsize=(24, 8),
                        border_thickness=8,
                        savefig=savefig,
                    )
            finally:
                matplotlib.pyplot.close("all")

            rows_buf.append(
                CsvRow(
                    student_id=student_id_csv,
                    answers=answers_csv,
                    img_path=str(img_path),
                    report_path=str(savefig),
                )
            )

            processed += 1

            if len(rows_buf) >= CHUNK_SIZE:
                append_rows_csv(csv_out, rows_buf)
                rows_buf.clear()
                print(f"[FLUSH] wrote {processed} rows to {csv_out}")

        except Exception as e:
            errors += 1
            print(f"[ERROR] {img_path.name}: {e}")

    if rows_buf:
        append_rows_csv(csv_out, rows_buf)
        print(f"[FLUSH] final batch: {len(rows_buf)} rows")
        rows_buf.clear()

    print(f"[DONE] processed={processed}, errors={errors}, csv={csv_out}")


# --- INSPECT RESULT ---


def inspect_results(
    root: str,
    results_dir: str,
    warp: WarpConfig,
    id_block: BlockSpec,
    resp_left_block: BlockSpec,
    resp_right_block: BlockSpec,
    thr: DecodeThresholds,
    limit: int = 10,
):
    root = Path(root)
    results_dir = Path(results_dir)

    img_dir = root / "images"
    lbl_dir = root / "labels"

    results_dir.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_done = 0

    for i, (img_path, txt_path) in enumerate(iter_labeled_images(img_dir, lbl_dir)):
        n_total += 1
        savefig = results_dir / img_path.name

        kps = load_yolo_pose_keypoints(img_path, txt_path)

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] could not read image: {img_path}")
            continue

        warped = warp_by_keypoints(img, kps, warp)

        id_cells = extract_block_cells(warped, id_block)
        respL_cells = extract_block_cells(warped, resp_left_block)
        respR_cells = extract_block_cells(warped, resp_right_block)

        student_id = decode_student_code(id_cells, thr)
        answers = decode_answers_mcq(respL_cells, thr) + decode_answers_mcq(
            respR_cells, thr
        )

        print(f"student_id: {student_id}\n")
        print(print_answers(answers))

        plot_results(
            img,
            warped,
            kps,
            id_cells,
            respL_cells,
            respR_cells,
            id_block,
            resp_left_block,
            resp_right_block,
            figsize=(24, 8),
            border_thickness=8,
            show_grids=True,
            show_painted=True,
            thr=thr,
            mask_shrink=0.85,
            painted_alpha=0.40,
            zoom_show_painted=True,
        )

        print("-" * 50, "\n" * 3)
        n_done += 1

        if i > limit:
            break

    print(f"[POST] processed {n_done} / {n_total} labeled pages from {img_dir}")
