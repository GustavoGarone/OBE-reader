from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
from ultralytics import YOLO
import json, yaml, shutil
import numpy as np, cv2


# --- Crop handler classes ---


@dataclass(frozen=True)
class CropCfg:
    """Crop size and bbox pad."""
    factor: float = 0.28
    bbox_pad: float = 0.15
    out_ext: str = ".jpg"

@dataclass(frozen=True)
class ImgAnn:
    """Image + one YOLO row."""
    img_path: Path
    image: np.ndarray
    row: List[float]

@dataclass(frozen=True)
class CropMeta:
    """Sidecar to invert crop→page."""
    page_rel: str
    page_h: int
    page_w: int
    corner_id: int # 0:TL, 1:TR, 2:BL, 3:BR  (TL,TR,BL,BR)
    x0: int
    y0: int
    cw: int
    ch: int 
    crop_name: str


class CornerCropper:
    """Page↔crops transform: 4-kpt page <-> four 1-kpt TL-canonical crops via flips."""
    def __init__(self, cfg: CropCfg):
        self.cfg = cfg

    def _read_row(self, p: Path) -> List[float] | None:
        if not p.exists(): return None
        lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        if not lines: return None
        return [float(x) for x in lines[0].split()]

    def _write_row(self, p: Path, row: List[float]) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(" ".join(f"{int(x)}" if i==0 else f"{x:.6f}" for i,x in enumerate(row)) + "\n")

    def _save_meta(self, meta: CropMeta, lbl_dir: Path) -> None:
        m = (lbl_dir / meta.crop_name).with_suffix(".meta.json")
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text(json.dumps(meta.__dict__, ensure_ascii=False))

    def _load_meta(self, p: Path) -> CropMeta:
        return CropMeta(**json.loads(p.read_text()))

    def _denorm(self, xy: np.ndarray, W: int, H: int) -> np.ndarray:
        xy = np.asarray(xy, float)
        return np.stack([xy[...,0]*W, xy[...,1]*H], -1)

    def _norm(self, xy: np.ndarray, W: int, H: int) -> np.ndarray:
        xy = np.asarray(xy, float)
        return np.stack([xy[...,0]/W, xy[...,1]/H], -1)

    def _flip_xy_to_tl(self, nx: float, ny: float, cid: int) -> Tuple[float,float]:
        if cid == 0: return nx, ny
        if cid == 1: return 1.0 - nx, ny
        if cid == 2: return nx, 1.0 - ny
        if cid == 3: return 1.0 - nx, 1.0 - ny
        raise ValueError

    def _flip_img_to_tl(self, img: np.ndarray, cid: int) -> np.ndarray:
        if cid == 0: return img
        if cid == 1: return cv2.flip(img, 1)   # H
        if cid == 2: return cv2.flip(img, 0)   # V
        if cid == 3: return cv2.flip(img, -1)  # H+V
        raise ValueError

    def _unflip_xy_from_tl(self, nx: float, ny: float, cid: int) -> Tuple[float,float]:
        return self._flip_xy_to_tl(nx, ny, cid)


    def split_page(self, page: ImgAnn) -> Tuple[List[np.ndarray], List[List[float]], List[CropMeta]]:
        """Page (TL,TR,BL,BR) -> 4 TL-canonical crops + 1-kpt labels + metas."""
        H, W = page.image.shape[:2]
        k = np.array(page.row[5:], float).reshape(-1,3)[:,:2]  # TL,TR,BL,BR
        kpxy = self._denorm(k, W, H)
        side = int(round(self.cfg.factor * min(H, W)))
        crops, rows, metas = [], [], []
        for cid, (x, y) in enumerate(kpxy):
            x0 = max(0, int(round(x - side/2))); y0 = max(0, int(round(y - side/2)))
            x1 = min(W, x0 + side); y1 = min(H, y0 + side)
            x0 = max(0, x1 - side); y0 = max(0, y1 - side)
            cw, ch = x1 - x0, y1 - y0
            crop = page.image[y0:y1, x0:x1].copy()
            if crop.size == 0: continue
            nx, ny = (x - x0)/cw, (y - y0)/ch
            nx = float(np.clip(nx, 0, 1)); ny = float(np.clip(ny, 0, 1))

            crop = self._flip_img_to_tl(crop, cid)
            nx, ny = self._flip_xy_to_tl(nx, ny, cid)

            pad = self.cfg.bbox_pad
            bx1, by1 = max(0.0, nx - pad), max(0.0, ny - pad)
            bx2, by2 = min(1.0, nx + pad), min(1.0, ny + pad)
            row = [0, (bx1+bx2)/2, (by1+by2)/2, (bx2-bx1), (by2-by1), nx, ny, 2.0]

            crops.append(crop)
            rows.append(row)
            metas.append(CropMeta(
                page_rel=str(page.img_path), page_h=H, page_w=W, corner_id=cid,
                x0=x0, y0=y0, cw=cw, ch=ch, crop_name=f"{page.img_path.stem}_kp{cid}"
            ))
        return crops, rows, metas


    def merge_page(self, meta_list: List[CropMeta], crop_rows: List[List[float]]) -> List[float]:
        """4 metas + 4 crop rows (TL-canonical) -> page row with 4 kpts (TL,TR,BL,BR)."""
        assert len(meta_list) == 4 and len(crop_rows) == 4
        pts = []
        for m, r in zip(meta_list, crop_rows):
            nx, ny = r[5], r[6]
            nx, ny = self._unflip_xy_from_tl(nx, ny, m.corner_id)
            x = m.x0 + nx * m.cw
            y = m.y0 + ny * m.ch
            pts.append([x, y])
        pts = np.array(pts, float)  # TL,TR,BL,BR order
        W, H = meta_list[0].page_w, meta_list[0].page_h
        k = self._norm(pts, W, H)
        pad = 0.001
        x1, y1 = np.clip(k.min(0) - pad, 0, 1); x2, y2 = np.clip(k.max(0) + pad, 0, 1)
        xc, yc = (x1+x2)/2, (y1+y2)/2; bw, bh = (x2-x1), (y2-y1)
        klist = []
        for x, y in k: klist += [float(x), float(y), 2.0]
        return [0, float(xc), float(yc), float(bw), float(bh), *klist]



# --- Build training dataset --- 


def build_crops(data_yaml: Path, out_root: Path, cfg: CropCfg = CropCfg()) -> None:
    d = yaml.safe_load(data_yaml.read_text()); base = data_yaml.parent
    out_img_tr = out_root/"images"/"train"; out_img_va = out_root/"images"/"val"
    out_lbl_tr = out_root/"labels"/"train"; out_lbl_va = out_root/"labels"/"val"
    for p in [out_img_tr, out_img_va, out_lbl_tr, out_lbl_va]: p.mkdir(parents=True, exist_ok=True)

    cropper = CornerCropper(cfg)

    for split in ("train","val"):
        idir = Path(d[split]) if Path(d[split]).is_absolute() else base/ d[split]
        ldir = Path(str(idir).replace("/images/","/labels/").replace("\\images\\","\\labels\\"))
        oimg = out_img_tr if split=="train" else out_img_va
        olbl = out_lbl_tr if split=="train" else out_lbl_va

        for ip in sorted(idir.rglob("*")):
            if ip.suffix.lower() not in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}: continue
            rel = ip.relative_to(idir)
            lp  = (ldir / rel).with_suffix(".txt")
            row = cropper._read_row(lp)
            if row is None: continue
            bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
            if bgr is None: continue

            crops, rows, metas = cropper.split_page(ImgAnn(ip, bgr, row))
            for crop, crow, meta in zip(crops, rows, metas):
                op = (oimg / rel.parent / f"{meta.crop_name}{cfg.out_ext}")
                ol = (olbl / rel.parent / f"{meta.crop_name}.txt")
                op.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(op), crop)
                cropper._write_row(ol, crow)
                cropper._save_meta(meta, olbl / rel.parent)

    yd = {"path": str(out_root), "train": "images/train", "val": "images/val",
          "names": ["corner"], "kpt_shape": [1,3], "flip_idx": [0]}
    (out_root/"data.yaml").write_text(yaml.safe_dump(yd, sort_keys=False))

# USED AS
# build_crops(
#     data_yaml=Path("data/02--labeled/yolo/data.yaml"),
#     out_root=Path("data/02--labeled/yolo_crops"),
#     cfg=CropCfg(factor=0.28, bbox_pad=0.15)
# )


# --- Inference ---


def _ensure_dirs(out_root: Path, tmp_root: Path) -> Dict[str, Path]:
    out_img = out_root / "images"
    out_lbl = out_root / "labels"
    out_unl = out_root / "unlabeled"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    out_unl.mkdir(parents=True, exist_ok=True)
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    (tmp_root / "images").mkdir(parents=True, exist_ok=True)
    (tmp_root / "labels").mkdir(parents=True, exist_ok=True)
    return {"out_img": out_img, "out_lbl": out_lbl, "out_unl": out_unl}

def _list_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)

def crop_images(images: List[Path], cropper: CornerCropper, tmp_root: Path, factor: float) -> List[CropMeta]:
    metas: List[CropMeta] = []
    for ip in images:
        bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if bgr is None: continue
        H, W = bgr.shape[:2]
        side = int(round(factor * min(H, W)))
        corners = {0: (0, 0), 1: (W - side, 0), 2: (0, H - side), 3: (W - side, H - side)}
        for cid, (x0, y0) in corners.items():
            x0 = int(np.clip(x0, 0, max(0, W - 1)))
            y0 = int(np.clip(y0, 0, max(0, H - 1)))
            x1 = int(np.clip(x0 + side, 0, W))
            y1 = int(np.clip(y0 + side, 0, H))
            x0 = max(0, x1 - side)
            y0 = max(0, y1 - side)
            crop = bgr[y0:y1, x0:x1].copy()
            if crop.size == 0: continue
            crop = cropper._flip_img_to_tl(crop, cid)
            stem = f"{ip.stem}_kp{cid}"
            cv2.imwrite(str(tmp_root / "images" / f"{stem}.jpg"), crop)
            meta = CropMeta(
                page_rel=str(ip), page_h=H, page_w=W, corner_id=cid,
                x0=int(x0), y0=int(y0), cw=int(x1 - x0), ch=int(y1 - y0),
                crop_name=stem
            )
            (tmp_root / "labels" / f"{stem}.meta.json").write_text(json.dumps(meta.__dict__, ensure_ascii=False))
            metas.append(meta)
    return metas

def predict_yolo(model_path: str, tmp_root: Path, conf: float, imgsz: int) -> Path | None:
    model = YOLO(model_path)
    preds = model.predict(
        source=str(tmp_root / "images"),
        conf=conf,
        imgsz=imgsz,
        save=False,
        save_txt=True,
        verbose=False,
        max_det=1,
        stream=True,
    )
    save_dir = None
    for r in preds:
        save_dir = r.save_dir
    if save_dir is None:
        return None
    return Path(save_dir) / "labels"

def _row_to_page_xy(meta: CropMeta, row: List[float], cropper: CornerCropper) -> Tuple[float, float]:
    nx, ny = float(row[5]), float(row[6])
    nx, ny = cropper._unflip_xy_from_tl(nx, ny, meta.corner_id)
    px = meta.x0 + nx * meta.cw
    py = meta.y0 + ny * meta.ch
    return float(px), float(py)

def _infer_fourth(pts4: List[Tuple[float,float] | None]) -> Tuple[List[Tuple[float,float] | None], int | None]:
    P = pts4[:]
    miss = [i for i,p in enumerate(P) if p is None]
    if len(miss) != 1:
        return P, None
    m = miss[0]
    TL, TR, BL, BR = P
    def add(a,b): return (a[0]+b[0], a[1]+b[1])
    def sub(a,b): return (a[0]-b[0], a[1]-b[1])
    if m == 0: P[0] = sub(add(TR, BL), BR)
    elif m == 1: P[1] = sub(add(TL, BR), BL)
    elif m == 2: P[2] = sub(add(TL, BR), TR)
    else:        P[3] = sub(add(TR, BL), TL)
    return P, m

def _row_from_points(pts4: List[Tuple[float,float]], W: int, H: int, cls_id: int = 0) -> List[float]:
    xs = [p[0] for p in pts4]; ys = [p[1] for p in pts4]
    x1, x2 = min(xs), max(xs); y1, y2 = min(ys), max(ys)
    cx = ((x1 + x2) / 2.0) / W
    cy = ((y1 + y2) / 2.0) / H
    bw = max((x2 - x1) / W, 1e-6)
    bh = max((y2 - y1) / H, 1e-6)
    k = []
    for (x, y) in pts4:
        k += [x / W, y / H, 2.0]
    return [float(cls_id), float(cx), float(cy), float(bw), float(bh), *[float(x) for x in k]]

def fallback_logic(
    cropper: CornerCropper, 
    metas4: List[CropMeta | None], 
    rows4: List[List[float] | None], 
    W: int, H: int
) -> Tuple[List[float] | None, bool]:
    missing = [i for i,(m,r) in enumerate(zip(metas4, rows4)) if (m is None or r is None)]
    if not missing:
        return cropper.merge_page(metas4, rows4), False
    if len(missing) == 1:
        pts = [None]*4
        for cid in range(4):
            m, r = metas4[cid], rows4[cid]
            if m is None or r is None: pts[cid] = None
            else: pts[cid] = _row_to_page_xy(m, r, cropper)
        pts_filled, miss_idx = _infer_fourth(pts)
        if miss_idx is None or any(p is None for p in pts_filled):
            return None, False
        row = _row_from_points(pts_filled, W=W, H=H, cls_id=0)
        return row, True
    return None, False

def merge_images(
    by_page: Dict[str, Dict[str, List]], 
    out_img: Path, 
    out_lbl: Path,
    out_unl: Path, 
    cropper: CornerCropper
) -> Tuple[int,int,int]:
    merged_pages = 0
    partial_pages = 0
    filled_pages = 0
    for page_rel, pack in sorted(by_page.items(), key=lambda kv: kv[0]):
        metas4, rows4 = pack["metas"], pack["rows"]
        ip = Path(page_rel)
        bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if bgr is None: 
            print(f"[WARN] could not read page image: {ip}")
            continue
        cv2.imwrite(str(out_img / ip.name), bgr)
        H, W = bgr.shape[:2]
        row, used_fallback = fallback_logic(cropper, metas4, rows4, W, H)
        if row is None:
            cv2.imwrite(str(out_unl / ip.name), bgr)
            miss = [i for i,(m,r) in enumerate(zip(metas4, rows4)) if (m is None or r is None)]
            print(f"[MISS] {ip.name}: missing corners {miss}")
            partial_pages += 1
            continue
        cropper._write_row(out_lbl / (ip.stem + ".txt"), row)
        if used_fallback:
            filled_pages += 1
        else:
            merged_pages += 1
    return merged_pages, filled_pages, partial_pages




@dataclass(frozen=True)
class DataYamlSpec:
    """YOLO-style header for page 4-keypoint pose."""
    names: Tuple[str, ...] = ("page_corners",)
    kpt_shape: Tuple[int, int] = (4, 3)
    flip_idx: Tuple[int, ...] = (1, 0, 3, 2)  # H-flip: TL↔TR, BL↔BR

def _dump_with_flow_lists(obj: dict) -> str:
    """YAML dump keeping lists in flow style."""
    class _FlowSeqDumper(yaml.SafeDumper): ...
    def _repr_seq(dumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
    _FlowSeqDumper.add_representer(list, _repr_seq)
    _FlowSeqDumper.add_representer(tuple, _repr_seq)
    return yaml.dump(obj, Dumper=_FlowSeqDumper, sort_keys=False)

def write_data_yaml(out_root: Path, spec: DataYamlSpec = DataYamlSpec()) -> None:
    """Create data.yaml co-located with outputs; splits all point to images/."""
    yd = {
        "path": ".",
        "train": "images",
        "val": "images",
        "test": "images",
        "names": list(spec.names),
        "kpt_shape": list(spec.kpt_shape),
        "flip_idx": list(spec.flip_idx),
    }
    (out_root / "data.yaml").write_text(_dump_with_flow_lists(yd))
    

def run_corner_pose(
    images_dir: str,
    model_path: str,
    out_root: str,
    tmp_root: str,
    factor: float = 0.28,
    conf: float = 0.10,
    imgsz: int = 640,
):
    images_dir = Path(images_dir)
    model_path = Path(model_path)
    out_root   = Path(out_root)
    tmp_root   = Path(tmp_root)

    d = _ensure_dirs(out_root, tmp_root)
    cropper = CornerCropper(CropCfg(factor=factor, bbox_pad=0.15, out_ext=".jpg"))

    imgs = _list_images(images_dir)
    metas = crop_images(imgs, cropper, tmp_root, factor=factor)
    if not metas:
        print("no images found.")
        return

    lbl_pred_root = predict_yolo(model_path, tmp_root, conf=conf, imgsz=imgsz)
    if lbl_pred_root is None:
        print("[ERROR] No predictions produced (save_dir is None).")
        shutil.rmtree(tmp_root)
        return

    by_page: Dict[str, Dict[str, List]] = {}
    for m in metas:
        pr = lbl_pred_root / f"{m.crop_name}.txt"
        pack = by_page.setdefault(m.page_rel, {"metas": [None]*4, "rows": [None]*4})
        if pr.exists():
            row = [float(x) for x in pr.read_text().split()]
            pack["metas"][m.corner_id] = m
            pack["rows"][m.corner_id] = row

    merged, filled, partial = merge_images(by_page, d["out_img"], d["out_lbl"], d["out_unl"], cropper)
    write_data_yaml(out_root)
    shutil.rmtree(tmp_root)
    print(f"[DONE] merged_pages={merged}, filled_pages={filled}, partial_pages={partial}, out_root={out_root}")
