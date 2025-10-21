"""
Flatten into DST: copy all images/PDFs from folders and from ZIP/RAR (optionally nested).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Set, Tuple

try:
    import rarfile
except Exception:
    rarfile = None

IMG_PDF_EXTS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".gif",
    ".bmp",
    ".pdf",
}
ARCHIVE_EXTS: Set[str] = {".zip", ".rar"}


@dataclass(frozen=True)
class Config:
    src: Path
    dst: Path
    nested_zips: bool = True
    exts: Set[str] = field(default_factory=lambda: IMG_PDF_EXTS)


@dataclass
class Stats:
    copied: int = 0
    extracted: int = 0
    skipped: int = 0
    errors: int = 0


def copy_unique(src: Path, dst_dir: Path) -> Path:
    stem, ext = os.path.splitext(src.name)
    out = dst_dir / src.name
    n = 1
    while out.exists():
        out = dst_dir / (f"{stem}__dup{n}{ext}" if ext else f"{stem}__dup{n}")
        n += 1
    shutil.copy2(src, out)
    return out


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def _extract_zip(zp: Path, td: str) -> None:
    with zipfile.ZipFile(zp) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        z.extractall(td, members=members)


def _extract_rar(rp: Path, td: str) -> None:
    if rarfile is not None:
        with rarfile.RarFile(rp) as rf:
            rf.extractall(td)
        return
    bsdtar = shutil.which("bsdtar")
    if bsdtar:
        subprocess.run(
            [bsdtar, "-x", "-f", str(rp), "-C", td],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    unrar = shutil.which("unrar")
    if unrar:
        Path(td).mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [unrar, "x", "-o+", str(rp), td],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    raise RuntimeError(
        "No RAR extractor found. Install 'bsdtar' or 'unrar', or the 'rarfile' Python package."
    )


def handle_archive(
    ap: Path, dst: Path, nested: bool, exts: Set[str], st: Stats
) -> None:
    try:
        with tempfile.TemporaryDirectory() as td:
            if ap.suffix.lower() == ".zip":
                _extract_zip(ap, td)
            elif ap.suffix.lower() == ".rar":
                _extract_rar(ap, td)
            else:
                st.skipped += 1
                print(f"SKIP (unknown archive): {ap}", file=sys.stderr)
                return
            for f in iter_files(Path(td)):
                suf = f.suffix.lower()
                if nested and suf in ARCHIVE_EXTS:
                    handle_archive(f, dst, nested, exts, st)
                elif suf in exts:
                    copy_unique(f, dst)
                    st.extracted += 1
                else:
                    st.skipped += 1
                    print(f"SKIP: {f}", file=sys.stderr)
    except Exception as e:
        st.errors += 1
        print(f"ERROR extract '{ap}': {e}", file=sys.stderr)


def flatten_unzip(cfg: Config) -> Stats:
    cfg.dst.mkdir(parents=True, exist_ok=True)
    st = Stats()
    for f in iter_files(cfg.src):
        suf = f.suffix.lower()
        try:
            if suf in ARCHIVE_EXTS:
                handle_archive(f, cfg.dst, cfg.nested_zips, cfg.exts, st)
            elif suf in cfg.exts:
                copy_unique(f, cfg.dst)
                st.copied += 1
            else:
                st.skipped += 1
                print(f"SKIP: {f}", file=sys.stderr)
        except Exception as e:
            st.errors += 1
            print(f"ERROR copy '{f}': {e}", file=sys.stderr)
    return st


def parse_args() -> Tuple[Path, Path, bool]:
    ap = argparse.ArgumentParser(
        description="Flatten: copy all images/PDFs and extract ZIP/RAR into DST."
    )
    ap.add_argument("src", type=Path, help="Origem.")
    ap.add_argument("dst", type=Path, help="Destino achatado.")
    ap.add_argument(
        "--no-nested",
        action="store_true",
        help="Não processar arquivos-compactados dentro de compactados.",
    )
    a = ap.parse_args()
    return a.src, a.dst, (not a.no_nested)


if __name__ == "__main__":
    src, dst, nested = parse_args()
    st = flatten_unzip(Config(src=src, dst=dst, nested_zips=nested))
    print(
        f"copied={st.copied} extracted={st.extracted} skipped={st.skipped} errors={st.errors}"
    )
