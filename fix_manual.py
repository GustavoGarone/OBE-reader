import argparse
import atexit
import csv
import select
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_list_allow_dash(s: str) -> List[int]:
    s = s.strip().strip('"').strip("'")
    if not s:
        return []
    out: List[int] = []
    for token in s.replace(",", " ").split():
        if not token:
            continue
        if token == "-":
            out.append(-1)
            continue
        try:
            out.append(int(token))
        except ValueError:
            raise ValueError(f"invalid token '{token}' (use digits or '-')")
    return out


LETTER2CHOICE = {"a": 0, "b": 1, "c": 2, "d": 3}


def parse_answers_tokens(s: str) -> List[int]:
    s = s.strip().strip('"').strip("'")
    if not s:
        return []
    out: List[int] = []
    for token in s.replace(",", " ").split():
        if not token:
            continue
        if token == "-":
            out.append(-1)
            continue
        low = token.lower()
        if low in LETTER2CHOICE:
            out.append(LETTER2CHOICE[low])
            continue
        try:
            out.append(int(token))
        except ValueError:
            raise ValueError(
                f"invalid answers token '{token}' (use a/b/c/d, digits, or '-')"
            )
    return out


def fmt_list(xs: Sequence[int]) -> str:
    return ",".join(str(int(x)) for x in xs)


def has_minus_one(a: Sequence[int]) -> bool:
    return any(x == -1 for x in a)


def ensure_csv_with_header(csv_path: Path) -> None:
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["student_id", "answers", "img_path", "report_path"])


def read_csv_rows(csv_path: Path) -> List[List[str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return [r for r in csv.reader(f)]


def write_csv_rows(csv_path: Path, rows: List[List[str]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def image_iter(
    images_arg: Path, recursive: bool, pattern: Optional[str]
) -> Iterable[Path]:
    base = images_arg
    if base.is_file():
        if base.suffix.lower() in {".txt", ".lst", ".list"}:
            for line in base.read_text(encoding="utf-8").splitlines():
                p = Path(line.strip())
                if p.suffix.lower() in IMG_EXTS and p.exists():
                    yield p
        else:
            if base.suffix.lower() in IMG_EXTS:
                yield base
        return
    if not base.exists():
        return
    if pattern:
        it = base.rglob(pattern) if recursive else base.glob(pattern)
        for p in it:
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p
    else:
        it = base.rglob("*") if recursive else base.glob("*")
        for p in it:
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p


def to_rel_data_str(path_like: str | Path) -> str:
    """
    Return 'data/...' relative from first 'data' ancestor in the path.
    If no 'data' segment exists, return the original string.
    Uses POSIX-style separators.
    """
    p = Path(str(path_like))
    parts = p.parts
    if "data" in parts:
        idx = parts.index("data")
        return Path(*parts[idx:]).as_posix()
    try:
        rp = p.resolve()
        parts = rp.parts
        if "data" in parts:
            idx = parts.index("data")
            return Path(*parts[idx:]).as_posix()
    except Exception:
        pass
    return p.as_posix()


def make_mapped_report_path(
    img_path: Path, images_root: Path, report_root: Optional[Path]
) -> Path:
    if report_root is None:
        return img_path
    try:
        rel = img_path.resolve().relative_to(images_root.resolve())
    except Exception:
        rel = Path(img_path.name)
    return report_root / rel


class Viewer:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 10))
        self.img_artist = None
        self.text_artist = None
        self.ax.axis("off")
        self.skip_flag = False
        btn_ax = self.fig.add_axes([0.80, 0.01, 0.18, 0.055])
        self.btn = Button(btn_ax, "Skip row")
        self.btn.on_clicked(self._on_skip_clicked)

    def _on_skip_clicked(self, event=None):
        self.skip_flag = True

    def consume_skip(self) -> bool:
        if self.skip_flag:
            self.skip_flag = False
            return True
        return False

    def show(
        self,
        display_path: Path,
        row_idx: int,
        sid: Sequence[int],
        ans: Sequence[int],
        mode: str,
    ) -> None:
        try:
            im = Image.open(display_path)
        except Exception as e:
            print(f"[warn] could not open image: {display_path} ({e})")
            im = Image.new("RGB", (800, 1000), (240, 240, 240))

        if self.img_artist is None:
            self.img_artist = self.ax.imshow(im)
        else:
            self.img_artist.set_data(im)

        if self.text_artist is not None:
            self.text_artist.remove()

        text = (
            f"{display_path.name}\n"
            f"Mode: {mode}  |  Row: {row_idx}\n"
            f"student_id: {sid}\n"
            f"answers   : {ans}\n"
            f"Tips: '-' -> -1 | answers accept a/b/c/d | 'q' quits | Ctrl+C quits | Button: Skip row"
        )
        self.text_artist = self.ax.text(
            5,
            5,
            text,
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="black", pad=5),
            fontsize=10,
        )
        self.fig.canvas.manager.set_window_title(
            f"[{mode}] {row_idx} — {display_path.name}"
        )
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def ping(self):
        plt.pause(0.001)

    def close(self):
        try:
            plt.close(self.fig)
        except Exception:
            pass


def typed(prompt: str, viewer: Viewer) -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    while True:
        viewer.ping()
        if viewer.consume_skip():
            print("[skip] skipping this row.")
            return "__SKIP__"
        try:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                return sys.stdin.readline().rstrip("\r\n")
        except KeyboardInterrupt:
            print("\n[interrupt] quitting now.")
            raise


def run_fix_mode(
    csv_path: Path, lo_id: int, hi_id: int, lo_an: int, hi_an: int, jump: int
) -> None:
    rows = read_csv_rows(csv_path)
    header = rows[0]
    try:
        i_sid = header.index("student_id")
        i_ans = header.index("answers")
        i_rep = header.index("report_path")
    except ValueError as e:
        print(f"missing expected column: {e}")
        sys.exit(1)

    viewer = Viewer()
    atexit.register(viewer.close)

    i = max(1, 1 + jump)
    try:
        while i < len(rows):
            r = rows[i]
            sid_cur = parse_list_allow_dash(r[i_sid])
            ans_cur = parse_list_allow_dash(r[i_ans])
            if not has_minus_one(sid_cur) and not has_minus_one(ans_cur):
                i += 1
                continue

            report_img_fs = Path(r[i_rep])
            viewer.show(report_img_fs, i, sid_cur, ans_cur, mode="fix")

            print("\n" + "=" * 60)
            print(f"[FIX] Row {i} | file: {r[i_rep]}")
            print(f" current student_id: {sid_cur}")
            print(f" current answers   : {ans_cur}")
            print("student_id: digits; use '-' for missing (-1)")
            print("answers   : digits OR letters a/b/c/d; '-' for missing (-1)")
            print("ENTER keeps field | 'q' quits | Skip button skips row")

            if viewer.consume_skip():
                print("[skip] skipping this row.")
                i += 1
                continue

            s = typed("student_id = ", viewer).strip()
            if s == "__SKIP__":
                i += 1
                continue
            if s.lower() == "q":
                break
            if s == "":
                sid_new = sid_cur
            else:
                try:
                    sid_new = parse_list_allow_dash(s)
                    if not all((x == -1) or (lo_id <= x <= hi_id) for x in sid_new):
                        raise ValueError
                except Exception:
                    print(
                        f"  invalid student_id; use digits [{lo_id}..{hi_id}] or '-' for -1."
                    )
                    continue
            sid_cur = sid_new
            viewer.show(report_img_fs, i, sid_cur, ans_cur, mode="fix")

            if viewer.consume_skip():
                print("[skip] skipping this row.")
                i += 1
                continue

            s = typed("answers = ", viewer).strip()
            if s == "__SKIP__":
                i += 1
                continue
            if s.lower() == "q":
                break
            if s == "":
                ans_new = ans_cur
            else:
                try:
                    ans_new = parse_answers_tokens(s)
                    if not all((x == -1) or (lo_an <= x <= hi_an) for x in ans_new):
                        raise ValueError
                except Exception as e:
                    print(f"  {e}")
                    print(
                        f"  valid answers: a/b/c/d (→0/1/2/3), digits [{lo_an}..{hi_an}], or '-'"
                    )
                    continue
            ans_cur = ans_new
            viewer.show(report_img_fs, i, sid_cur, ans_cur, mode="fix")

            if viewer.consume_skip():
                print("[skip] skipping this row.")
                i += 1
                continue

            print(f" -> new student_id: {sid_cur}")
            print(f" -> new answers   : {ans_cur}")
            confirm = typed("save this row? [y/N] ", viewer).strip().lower()
            if confirm == "__SKIP__":
                i += 1
                continue
            if confirm == "y":
                r[i_sid] = fmt_list(sid_cur)
                r[i_ans] = fmt_list(ans_cur)
                r[i_rep] = to_rel_data_str(r[i_rep])
                rows[i] = r
                try:
                    idx_img = header.index("img_path")
                    rows[i][idx_img] = to_rel_data_str(rows[i][idx_img])
                except ValueError:
                    pass
                write_csv_rows(csv_path, rows)
                print("saved.")
            else:
                print("skipped (not saved).")
            i += 1
    except KeyboardInterrupt:
        print("\n[interrupt] stopped by user.")
    finally:
        viewer.close()
        print("bye.")


def run_append_mode(
    csv_path: Path,
    images: Path,
    recursive: bool,
    pattern: Optional[str],
    lo_id: int,
    hi_id: int,
    lo_an: int,
    hi_an: int,
    report_root: Optional[Path],
) -> None:
    ensure_csv_with_header(csv_path)
    rows = read_csv_rows(csv_path)
    header = rows[0]
    try:
        i_sid = header.index("student_id")
        i_ans = header.index("answers")
        i_img = header.index("img_path")
        i_rep = header.index("report_path")
    except ValueError as e:
        print(f"missing expected column: {e}")
        sys.exit(1)

    existing_pairs = {
        (to_rel_data_str(r[i_img]), to_rel_data_str(r[i_rep])) for r in rows[1:]
    }

    viewer = Viewer()
    atexit.register(viewer.close)

    appended = 0
    try:
        idx = 1
        images_root = images.resolve()
        rep_root_resolved = report_root.resolve() if report_root else None

        for img_path in image_iter(images, recursive, pattern):
            img_real = img_path.resolve()
            rep_real = make_mapped_report_path(
                img_real, images_root, rep_root_resolved
            ).resolve()

            img_rel = to_rel_data_str(img_real)
            rep_rel = to_rel_data_str(rep_real)
            pair_rel = (img_rel, rep_rel)
            if pair_rel in existing_pairs:
                continue

            sid_cur: List[int] = []
            ans_cur: List[int] = []
            viewer.show(img_real, idx, sid_cur, ans_cur, mode="append")

            print("\n" + "=" * 60)
            print(f"[APPEND] {img_rel}")
            print("Enter new values.")
            print("student_id: digits; use '-' for missing (-1)")
            print("answers   : digits OR letters a/b/c/d; '-' for missing (-1)")
            print(
                "ENTER keeps field (empty → []) | 'q' quits | Skip button skips image"
            )

            if viewer.consume_skip():
                print("[skip] skipping this image.")
                idx += 1
                continue

            s = typed("student_id = ", viewer).strip()
            if s == "__SKIP__":
                idx += 1
                continue
            if s.lower() == "q":
                break
            if s == "":
                sid_new = sid_cur
            else:
                try:
                    sid_new = parse_list_allow_dash(s)
                    if not all((x == -1) or (lo_id <= x <= hi_id) for x in sid_new):
                        raise ValueError
                except Exception:
                    print(
                        f"  invalid student_id; use digits [{lo_id}..{hi_id}] or '-' for -1."
                    )
                    continue
            sid_cur = sid_new
            viewer.show(img_real, idx, sid_cur, ans_cur, mode="append")

            if viewer.consume_skip():
                print("[skip] skipping this image.")
                idx += 1
                continue

            s = typed("answers = ", viewer).strip()
            if s == "__SKIP__":
                idx += 1
                continue
            if s.lower() == "q":
                break
            if s == "":
                ans_new = ans_cur
            else:
                try:
                    ans_new = parse_answers_tokens(s)
                    if not all((x == -1) or (lo_an <= x <= hi_an) for x in ans_new):
                        raise ValueError
                except Exception as e:
                    print(f"  {e}")
                    print(
                        f"  valid answers: a/b/c/d (→0/1/2/3), digits [{lo_an}..{hi_an}], or '-'"
                    )
                    continue
            ans_cur = ans_new
            viewer.show(img_real, idx, sid_cur, ans_cur, mode="append")

            if viewer.consume_skip():
                print("[skip] skipping this image.")
                idx += 1
                continue

            print(f" -> new student_id: {sid_cur}")
            print(f" -> new answers   : {ans_cur}")
            confirm = typed("append this row? [y/N] ", viewer).strip().lower()
            if confirm == "__SKIP__":
                idx += 1
                continue
            if confirm == "y":
                rows.append([fmt_list(sid_cur), fmt_list(ans_cur), img_rel, rep_rel])
                write_csv_rows(csv_path, rows)
                existing_pairs.add(pair_rel)
                appended += 1
                print(f"appended (total now {appended}).")
            else:
                print("skipped (not saved).")
            idx += 1
    except KeyboardInterrupt:
        print("\n[interrupt] stopped by user.")
    finally:
        viewer.close()
        print(f"bye. appended {appended} rows.")


def main():
    ap = argparse.ArgumentParser(
        description="Fix -1 rows in CSV (default) or append new rows by labeling images. Paths are saved relative to 'data/'."
    )
    ap.add_argument("csv_path", type=Path, help="CSV to read/write.")
    ap.add_argument(
        "--jump",
        type=int,
        default=0,
        help="[fix] Start from this row index (after header).",
    )
    ap.add_argument(
        "--id-range",
        type=str,
        default="0-9",
        help="Valid digit range for student_id (e.g., 0-9).",
    )
    ap.add_argument(
        "--ans-range",
        type=str,
        default="0-3",
        help="Valid choice range for answers (e.g., 0-3).",
    )

    ap.add_argument(
        "--images",
        type=Path,
        default=None,
        help="[append] Directory, glob-list file (.txt), or single image to label.",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="[append] Recurse subfolders when scanning a directory.",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default=None,
        help='[append] Glob pattern inside --images (e.g., "*.jpg"). If omitted, all images are used.',
    )
    ap.add_argument(
        "--report-root",
        type=Path,
        default=None,
        help="[append] If set, map report_path = report_root / relative(img_path, --images).",
    )

    args = ap.parse_args()

    lo_id, hi_id = [int(x) for x in args.id_range.split("-")]
    lo_an, hi_an = [int(x) for x in args.ans_range.split("-")]

    ensure_csv_with_header(args.csv_path)

    if args.images is None:
        run_fix_mode(args.csv_path, lo_id, hi_id, lo_an, hi_an, args.jump)
    else:
        run_append_mode(
            csv_path=args.csv_path,
            images=args.images,
            recursive=args.recursive,
            pattern=args.pattern,
            lo_id=lo_id,
            hi_id=hi_id,
            lo_an=lo_an,
            hi_an=hi_an,
            report_root=args.report_root,
        )


if __name__ == "__main__":
    main()
