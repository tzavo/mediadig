"""Per-kind processors: turn one file into segments (text + image or text to embed)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, ImageOps
from rapidfuzz import fuzz

from mediadig import config
from mediadig.llama import doc_prompt
from mediadig.ocr import Ocr
from mediadig.sizing import fit_down, prepare_png
from mediadig.textnorm import chunk_text, normalize

# Accept PNGs with a bad CRC in an ancillary chunk (e.g. iCCP) and truncated files; other
# viewers open them, and a partly decoded image is still searchable.
ImageFile.LOAD_TRUNCATED_IMAGES = True

THUMB_SIZE = (96, 96)
FFMPEG_TIMEOUT = 600
PDF_MAX_PAGES = 500
PDF_MIN_PAGE_TEXT = 200
PDF_RENDER_SIDE = 1400
TXT_MAX_BYTES = 5 * 1024 * 1024
HASH_KEEP_DISTANCE = 4
TEXT_KEEP_RATIO = 90


class ExtractError(Exception):
    pass


class Skipped(ExtractError):
    """The file is deliberately not indexed (kept as status='skipped' until it changes)."""


@dataclass
class Seg:
    kind: str  # 'image' | 'frame' | 'page_image' | 'chunk'
    t: float | None = None
    page: int | None = None
    text: str = ""
    png: bytes | None = None  # image to embed, already sized for its budget
    doc_text: str | None = None  # text to embed


Result = tuple[list[Seg], Image.Image | None]


def make_thumb(img: Image.Image) -> Image.Image:
    out = fit_down(img, THUMB_SIZE, Image.Resampling.LANCZOS)
    return out.copy() if out is img else out


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img.convert("RGB")


def process_image(path: Path, ocr: Ocr) -> Result:
    with Image.open(path) as im:
        im.seek(0)
        img = _to_rgb(ImageOps.exif_transpose(im))
        if max(img.size) < 32:
            raise ExtractError("too small")
        seg = Seg("image", text=ocr.text(img), png=prepare_png(img, config.IMAGE_BUDGET))
        return [seg], make_thumb(img)


# --- video and gif -------------------------------------------------------------

class ChildTracker:
    """Tracks running ffmpeg/ffprobe children for one index run."""

    def __init__(self) -> None:
        self._procs: set[subprocess.Popen] = set()
        self._lock = threading.Lock()
        self._stopping = False

    def stop(self) -> None:
        """Kill running children and refuse new ones."""
        with self._lock:
            self._stopping = True
            procs = list(self._procs)
        for p in procs:
            if p.poll() is None:
                p.kill()

    def run(self, argv: list[str], timeout: float) -> subprocess.CompletedProcess:
        with self._lock:
            if self._stopping:
                raise ExtractError("index run stopping")
            p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self._procs.add(p)
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            raise
        finally:
            with self._lock:
                self._procs.discard(p)
        return subprocess.CompletedProcess(argv, p.returncode, out, err)


def probe_duration(path: Path, tracker: ChildTracker | None = None) -> float | None:
    tr = tracker or ChildTracker()
    try:
        r = tr.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)], 60)
        return float(r.stdout.decode().strip())
    except (subprocess.TimeoutExpired, ValueError):
        return None


@dataclass
class Frames:
    dir: Path
    frames: list[tuple[float, Path]]

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def _ffmpeg_frames(path: Path, out_dir: str, vf: str, extra: list[str], tracker: ChildTracker) -> list[Path]:
    # format=yuvj420p: GIFs decode to bgra, which the mjpeg encoder rejects otherwise.
    argv = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-an", "-sn",
            "-vf", f"{vf},scale=1024:1024:force_original_aspect_ratio=decrease,format=yuvj420p",
            *extra, "-q:v", "3", "-f", "image2", f"{out_dir}/%06d.jpg"]
    try:
        r = tracker.run(argv, FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise ExtractError("ffmpeg timeout") from None
    if r.returncode != 0:
        raise ExtractError(r.stderr.decode(errors="replace")[-300:].strip() or f"ffmpeg exit {r.returncode}")
    return sorted(Path(out_dir).glob("*.jpg"))


def extract_frames(path: Path, work_dir: Path, tracker: ChildTracker | None = None) -> Frames:
    """Runs on a worker thread: sample frames to JPEGs in a fresh directory under `work_dir`."""
    tr = tracker or ChildTracker()
    dur = probe_duration(path, tr)
    if dur and dur > config.MAX_MEDIA_SECONDS:
        raise Skipped(f"longer than {config.MAX_MEDIA_SECONDS // 60} min ({dur / 60:.0f} min)")
    step = config.FRAME_STEP
    first = min(0.5, dur / 2) if dur else 0.0
    out = Path(tempfile.mkdtemp(dir=work_dir))
    try:
        files = _ffmpeg_frames(path, str(out), f"fps=fps=1/{step}:start_time={first}:round=up", [], tr)
        if not files:
            # Single-frame clips (static GIFs) have no frame at or after `first`: take frame 0.
            first = 0.0
            files = _ffmpeg_frames(path, str(out), "null", ["-frames:v", "1"], tr)
        if not files:
            raise ExtractError("no frames")
        return Frames(out, [(first + k * step, f) for k, f in enumerate(files)])
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise


def _ahash(img: Image.Image) -> int:
    a = np.asarray(img.convert("L").resize((8, 8), Image.Resampling.BOX), dtype=np.float32)
    return int.from_bytes(np.packbits(a > a.mean()).tobytes(), "big")


def process_frames(fr: Frames, ocr: Ocr, cancel: threading.Event | None = None) -> Result:
    segs: list[Seg] = []
    thumb = None
    thumb_idx = round(0.1 * (len(fr.frames) - 1))
    last_hash = 0
    last_tn = ""
    for k, (t, f) in enumerate(fr.frames):
        if cancel is not None and cancel.is_set():
            raise ExtractError("index run stopping")
        with Image.open(f) as im:
            img = im if im.mode == "RGB" else im.convert("RGB")
            if k == thumb_idx:
                thumb = make_thumb(img)
            text = ocr.text(img)
            tn = normalize(text)
            h = _ahash(img)
            keep = (
                k == 0
                or (h ^ last_hash).bit_count() > HASH_KEEP_DISTANCE
                or (tn and fuzz.ratio(tn, last_tn) < TEXT_KEEP_RATIO)
            )
            if keep:
                segs.append(Seg("frame", t=round(t, 2), text=text, png=prepare_png(img, config.FRAME_BUDGET)))
                last_hash, last_tn = h, tn
    return segs, thumb


# --- documents -----------------------------------------------------------------

def _chunks(path: Path, text: str, page: int | None = None) -> list[Seg]:
    return [Seg("chunk", page=page, text=c, doc_text=doc_prompt(path.stem, c)) for c in chunk_text(text)]


def process_pdf(path: Path, ocr: Ocr, cancel: threading.Event | None = None) -> Result:
    import pypdfium2 as pdfium

    segs: list[Seg] = []
    pdf = pdfium.PdfDocument(path)
    try:
        for i in range(min(len(pdf), PDF_MAX_PAGES)):
            if cancel is not None and cancel.is_set():
                raise ExtractError("index run stopping")
            page = pdf[i]
            try:
                tp = page.get_textpage()
                text = tp.get_text_bounded()
                tp.close()
                if len(text.strip()) >= PDF_MIN_PAGE_TEXT:
                    segs.extend(_chunks(path, text, page=i + 1))
                else:
                    scale = PDF_RENDER_SIDE / max(page.get_size())
                    img = page.render(scale=scale).to_pil()
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    segs.append(Seg("page_image", page=i + 1, text=ocr.text(img),
                                    png=prepare_png(img, config.PAGE_BUDGET)))
            finally:
                page.close()
    finally:
        pdf.close()
    return segs, None


def process_docx(path: Path, ocr: Ocr) -> Result:
    import docx

    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs]
    for table in d.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return _chunks(path, "\n".join(parts)), None


def process_txt(path: Path, ocr: Ocr) -> Result:
    if path.stat().st_size > TXT_MAX_BYTES:
        raise ExtractError("too large")
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):  # Windows Notepad "Unicode"
        text = raw.decode("utf-16", errors="replace")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("cp1250")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
    return _chunks(path, text), None


PROCESSORS = {
    "image": process_image,
    "pdf": process_pdf,
    "docx": process_docx,
    "txt": process_txt,
}
