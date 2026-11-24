"""Paths, roots and constants."""

from __future__ import annotations

import hashlib
import os
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path

FINGERPRINT = "mediadig-v1 eg2-q8 img560 frame140/2.0s page560 chunk1800/200 ocr=ppocrv6-medium-cs"

EXIT_OK = 0
EXIT_FATAL = 1
EXIT_RETRY = 75

EMBED_DIM = 768
IMAGE_BUDGET = 560
FRAME_BUDGET = 140
PAGE_BUDGET = 560
FRAME_STEP = 2.0
MAX_MEDIA_SECONDS = 600  # longer videos/GIFs are skipped, not indexed

EXT_KIND = {
    ".jpg": "image", ".jpeg": "image", ".jfif": "image", ".png": "image", ".webp": "image",
    ".gif": "gif",
    ".mp4": "video", ".webm": "video", ".mov": "video", ".mkv": "video", ".m4v": "video",
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
}
KIND_ORDER = ("image", "pdf", "docx", "txt", "gif", "video")

HOME = Path.home()
DEFAULT_DATA_DIR = HOME / ".local/share/mediadig"
MODELS_DIR = DEFAULT_DATA_DIR / "models"
TEXT_MODEL = MODELS_DIR / "embeddinggemma-2-Q8_0.gguf"
MMPROJ_MODEL = MODELS_DIR / "mmproj-embeddinggemma-2-Q8_0.gguf"
CONFIG_FILE = HOME / ".config/mediadig/config.toml"
DEFAULT_THUMBS_DIR = HOME / ".cache/mediadig/thumbs"
LLAMA_SERVER = shutil.which("llama-server") or "/run/current-system/sw/bin/llama-server"


def runtime_dir() -> Path:
    return Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}")


def search_sock_for(data_dir: str | Path | None = None) -> Path:
    """Daemon search socket path; custom `--data-dir` gets an isolated socket."""
    if data_dir is None:
        return runtime_dir() / "mediadig/search.sock"
    d = _expand(str(data_dir))
    if d == DEFAULT_DATA_DIR:
        return runtime_dir() / "mediadig/search.sock"
    tag = hashlib.sha1(str(d).encode()).hexdigest()[:12]
    return runtime_dir() / f"mediadig/search-{tag}.sock"


def is_watched_path(path: str | Path, roots: tuple[Path, ...]) -> bool:
    """True if a watchfiles event path can affect the index (matches indexer.walk rules)."""
    p_str = str(path)
    ext = os.path.splitext(p_str)[1].lower()
    is_known_ext = ext in EXT_KIND
    is_dir = os.path.isdir(p_str)
    if not is_known_ext and not is_dir and (ext != "" or os.path.isfile(p_str)):
        return False
    for root in roots:
        r_str = str(root)
        if p_str == r_str:
            return not is_known_ext
        prefix = r_str if r_str.endswith(os.sep) else r_str + os.sep
        if p_str.startswith(prefix):
            parts = p_str[len(prefix):].split(os.sep)
            check = parts[:-1] if (is_known_ext and not is_dir) else parts
            return not any(part.startswith(".") for part in check)
    return True


@dataclass(frozen=True)
class Config:
    roots: tuple[Path, ...]
    data_dir: Path
    thumbs_dir: Path

    @property
    def db(self) -> Path:
        return self.data_dir / "index.sqlite"

    @property
    def lock(self) -> Path:
        return self.data_dir / "index.lock"

    @property
    def ocr_models(self) -> Path:
        return self.data_dir / "ocr-models"

    @property
    def index_log(self) -> Path:
        return self.data_dir / "index-embed.log"

    @property
    def frames_dir(self) -> Path:
        """Scratch space for sampled video frames; owned by the index run holding the lock."""
        return self.data_dir / "frames"

    @property
    def embed_sock(self) -> Path:
        return runtime_dir() / "mediadig-embed/embed.sock"

    @property
    def search_sock(self) -> Path:
        return search_sock_for(self.data_dir)

    def index_embed_sock(self, pid: int) -> Path:
        return runtime_dir() / f"mediadig/index-embed-{pid}.sock"


def _expand(p: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(p)))


def load(roots: list[str] | None = None, data_dir: str | None = None) -> Config:
    if roots:
        root_paths = [_expand(r) for r in roots]
    else:
        file_roots: list[str] = []
        if CONFIG_FILE.exists():
            with CONFIG_FILE.open("rb") as f:
                file_roots = tomllib.load(f).get("roots", [])
        root_paths = [_expand(r) for r in file_roots]
    if data_dir:
        d = _expand(data_dir)
        return Config(tuple(root_paths), d, d / "thumbs")
    return Config(tuple(root_paths), DEFAULT_DATA_DIR, DEFAULT_THUMBS_DIR)
