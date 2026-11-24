# mediadig

Search your local images, memes, GIFs, videos, PDFs, and documents by visible text or by meaning, completely offline on Linux.

## What it does

`mediadig` scans folders you pick so you can find files without remembering their filenames:

- Search by text inside images, subtitles, scanned PDFs, and text documents.
- Search by description to find a photo, GIF, or video clip from what it shows, such as `"cat falling off table"`.
- Jump to the matching timestamp in a video or the matching page in a PDF.
- Filter results by file type by putting a word like `vid`, `img`, `gif`, or `pdf` at the start or end of a query.
- Keep the index updated automatically while a background daemon watches your folders.
- Search from the terminal or directly from the Noctalia desktop launcher.

Everything runs locally on your NVIDIA GPU. Your files never leave your computer.

## Installation

You need Linux, an NVIDIA GPU with CUDA, Python 3.12 or newer with `uv`, `ffmpeg`, `ffprobe`, and `llama-server` from llama.cpp with multimodal EmbeddingGemma 2 support.

Clone the repository and install Python dependencies with `uv`:

```bash
git clone <repo-url>
cd mediadig
uv sync
```

Install `ffmpeg`, `ffprobe`, and `llama-server` through your system package manager or build them from source.

Download the EmbeddingGemma 2 GGUF weights and place them in `~/.local/share/mediadig/models/`:

- `embeddinggemma-2-Q8_0.gguf`
- `mmproj-embeddinggemma-2-Q8_0.gguf`

RapidOCR downloads its OCR models automatically into `~/.local/share/mediadig/ocr-models/` on the first index run.

Configure the folders to index in `~/.config/mediadig/config.toml`:

```toml
roots = ["~/Pictures", "~/Documents"]
```

To enable meaning search at query time, run a persistent text-only `llama-server` on the query socket:

```bash
mkdir -p "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/mediadig-embed"
llama-server \
  -m ~/.local/share/mediadig/models/embeddinggemma-2-Q8_0.gguf \
  --embeddings \
  --host "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/mediadig-embed/embed.sock"
```

If that server is not running, `mediadig` still searches captions and extracted text.

## Usage

Run a one-shot index pass:

```bash
uv run mediadig index
uv run mediadig index --roots ~/Pictures --data-dir /tmp/md-test --limit 20
```

Run the background daemon to watch configured folders and serve searches:

```bash
uv run mediadig serve
```

Query the daemon, or pass `--local` to read the SQLite index directly without a running daemon:

```bash
uv run mediadig search "cat falling off table"
uv run mediadig search "vid dog" -n 5 --min 1.5 --json
uv run mediadig search "invoice" --local --data-dir /tmp/md-test
```

Putting a type word at the start or end of a multi-word query restricts results to that kind. Supported words include `img`, `photo`, `vid`, `gif`, `pdf`, `docx`, `txt`, `doc`, and Czech equivalents such as `fotky`, `obrazek`, `videa`, and `dokument`.

Check daemon status:

```bash
uv run mediadig status
```

## Noctalia plugin

This repository also works as a Noctalia v5 plugin source through `catalog.toml` and `mediadig/` for plugin ID `tzavo/mediadig`. The plugin requires `curl`, `mpv`, `xdg-open`, and a running `mediadig serve` daemon.

Add this repository as a Git plugin source and enable the plugin:

```bash
noctalia msg plugins source add mediadig git <repo-url>
noctalia msg plugins enable tzavo/mediadig
```

From a local checkout, you can register the checkout as a `path` source or copy `mediadig/` into your local Noctalia plugin directory:

```bash
noctalia msg plugins source add mediadig path /path/to/mediadig
# or
cp -r mediadig ~/.local/share/noctalia/plugins/mediadig
```

Once enabled, type `/md <query>` in the launcher for up to 30 results, or type at least 3 characters in global launcher search for up to 6 strong matches.

## Technical details

### How it works

`mediadig` indexes local folders into one SQLite database at `~/.local/share/mediadig/index.sqlite` and stores thumbnails in `~/.cache/mediadig/thumbs/`. Passing `--data-dir` to `serve`, `search`, or `status` isolates the database, thumbnail cache, and daemon socket path.

For each file, the indexer extracts text and 768-dimensional EmbeddingGemma 2 vectors:

- Images get OCR text from RapidOCR PP-OCRv6 and one visual embedding.
- `ffmpeg` samples videos and GIFs up to 10 minutes long every 2 seconds. The indexer drops duplicate frames using perceptual hashing and OCR text similarity, then embeds each kept frame.
- PDFs up to 500 pages use extracted text chunks on text pages and rendered page images with OCR on scanned pages.
- The indexer splits DOCX and TXT files up to 5 MB into 1800-character chunks with a 200-character overlap.

Search combines cosine similarity over embeddings, z-scored separately for visual and text segments, with fuzzy word coverage from RapidFuzz.

Two processes keep VRAM free between indexing runs:

- `mediadig index` runs as a one-shot batch, spawns a temporary multimodal `llama-server`, indexes new or changed files, and exits.
- `mediadig serve` watches configured roots, triggers incremental index runs when files change, keeps the index in memory, and answers queries over a UNIX socket at `$XDG_RUNTIME_DIR/mediadig/search.sock`.

Semantic queries use the separate text-only `llama-server` on `$XDG_RUNTIME_DIR/mediadig-embed/embed.sock`. If that socket is unreachable, search falls back to word matching.

The daemon exposes two JSON endpoints over its UNIX socket:

- `GET /search?q=<query>&n=<1..50>&min=<float>`
- `GET /status`

### Compatibility and limits

- Linux only. The code uses `fcntl`, `AF_UNIX` sockets, and `XDG_RUNTIME_DIR` with a fallback to `/run/user/<uid>`.
- NVIDIA GPU with CUDA required. OCR uses `onnxruntime-gpu[cuda,cudnn]==1.32.0` with CUDA 13 wheels and aborts if `CUDAExecutionProvider` is not active. Indexing runs `llama-server` with `-ngl 99`. There is no CPU fallback.
- Python 3.12 or newer, managed with `uv`.
- External binaries required on `PATH`: `llama-server` from llama.cpp with multimodal EmbeddingGemma 2 support, `ffmpeg`, and `ffprobe`.
- Supported extensions: `.jpg`, `.jpeg`, `.jfif`, `.png`, `.webp`, `.gif`, `.mp4`, `.webm`, `.mov`, `.mkv`, `.m4v`, `.pdf`, `.docx`, `.txt`.
- Normalization strips diacritics, and OCR loads the PP-OCRv6 multilingual models via `lang_type="cs"`. Stopwords and kind-filter words include English and Czech.
