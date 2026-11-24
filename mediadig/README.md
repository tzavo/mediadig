# mediadig

Search local memes, images, videos, and documents by caption text or meaning from the Noctalia launcher through the `mediadig` daemon.

## Plugin

| Field | Value |
| --- | --- |
| ID | `tzavo/mediadig` |
| Plugin API | `24` |
| Entries | Launcher providers `search` (`/md`) and `global` (`/mdg`) |

## Requirements

- A running `mediadig serve` daemon listening on `${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/mediadig/search.sock`
- `curl` on `PATH` to query the daemon socket
- `mpv` on `PATH` to open matched videos at the matching timestamp
- `xdg-open` on `PATH` to open images, GIFs, and documents

## Usage

Type `/md <query>` in the launcher to get up to 30 results with status and error rows.

Normal launcher queries of at least 3 characters also query the `global` provider (`/mdg`), returning up to 6 results with a minimum score of `3.0` and suppressing errors when the daemon is stopped.

Selecting a video opens it in `mpv` 5 seconds before the matched frame. Selecting any other file opens it with `xdg-open`.
