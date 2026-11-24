"""Command line: serve, index, search, status."""

from __future__ import annotations

import argparse
import json
import sys
from urllib.parse import urlencode

from mediadig import config


def _add_paths(p: argparse.ArgumentParser) -> None:
    p.add_argument("--roots", nargs="+", metavar="P", help="folders to index (overrides config.toml)")
    p.add_argument("--data-dir", metavar="D", help="index data directory")


def _passthrough(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    if args.roots:
        out += ["--roots", *args.roots]
    if args.data_dir:
        out += ["--data-dir", args.data_dir]
    return out


def _fmt_time(r: dict) -> str:
    if isinstance(r.get("t"), (int, float)):
        t = int(r["t"])
        return f"{t // 60:02d}:{t % 60:02d}"
    if isinstance(r.get("page"), int):
        return f"p.{r['page']}"
    return "-"


def _daemon_get(url: str, data_dir: str | None = None) -> dict:
    from mediadig.llama import request

    sock = config.search_sock_for(data_dir)
    try:
        status, body = request(sock, "GET", url, None, 10.0)
    except OSError as e:
        sys.exit(f"mediadig: daemon not reachable at {sock}: {e}")
    data = json.loads(body)
    if status != 200:
        sys.exit(f"mediadig: {data.get('error', status)}")
    return data


def cmd_search(args: argparse.Namespace) -> int:
    if args.local:
        from mediadig.search import load_state, make_embedder, search

        cfg = config.load(data_dir=args.data_dir)
        embed = make_embedder(cfg.embed_sock)
        try:
            res = search(load_state(cfg.db), args.query, args.n, args.min, embed)
        finally:
            embed.close()  # type: ignore[attr-defined]
    else:
        res = _daemon_get("/search?" + urlencode({"q": args.query, "n": args.n, "min": args.min}), args.data_dir)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    if not res["semantic"]:
        print("(meaning search unavailable: query encoder not reachable)", file=sys.stderr)
    for r in res["results"]:
        line = f"{r['score']:7.3f}  {r['match']:<7}  {_fmt_time(r):>6}  {r['path']}"
        if r["snippet"]:
            line += f"  | {r['snippet']}"
        print(line)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    print(json.dumps(_daemon_get("/status", args.data_dir), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mediadig")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="run the search daemon")
    _add_paths(p)

    p = sub.add_parser("index", help="index the roots once")
    _add_paths(p)
    p.add_argument("--limit", type=int, metavar="N", help="process at most N queued files")

    p = sub.add_parser("search", help="search the index")
    p.add_argument("query")
    p.add_argument("-n", type=int, default=10)
    p.add_argument("--min", type=float, default=0.0, help="minimum score")
    p.add_argument("--json", action="store_true")
    p.add_argument("--local", action="store_true", help="search in-process instead of via the daemon")
    p.add_argument("--data-dir", metavar="D", help="data directory (isolates daemon socket and --local DB)")

    p = sub.add_parser("status", help="show daemon status")
    p.add_argument("--data-dir", metavar="D", help="data directory for an isolated daemon")

    args = ap.parse_args(argv)
    if args.cmd == "serve":
        from mediadig import server

        return server.serve(config.load(args.roots, args.data_dir), _passthrough(args))
    if args.cmd == "index":
        from mediadig import indexer

        return indexer.run(config.load(args.roots, args.data_dir), args.limit)
    if args.cmd == "search":
        return cmd_search(args)
    return cmd_status(args)
