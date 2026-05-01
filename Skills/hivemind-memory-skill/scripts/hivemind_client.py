#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import date
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def request_json(
    method: str,
    path: str,
    *,
    base_url: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not connect to HiveMind API at {base_url}: {exc}") from exc


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def parse_tags(value: str) -> list[str]:
    return [tag.strip() for tag in value.split(",") if tag.strip()]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)


def cmd_health(args: argparse.Namespace) -> None:
    print_json(request_json("GET", "/health", base_url=args.base_url))


def cmd_add(args: argparse.Namespace) -> None:
    content = args.content
    if args.content_file:
        content = open(args.content_file, "r", encoding="utf-8").read()
    if not content:
        content = sys.stdin.read()
    if not content.strip():
        raise SystemExit("Provide --content, --content-file, or stdin content.")

    payload = {
        "category": args.category,
        "title": args.title,
        "event_date": args.event_date or date.today().isoformat(),
        "account": args.account,
        "product": args.product,
        "source": args.source,
        "tags": parse_tags(args.tags),
        "content": content,
    }
    print_json(request_json("POST", "/memories", base_url=args.base_url, payload=payload))


def cmd_list(args: argparse.Namespace) -> None:
    params = {
        "limit": args.limit,
        "category": args.category,
        "account": args.account,
        "product": args.product,
        "start_date": args.start_date,
        "end_date": args.end_date,
    }
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v not in ("", None)})
    print_json(request_json("GET", f"/memories?{query}", base_url=args.base_url))


def cmd_search(args: argparse.Namespace) -> None:
    params = {
        "q": args.query,
        "limit": args.limit,
        "category": args.category,
        "account": args.account,
        "product": args.product,
        "start_date": args.start_date,
        "end_date": args.end_date,
    }
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v not in ("", None)})
    print_json(request_json("GET", f"/search?{query}", base_url=args.base_url))


def cmd_delete(args: argparse.Namespace) -> None:
    print_json(
        request_json(
            "POST",
            "/memories/delete",
            base_url=args.base_url,
            payload={"id": args.id},
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HiveMind AI memory API client.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    health = subparsers.add_parser("health")
    add_common_args(health)
    health.set_defaults(func=cmd_health)

    add = subparsers.add_parser("add")
    add_common_args(add)
    add.add_argument("--category", choices=["ACCOUNT", "ORG", "PRODUCT"], default="PRODUCT")
    add.add_argument("--title", default="")
    add.add_argument("--event-date", default="")
    add.add_argument("--account", default="")
    add.add_argument("--product", default="")
    add.add_argument("--source", default="cline-codex")
    add.add_argument("--tags", default="")
    add.add_argument("--content", default="")
    add.add_argument("--content-file", default="")
    add.set_defaults(func=cmd_add)

    list_cmd = subparsers.add_parser("list")
    add_common_args(list_cmd)
    list_cmd.add_argument("--limit", type=int, default=20)
    list_cmd.add_argument("--category", default="ANY")
    list_cmd.add_argument("--account", default="")
    list_cmd.add_argument("--product", default="")
    list_cmd.add_argument("--start-date", default="")
    list_cmd.add_argument("--end-date", default="")
    list_cmd.set_defaults(func=cmd_list)

    search = subparsers.add_parser("search")
    add_common_args(search)
    search.add_argument("--query", required=True)
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--category", default="ANY")
    search.add_argument("--account", default="")
    search.add_argument("--product", default="")
    search.add_argument("--start-date", default="")
    search.add_argument("--end-date", default="")
    search.set_defaults(func=cmd_search)

    delete = subparsers.add_parser("delete")
    add_common_args(delete)
    delete.add_argument("--id", required=True)
    delete.set_defaults(func=cmd_delete)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
