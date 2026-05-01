from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any


BASE_URL = "http://127.0.0.1:8000"


def request_json(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def print_response(title: str, response: Any) -> None:
    print(f"\n## {title}")
    print(json.dumps(response, indent=2))


def health_check() -> None:
    response = request_json("GET", "/health")
    print_response("GET /health", response)


def add_memory() -> str:
    payload = {
        "category": "PRODUCT",
        "title": "Example API note",
        "event_date": "2026-05-01",
        "account": "",
        "product": "Oracle Agent Memory",
        "source": "hivemind_api_examples.py",
        "tags": ["api", "example"],
        "content": "This is an example note inserted through the HiveMind AI API.",
    }
    response = request_json("POST", "/memories", payload)
    print_response("POST /memories", response)
    return response["id"]


def list_memories() -> None:
    query = urllib.parse.urlencode(
        {
            "limit": 5,
            "category": "PRODUCT",
        }
    )
    response = request_json("GET", f"/memories?{query}")
    print_response("GET /memories", response)


def search_memories() -> None:
    query = urllib.parse.urlencode(
        {
            "q": "Oracle Agent Memory API example",
            "limit": 5,
            "category": "PRODUCT",
        }
    )
    response = request_json("GET", f"/search?{query}")
    print_response("GET /search", response)


def delete_memory(memory_id: str) -> None:
    response = request_json("POST", "/memories/delete", {"id": memory_id})
    print_response("POST /memories/delete", response)


if __name__ == "__main__":
    health_check()
    created_id = add_memory()
    list_memories()
    search_memories()
    delete_memory(created_id)
