from __future__ import annotations

import atexit
import os
import re
import socket
import subprocess
import sys
import textwrap
import time
import urllib.request
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import oracledb
import streamlit as st
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_oci.chat_models import ChatOCIGenAI
from oci.config import from_file
from oracleagentmemory.core import OracleAgentMemory
from oracleagentmemory.core.dbschemapolicy import SchemaPolicy
from oracleagentmemory.core.embedders.embedder import Embedder


CATEGORIES = ("ACCOUNT", "ORG", "PRODUCT")
DB_USER = os.getenv("HIVEMIND_DB_USER", "")
DB_PASSWORD = os.getenv("HIVEMIND_DB_PASSWORD", "")
CONNECT_STRING = os.getenv("HIVEMIND_DB_DSN", "")
AGENT_ID = os.getenv("HIVEMIND_AGENT_ID", "engagement_tracker")
USER_ID = os.getenv("HIVEMIND_USER_ID", "your-user-id")
TABLE_PREFIX = os.getenv("HIVEMIND_TABLE_PREFIX", "OAM_")
OCI_CONFIG_FILE = os.getenv("OCI_CONFIG_FILE", "~/.oci/config")
OCI_PROFILE = os.getenv("OCI_PROFILE", "DEFAULT")
EMBEDDING_MODEL = os.getenv("HIVEMIND_EMBEDDING_MODEL", "oci/cohere.embed-english-v3.0")
CHAT_MODEL = os.getenv("HIVEMIND_CHAT_MODEL", "openai.gpt-oss-120b")
MAX_RESPONSE_TOKENS = int(os.getenv("HIVEMIND_MAX_RESPONSE_TOKENS", "900"))
TEMPERATURE = float(os.getenv("HIVEMIND_TEMPERATURE", "0.2"))
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
HEADER_LOGO_PATH = ASSETS_DIR / "hivemind_ai_header.png"
ICON_PATH = ASSETS_DIR / "hivemind_ai_icon.png"
API_HOST = "127.0.0.1"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}"
LOGS_DIR = Path(__file__).resolve().parent / "logs"


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._skip = tag.lower() in {"script", "style", "noscript"}

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            cleaned = " ".join(data.split())
            if cleaned:
                self.parts.append(cleaned)

    def text(self) -> str:
        return "\n".join(self.parts)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def today_iso() -> str:
    return date.today().isoformat()


def normalize_event_date(event_date: str | date | None = None) -> str:
    if isinstance(event_date, date):
        return event_date.isoformat()
    value = str(event_date or "").strip()
    if not value:
        return today_iso()
    try:
        return date.fromisoformat(value[:10]).isoformat()
    except ValueError as exc:
        raise ValueError("event_date must use YYYY-MM-DD format") from exc


def normalize_category(category: str) -> str:
    resolved = (category or "").strip().upper()
    if resolved not in CATEGORIES:
        raise ValueError(f"category must be one of: {', '.join(CATEGORIES)}")
    return resolved


def guess_provider(model_id: str) -> str:
    lower = model_id.lower()
    if lower.startswith("cohere."):
        return "cohere"
    if lower.startswith("openai."):
        return "openai"
    if lower.startswith("meta."):
        return "meta"
    if lower.startswith("google."):
        return "google"
    return "generic"


def service_endpoint(region: str) -> str:
    return f"https://inference.generativeai.{region}.oci.oraclecloud.com"


def llm_kwargs(model_id: str, max_tokens: int, temperature: float) -> dict[str, Any]:
    if model_id.startswith("openai."):
        return {"temperature": temperature, "max_completion_tokens": max_tokens}
    return {"temperature": temperature, "max_tokens": max_tokens}


def format_memory_content(
    *,
    category: str,
    content: str,
    title: str = "",
    account: str = "",
    product: str = "",
    source: str = "",
    tags: str = "",
    event_date: str | date | None = None,
) -> str:
    resolved_event_date = normalize_event_date(event_date)
    header = [
        f"Category: {category}",
        f"Title: {title or 'Untitled'}",
        f"Event date: {resolved_event_date}",
    ]
    if account:
        header.append(f"Account: {account}")
    if product:
        header.append(f"Product: {product}")
    if source:
        header.append(f"Source: {source}")
    if tags:
        header.append(f"Tags: {tags}")
    header.append(f"Captured: {utc_now()}")
    return "\n".join(header) + "\n\n" + content.strip()


def build_metadata(
    *,
    category: str,
    title: str = "",
    account: str = "",
    product: str = "",
    source: str = "",
    tags: str = "",
    event_date: str | date | None = None,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
) -> dict[str, Any]:
    resolved_event_date = normalize_event_date(event_date)
    metadata: dict[str, Any] = {
        "category": category,
        "title": title,
        "account": account,
        "product": product,
        "source": source,
        "event_date": resolved_event_date,
        "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
        "created_at": utc_now(),
    }
    if chunk_index is not None:
        metadata["chunk_index"] = chunk_index
    if chunk_count is not None:
        metadata["chunk_count"] = chunk_count
    return metadata


def result_to_dict(result: Any) -> dict[str, Any]:
    record = getattr(result, "record", result)
    metadata = getattr(record, "metadata", None) or {}
    return {
        "id": getattr(record, "id", ""),
        "type": getattr(record, "record_type", ""),
        "category": metadata.get("category", ""),
        "title": metadata.get("title", ""),
        "event_date": metadata.get("event_date", ""),
        "account": metadata.get("account", ""),
        "product": metadata.get("product", ""),
        "source": metadata.get("source", ""),
        "content": getattr(result, "content", None) or getattr(record, "content", ""),
    }


def filter_results(
    results: list[Any],
    *,
    category: str = "ANY",
    account: str = "",
    product: str = "",
    start_date: str = "",
    end_date: str = "",
) -> list[dict[str, Any]]:
    filtered = []
    category = (category or "ANY").upper()
    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None
    for result in results:
        item = result_to_dict(result)
        if category != "ANY" and item["category"] != category:
            continue
        if account and account.lower() not in str(item["account"]).lower():
            continue
        if product and product.lower() not in str(item["product"]).lower():
            continue
        if start or end:
            item_date_raw = item.get("event_date") or ""
            if not item_date_raw:
                continue
            try:
                item_date = date.fromisoformat(str(item_date_raw)[:10])
            except ValueError:
                continue
            if start and item_date < start:
                continue
            if end and item_date > end:
                continue
        filtered.append(item)
    return filtered


def render_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No matching engagement knowledge found."
    blocks = []
    for index, result in enumerate(results, start=1):
        content = textwrap.shorten(result["content"].replace("\n", " "), width=1200)
        blocks.append(
            "\n".join(
                [
                    f"{index}. id={result['id']}",
                    f"category={result['category']} title={result['title']}",
                    f"event_date={result['event_date']}",
                    f"account={result['account']} product={result['product']}",
                    f"source={result['source']}",
                    f"content={content}",
                ]
            )
        )
    return "\n\n".join(blocks)


def search_memory_context(
    memory: OracleAgentMemory,
    query: str,
    *,
    max_results: int = 6,
) -> list[dict[str, Any]]:
    raw_results = memory.search(
        query,
        user_id=USER_ID,
        agent_id=AGENT_ID,
        max_results=max(max_results * 5, 24),
        record_types=["memory"],
    )
    return filter_results(raw_results)[:max_results]


def format_context_for_prompt(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No HiveMind memory records were retrieved for this query."
    blocks = []
    for index, result in enumerate(results, start=1):
        content = result["content"].strip()
        blocks.append(
            "\n".join(
                [
                    f"[{index}] id={result['id']}",
                    f"category={result['category']} title={result['title']} event_date={result['event_date']}",
                    f"account={result['account']} product={result['product']} source={result['source']}",
                    "content:",
                    content[:2400],
                ]
            )
        )
    return "\n\n".join(blocks)


def wants_memory_write(prompt: str) -> bool:
    lowered = prompt.lower()
    write_phrases = (
        "remember this",
        "save this",
        "store this",
        "add this",
        "capture this",
        "note this",
        "update memory",
        "update the knowledge base",
        "add to knowledge base",
        "delete memory",
        "delete memories",
        "delete note",
        "remove memory",
        "remove note",
        "forget this",
    )
    return any(phrase in lowered for phrase in write_phrases)


def extract_memory_ids(prompt: str) -> list[str]:
    pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    return list(dict.fromkeys(re.findall(pattern, prompt, flags=re.IGNORECASE)))


def is_delete_request(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(word in lowered for word in ("delete", "remove", "forget"))


def delete_memories_by_id(memory: OracleAgentMemory, memory_ids: list[str]) -> str:
    if not memory_ids:
        return "I could not find any memory ids to delete."
    results = []
    for memory_id in memory_ids:
        row_count = memory._store.delete("memory", memory_id)
        if row_count:
            results.append(f"Deleted memory id={memory_id}.")
        else:
            results.append(f"No memory record found for id={memory_id}.")
    return "\n".join(results)


@st.cache_resource(show_spinner=False)
def get_memory() -> OracleAgentMemory:
    cfg = from_file(str(Path(OCI_CONFIG_FILE).expanduser()), OCI_PROFILE)
    key_file = str(Path(cfg["key_file"]).expanduser())
    pool = oracledb.create_pool(
        user=DB_USER,
        password=DB_PASSWORD,
        dsn=CONNECT_STRING,
    )
    embedder = Embedder(
        model=EMBEDDING_MODEL,
        oci_compartment_id=cfg["compartment_id"],
        oci_region=cfg["region"],
        oci_user=cfg["user"],
        oci_fingerprint=cfg["fingerprint"],
        oci_tenancy=cfg["tenancy"],
        oci_key_file=key_file,
    )
    return OracleAgentMemory(
        connection=pool,
        embedder=embedder,
        schema_policy=SchemaPolicy.CREATE_IF_NECESSARY,
        table_name_prefix=TABLE_PREFIX,
    )


@st.cache_resource(show_spinner=False)
def get_chat_model() -> ChatOCIGenAI:
    cfg = from_file(str(Path(OCI_CONFIG_FILE).expanduser()), OCI_PROFILE)
    return ChatOCIGenAI(
        model_id=CHAT_MODEL,
        provider=guess_provider(CHAT_MODEL),
        service_endpoint=service_endpoint(cfg["region"]),
        compartment_id=cfg["compartment_id"],
        auth_type="API_KEY",
        auth_profile=OCI_PROFILE,
        auth_file_location=str(Path(OCI_CONFIG_FILE).expanduser()),
        model_kwargs=llm_kwargs(CHAT_MODEL, MAX_RESPONSE_TOKENS, TEMPERATURE),
        max_sequential_tool_calls=6,
    )


@st.cache_data(ttl=30, show_spinner=False)
def check_database_status() -> tuple[bool, str]:
    try:
        with oracledb.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            dsn=CONNECT_STRING,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 FROM DUAL")
                cursor.fetchone()
        return True, "Database active"
    except Exception as exc:
        return False, str(exc)


def api_port_open() -> bool:
    try:
        with socket.create_connection((API_HOST, API_PORT), timeout=0.35):
            return True
    except OSError:
        return False


def api_http_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{API_URL}/openapi.json", timeout=0.75) as response:
            return response.status == 200
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def ensure_api_server() -> dict[str, Any]:
    if api_http_ready():
        return {
            "ok": True,
            "started": False,
            "message": f"API already running at {API_URL}",
            "url": API_URL,
        }

    if api_port_open():
        return {
            "ok": False,
            "started": False,
            "message": f"Port {API_PORT} is already in use by another process.",
            "url": API_URL,
        }

    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / "hivemind_api.log"
    log_file = log_path.open("a", encoding="utf-8")
    log_file.write(f"\n\n[{utc_now()}] Starting HiveMind AI API\n")
    log_file.flush()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "engagement_tracker_api:app",
            "--host",
            API_HOST,
            "--port",
            str(API_PORT),
        ],
        cwd=Path(__file__).resolve().parent,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def stop_api() -> None:
        if process.poll() is None:
            process.terminate()

    atexit.register(stop_api)

    for _ in range(30):
        if process.poll() is not None:
            return {
                "ok": False,
                "started": True,
                "message": f"API failed to start. See {log_path}.",
                "url": API_URL,
                "log_path": str(log_path),
            }
        if api_http_ready():
            return {
                "ok": True,
                "started": True,
                "message": f"API running at {API_URL}",
                "url": API_URL,
                "pid": process.pid,
                "log_path": str(log_path),
            }
        time.sleep(0.2)

    return {
        "ok": False,
        "started": True,
        "message": f"API did not become ready yet. See {log_path}.",
        "url": API_URL,
        "pid": process.pid,
        "log_path": str(log_path),
    }


def build_tools(memory: OracleAgentMemory, user_id: str, agent_id: str):
    @tool
    def add_engagement_note(
        category: str,
        content: str,
        title: str = "",
        account: str = "",
        product: str = "",
        source: str = "chat",
        tags: str = "",
        event_date: str = "",
    ) -> str:
        """Add durable engagement knowledge. Use ACCOUNT for customers, ORG for internal team knowledge, and PRODUCT for product knowledge. event_date is YYYY-MM-DD."""
        resolved_category = normalize_category(category)
        resolved_event_date = normalize_event_date(event_date)
        metadata = build_metadata(
            category=resolved_category,
            title=title,
            account=account,
            product=product,
            source=source,
            tags=tags,
            event_date=resolved_event_date,
        )
        formatted = format_memory_content(
            category=resolved_category,
            content=content,
            title=title,
            account=account,
            product=product,
            source=source,
            tags=tags,
            event_date=resolved_event_date,
        )
        memory_id = memory.add_memory(
            formatted,
            user_id=user_id,
            agent_id=agent_id,
            metadata=metadata,
        )
        return f"Stored {resolved_category} note with id={memory_id}."

    @tool
    def search_engagement_knowledge(
        query: str,
        category: str = "ANY",
        account: str = "",
        product: str = "",
        start_date: str = "",
        end_date: str = "",
        max_results: int = 5,
    ) -> str:
        """Search the engagement knowledge base by semantic query, category, account, product, and optional YYYY-MM-DD date range."""
        raw_results = memory.search(
            query,
            user_id=user_id,
            agent_id=agent_id,
            max_results=max(max_results * 4, 12),
            record_types=["memory"],
        )
        filtered = filter_results(
            raw_results,
            category=category,
            account=account,
            product=product,
            start_date=start_date,
            end_date=end_date,
        )
        return render_results(filtered[:max_results])

    @tool
    def update_engagement_note(
        memory_id: str,
        replacement_content: str,
        category: str = "ANY",
        title: str = "",
        account: str = "",
        product: str = "",
        source: str = "chat-update",
        tags: str = "",
        event_date: str = "",
    ) -> str:
        """Update an existing memory record by id. Search first if the id is unknown. event_date is YYYY-MM-DD."""
        existing = memory._store.get("memory", memory_id)
        if existing is None:
            return f"No memory record found for id={memory_id}."
        old_metadata = getattr(existing, "metadata", None) or {}
        resolved_category = (
            old_metadata.get("category", "PRODUCT")
            if category == "ANY"
            else normalize_category(category)
        )
        metadata = {
            **old_metadata,
            **build_metadata(
                category=resolved_category,
                title=title or old_metadata.get("title", ""),
                account=account or old_metadata.get("account", ""),
                product=product or old_metadata.get("product", ""),
                source=source or old_metadata.get("source", ""),
                tags=tags or ",".join(old_metadata.get("tags", [])),
                event_date=event_date or old_metadata.get("event_date", today_iso()),
            ),
            "updated_at": utc_now(),
        }
        formatted = format_memory_content(
            category=metadata["category"],
            content=replacement_content,
            title=metadata.get("title", ""),
            account=metadata.get("account", ""),
            product=metadata.get("product", ""),
            source=metadata.get("source", ""),
            tags=",".join(metadata.get("tags", [])),
            event_date=metadata.get("event_date", today_iso()),
        )
        row_count = memory._store.update("memory", memory_id, text=formatted, metadata=metadata)
        return f"Updated id={memory_id}." if row_count else f"No rows updated for id={memory_id}."

    @tool
    def delete_engagement_note(memory_ids: list[str]) -> str:
        """Delete one or more memory records by id. Use only when the user explicitly asks to delete/remove/forget memory ids."""
        if not memory_ids:
            return "No memory ids were provided."
        results = []
        for memory_id in memory_ids:
            row_count = memory._store.delete("memory", memory_id)
            if row_count:
                results.append(f"Deleted id={memory_id}.")
            else:
                results.append(f"No memory record found for id={memory_id}.")
        return "\n".join(results)

    return [
        add_engagement_note,
        search_engagement_knowledge,
        update_engagement_note,
        delete_engagement_note,
    ]


def build_read_only_tools(memory: OracleAgentMemory, user_id: str, agent_id: str):
    @tool
    def search_engagement_knowledge(
        query: str,
        category: str = "ANY",
        account: str = "",
        product: str = "",
        start_date: str = "",
        end_date: str = "",
        max_results: int = 5,
    ) -> str:
        """Search the engagement knowledge base by semantic query, category, account, product, and optional YYYY-MM-DD date range."""
        raw_results = memory.search(
            query,
            user_id=user_id,
            agent_id=agent_id,
            max_results=max(max_results * 4, 12),
            record_types=["memory"],
        )
        filtered = filter_results(
            raw_results,
            category=category,
            account=account,
            product=product,
            start_date=start_date,
            end_date=end_date,
        )
        return render_results(filtered[:max_results])

    return [search_engagement_knowledge]


SYSTEM_PROMPT = """You are an Engagement Tracker agent for a technical pre-sales Black Belt specialist in EMEA.
Your job is to capture and retrieve useful working knowledge across three categories:

- ACCOUNT: customer meetings, account context, org charts, stakeholders, behavior, opportunities, risks.
- ORG: team and internal organization knowledge, sales plays, internal network, operating model.
- PRODUCT: product tips, implementation notes, debugging findings, blogs, guides, SDK references.

For every answer, treat the "Retrieved HiveMind memory" section as the highest priority source.
If retrieved memory is relevant, answer from it and cite memory ids inline.
If retrieved memory conflicts with your general model knowledge, prefer retrieved memory.
If no relevant memory was retrieved, say that HiveMind has no matching note before giving general guidance.
Do not invent filenames, scripts, grants, or setup steps unless they appear in retrieved memory or the user supplied them.
Only add or update memory when the user explicitly asks to remember, save, store, capture, update, delete, or forget something.
Never add or update memory just because you answered a question.
Always preserve event dates when the user gives one. If they ask about a month or year, search with an appropriate date range.
When adding notes, ask for missing category/account/product only if the category cannot be inferred.
Keep answers concise, practical, and oriented toward future reuse.
"""


def extract_ai_text(agent_response: dict[str, Any]) -> str:
    messages = agent_response.get("messages", [])
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if content and getattr(message, "type", "") == "ai":
            return str(content)
    if messages:
        return str(getattr(messages[-1], "content", messages[-1]))
    return "I did not receive a response from the agent."


def strip_html(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    return parser.text()


def decode_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_uploaded_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".html", ".htm"}:
        return strip_html(decode_bytes(data))
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF upload requires `pip install pypdf`.") from exc
        import io

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("DOCX upload requires `pip install python-docx`.") from exc
        import io

        doc = Document(io.BytesIO(data))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    return decode_bytes(data)


def fetch_url_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "engagement-tracker/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
        content_type = response.headers.get("content-type", "")
    text = decode_bytes(body)
    if "html" in content_type.lower() or url.lower().endswith((".html", ".htm")):
        return strip_html(text)
    return text


def chunk_text(text: str, chunk_size: int = 4500, overlap: int = 300) -> list[str]:
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = "\n".join(part for part in text.splitlines() if part.strip())
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        boundary = text.rfind("\n", start, end)
        if boundary > start + chunk_size // 2:
            end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def store_document_chunks(
    memory: OracleAgentMemory,
    *,
    text: str,
    category: str,
    title: str,
    account: str,
    product: str,
    source: str,
    tags: str,
    event_date: str | date,
    user_id: str,
    agent_id: str,
) -> list[str]:
    resolved_category = normalize_category(category)
    resolved_event_date = normalize_event_date(event_date)
    chunks = chunk_text(text)
    ids = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_title = title if len(chunks) == 1 else f"{title} ({index}/{len(chunks)})"
        metadata = build_metadata(
            category=resolved_category,
            title=chunk_title,
            account=account,
            product=product,
            source=source,
            tags=tags,
            event_date=resolved_event_date,
            chunk_index=index,
            chunk_count=len(chunks),
        )
        formatted = format_memory_content(
            category=resolved_category,
            content=chunk,
            title=chunk_title,
            account=account,
            product=product,
            source=source,
            tags=tags,
            event_date=resolved_event_date,
        )
        ids.append(
            memory.add_memory(
                formatted,
                user_id=user_id,
                agent_id=agent_id,
                metadata=metadata,
            )
        )
    return ids


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ready. Tell me what to remember, or ask what we know about an account, product, or internal topic.",
            }
        ]


def main() -> None:
    st.set_page_config(page_title="Engagement Tracker", layout="centered")
    init_state()

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 920px;
            padding-top: 2.4rem;
        }
        div[data-testid="stToolbar"] {
            visibility: hidden;
        }
        .hivemind-title {
            font-size: clamp(2.25rem, 7vw, 4.75rem);
            line-height: 0.95;
            font-weight: 800;
            color: #f5f5f5;
            margin-top: 1.4rem;
            white-space: nowrap;
        }
        .hivemind-rule {
            width: min(19rem, 70%);
            height: 0.45rem;
            background: #d60020;
            margin: 1rem 0 1.2rem 0;
        }
        .hivemind-subtitle {
            color: #d4d4d4;
            font-size: clamp(1rem, 2.2vw, 1.35rem);
            line-height: 1.35;
            margin-bottom: 1.2rem;
        }
        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border: 1px solid #333;
            border-radius: 999px;
            padding: 0.28rem 0.75rem;
            font-size: 0.86rem;
            margin: 0.5rem 0 1rem 0;
            background: #101010;
        }
        .status-ok {
            color: #2fb344;
            font-weight: 700;
        }
        .status-bad {
            color: #e03131;
            font-weight: 700;
        }
        .service-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    logo_col, title_col = st.columns([1, 2.35], vertical_alignment="center")
    with logo_col:
        st.image(str(ICON_PATH), use_container_width=True)
    with title_col:
        st.markdown(
            """
            <div class="hivemind-title">HiveMind AI</div>
            <div class="hivemind-rule"></div>
            <div class="hivemind-subtitle">Engagement memory for accounts, orgs, and products</div>
            """,
            unsafe_allow_html=True,
        )

    db_ok, db_status_text = check_database_status()
    db_status_icon = "✓" if db_ok else "✕"
    db_status_class = "status-ok" if db_ok else "status-bad"
    st.markdown(
        '<div class="service-row">'
        f'<div class="status-chip"><span class="{db_status_class}">{db_status_icon}</span>'
        f'<span>{db_status_text if db_ok else "Database unavailable"}</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if not db_ok:
        st.error(db_status_text)
        st.stop()

    api_status = ensure_api_server()
    api_status_icon = "✓" if api_status["ok"] else "✕"
    api_status_class = "status-ok" if api_status["ok"] else "status-bad"
    st.markdown(
        '<div class="service-row">'
        f'<div class="status-chip"><span class="{api_status_class}">{api_status_icon}</span>'
        f'<span>{api_status["message"]}</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if not api_status["ok"]:
        st.warning(api_status["message"])

    try:
        memory = get_memory()
        chat_model = get_chat_model()
    except Exception as exc:
        st.error(f"Connection setup failed: {exc}")
        st.stop()

    with st.expander("Add file or URL to the knowledge base"):
        with st.form("knowledge_entry_form", clear_on_submit=True):
            upload_category = st.segmented_control(
                "Category",
                CATEGORIES,
                default="PRODUCT",
            )
            upload_title = st.text_input("Title", placeholder="SDK reference, PoC note, customer meeting...")
            upload_event_date = st.date_input("Event date", value=date.today())
            col1, col2 = st.columns(2)
            upload_account = col1.text_input("Account", placeholder="Customer account, if relevant")
            upload_product = col2.text_input("Product", placeholder="Product area, if relevant")
            upload_tags = st.text_input("Tags", placeholder="vector search, sdk, debugging")
            note_text = st.text_area(
                "Markdown note",
                placeholder="Paste or write meeting notes, PoC findings, product tips, or markdown snippets here.",
                height=180,
            )
            uploaded_file = st.file_uploader("File", type=None)
            source_url = st.text_input("URL")
            submitted = st.form_submit_button("Process and Store", type="primary")

        if submitted:
            try:
                if not note_text.strip() and uploaded_file is None and not source_url:
                    st.warning("Add a markdown note, upload a file, or provide a URL.")
                else:
                    if note_text.strip():
                        raw_text = note_text
                        source = "manual-note"
                        title = upload_title or "Manual note"
                    elif uploaded_file is not None:
                        raw_text = extract_uploaded_text(uploaded_file.name, uploaded_file.getvalue())
                        source = uploaded_file.name
                        title = upload_title or Path(uploaded_file.name).stem
                    else:
                        raw_text = fetch_url_text(source_url)
                        source = source_url
                        title = upload_title or source_url

                    ids = store_document_chunks(
                        memory,
                        text=raw_text,
                        category=upload_category,
                        title=title,
                        account=upload_account,
                        product=upload_product,
                        source=source,
                        tags=upload_tags,
                        event_date=upload_event_date,
                        user_id=USER_ID,
                        agent_id=AGENT_ID,
                    )
                    st.success(f"Stored {len(ids)} chunk(s). First id: {ids[0]}")
            except Exception as exc:
                st.error(f"Upload processing failed: {exc}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Add notes, search accounts/products, or ask what you know..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking with your engagement memory..."):
                try:
                    delete_ids = extract_memory_ids(prompt) if is_delete_request(prompt) else []
                    if delete_ids:
                        answer = delete_memories_by_id(memory, delete_ids)
                    else:
                        context_results = search_memory_context(memory, prompt)
                        context_message = (
                            "Retrieved HiveMind memory for the user's latest query:\n\n"
                            f"{format_context_for_prompt(context_results)}"
                        )
                        toolset = (
                            build_tools(memory, USER_ID, AGENT_ID)
                            if wants_memory_write(prompt)
                            else build_read_only_tools(memory, USER_ID, AGENT_ID)
                        )
                        agent = create_agent(
                            model=chat_model,
                            tools=toolset,
                            system_prompt=SYSTEM_PROMPT,
                            name="engagement_tracker_agent",
                        )
                        conversation_history = st.session_state.messages[-8:-1]
                        augmented_user_message = {
                            "role": "user",
                            "content": (
                                f"{context_message}\n\n"
                                "User question:\n"
                                f"{prompt}"
                            ),
                        }
                        messages = conversation_history + [augmented_user_message]
                        response = agent.invoke({"messages": messages})
                        answer = extract_ai_text(response)
                except Exception as exc:
                    answer = f"Agent call failed: {exc}"
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.expander("Manual Search"):
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        query = col1.text_input("Search query")
        category = col2.selectbox("Search category", ("ANY",) + CATEGORIES)
        account = col3.text_input("Filter account")
        product = col4.text_input("Filter product")
        date_col1, date_col2 = st.columns(2)
        start_date = date_col1.date_input("From date", value=None)
        end_date = date_col2.date_input("To date", value=None)
        if st.button("Search Knowledge"):
            raw_results = memory.search(
                query,
                user_id=USER_ID,
                agent_id=AGENT_ID,
                max_results=24,
                record_types=["memory"],
            )
            filtered = filter_results(
                raw_results,
                category=category,
                account=account,
                product=product,
                start_date=start_date.isoformat() if start_date else "",
                end_date=end_date.isoformat() if end_date else "",
            )
            if not filtered:
                st.write("No matching results.")
            for item in filtered[:8]:
                st.markdown(f"**{item['title'] or item['id']}**")
                st.caption(
                    f"{item['category']} | account={item['account'] or '-'} | "
                    f"product={item['product'] or '-'} | date={item['event_date'] or '-'} | id={item['id']}"
                )
                st.write(item["content"])


if __name__ == "__main__":
    main()
