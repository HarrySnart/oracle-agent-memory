from __future__ import annotations

import os
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import oracledb
from fastapi import FastAPI, HTTPException, Query
from oci.config import from_file
from oracleagentmemory.core import OracleAgentMemory
from oracleagentmemory.core.dbschemapolicy import SchemaPolicy
from oracleagentmemory.core.embedders.embedder import Embedder
from pydantic import BaseModel, Field


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


Category = Literal["ACCOUNT", "ORG", "PRODUCT"]


class MemoryCreate(BaseModel):
    content: str = Field(..., min_length=1)
    category: Category
    title: str = ""
    event_date: date | None = None
    account: str = ""
    product: str = ""
    source: str = "api"
    tags: list[str] = Field(default_factory=list)


class MemoryDelete(BaseModel):
    id: str = Field(..., min_length=1)


class MemoryRecord(BaseModel):
    id: str
    type: str
    category: str = ""
    title: str = ""
    event_date: str = ""
    account: str = ""
    product: str = ""
    source: str = ""
    tags: list[str] = Field(default_factory=list)
    content: str = ""


class CreateResponse(BaseModel):
    id: str


class DeleteResponse(BaseModel):
    id: str
    deleted: bool
    row_count: int


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
        raise HTTPException(status_code=422, detail="event_date must use YYYY-MM-DD format") from exc


def format_memory_content(
    *,
    category: str,
    content: str,
    title: str = "",
    account: str = "",
    product: str = "",
    source: str = "",
    tags: list[str] | None = None,
    event_date: str | date | None = None,
) -> str:
    resolved_event_date = normalize_event_date(event_date)
    tag_text = ", ".join(tags or [])
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
    if tag_text:
        header.append(f"Tags: {tag_text}")
    header.append(f"Captured: {utc_now()}")
    return "\n".join(header) + "\n\n" + content.strip()


def build_metadata(item: MemoryCreate) -> dict[str, Any]:
    return {
        "category": item.category,
        "title": item.title,
        "account": item.account,
        "product": item.product,
        "source": item.source,
        "event_date": normalize_event_date(item.event_date),
        "tags": item.tags,
        "created_at": utc_now(),
    }


def record_to_response(record: Any) -> MemoryRecord:
    metadata = getattr(record, "metadata", None) or {}
    return MemoryRecord(
        id=getattr(record, "id", ""),
        type=getattr(record, "record_type", ""),
        category=metadata.get("category", ""),
        title=metadata.get("title", ""),
        event_date=metadata.get("event_date", ""),
        account=metadata.get("account", ""),
        product=metadata.get("product", ""),
        source=metadata.get("source", ""),
        tags=metadata.get("tags", []) or [],
        content=getattr(record, "content", "") or "",
    )


def result_to_response(result: Any) -> MemoryRecord:
    record = getattr(result, "record", result)
    response = record_to_response(record)
    content = getattr(result, "content", None)
    if content:
        response.content = content
    return response


def in_date_range(record: MemoryRecord, start_date: date | None, end_date: date | None) -> bool:
    if not start_date and not end_date:
        return True
    if not record.event_date:
        return False
    try:
        event_date = date.fromisoformat(record.event_date[:10])
    except ValueError:
        return False
    if start_date and event_date < start_date:
        return False
    if end_date and event_date > end_date:
        return False
    return True


@lru_cache(maxsize=1)
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


app = FastAPI(title="HiveMind AI Knowledge API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    try:
        memory = get_memory()
        memory._store.list("memory", limit=1, user_id=USER_ID, agent_id=AGENT_ID)
        return {"status": "ok"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/memories", response_model=CreateResponse)
def add_memory(item: MemoryCreate) -> CreateResponse:
    try:
        memory = get_memory()
        content = format_memory_content(
            category=item.category,
            content=item.content,
            title=item.title,
            account=item.account,
            product=item.product,
            source=item.source,
            tags=item.tags,
            event_date=item.event_date,
        )
        memory_id = memory.add_memory(
            content,
            user_id=USER_ID,
            agent_id=AGENT_ID,
            metadata=build_metadata(item),
        )
        return CreateResponse(id=memory_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/memories", response_model=list[MemoryRecord])
def list_memories(
    limit: int = Query(100, ge=1, le=1000),
    category: str = "ANY",
    account: str = "",
    product: str = "",
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[MemoryRecord]:
    try:
        memory = get_memory()
        records = memory._store.list("memory", limit=limit, user_id=USER_ID, agent_id=AGENT_ID)
        response = []
        for record in records:
            item = record_to_response(record)
            if category != "ANY" and item.category != category:
                continue
            if account and account.lower() not in item.account.lower():
                continue
            if product and product.lower() not in item.product.lower():
                continue
            if not in_date_range(item, start_date, end_date):
                continue
            response.append(item)
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/search", response_model=list[MemoryRecord])
def search_memories(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=100),
    category: str = "ANY",
    account: str = "",
    product: str = "",
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[MemoryRecord]:
    try:
        memory = get_memory()
        results = memory.search(
            q,
            user_id=USER_ID,
            agent_id=AGENT_ID,
            max_results=max(limit * 4, 12),
            record_types=["memory"],
        )
        response = []
        for result in results:
            item = result_to_response(result)
            if category != "ANY" and item.category != category:
                continue
            if account and account.lower() not in item.account.lower():
                continue
            if product and product.lower() not in item.product.lower():
                continue
            if not in_date_range(item, start_date, end_date):
                continue
            response.append(item)
            if len(response) >= limit:
                break
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/memories/delete", response_model=DeleteResponse)
def delete_memory(item: MemoryDelete) -> DeleteResponse:
    try:
        memory = get_memory()
        row_count = memory._store.delete("memory", item.id)
        return DeleteResponse(id=item.id, deleted=row_count > 0, row_count=row_count)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
