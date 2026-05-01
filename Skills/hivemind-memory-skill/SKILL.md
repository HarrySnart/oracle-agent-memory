---
name: hivemind-memory-skill
description: Use this skill when debugging, building, or reviewing code and you discover reusable product knowledge, customer/account context, org knowledge, implementation notes, SDK gotchas, or troubleshooting fixes that should be saved to the local HiveMind AI knowledge base via its HTTP API.
---

# HiveMind Memory Skill

Use HiveMind AI as a personal durable knowledge base while working in Cline or Codex. Capture things that will help future debugging, PoCs, customer work, or product research.

## When To Use

Use this skill when:

- A debugging session reveals a useful fix, workaround, error pattern, version issue, setup step, or SDK behavior.
- A script or test teaches something reusable about an Oracle product, API, model, database feature, or integration.
- The user asks to remember, store, save, search, list, or delete HiveMind knowledge.
- You need prior notes about an account, product, internal org topic, or previous troubleshooting trail.

Default category guidance:

- `PRODUCT`: product docs, SDK notes, PoC/debugging findings, blogs, guides, code patterns.
- `ACCOUNT`: customer interactions, meeting notes, stakeholders, account context.
- `ORG`: team knowledge, sales plays, internal network, operating model.

For Cline/Codex debugging work, default to `PRODUCT` unless the note is clearly about an account or organization.

## API Assumptions

HiveMind API should be running locally, usually started automatically by the Streamlit app:

```text
http://127.0.0.1:8000
```

Health check:

```bash
python Skills/hivemind-memory-skill/scripts/hivemind_client.py health
```

If the API is down, start HiveMind AI from your local project checkout:

```bash
cd /path/to/oracle-agent-memory
source .venv/bin/activate
streamlit run engagement_tracker.py
```

## Capture Workflow

When saving a note from debugging:

1. Summarize the reusable lesson, not the whole raw transcript.
2. Include exact error text, package versions, model names, endpoints, command names, or file paths when useful.
3. Add what fixed it and how to verify it.
4. Use `event_date` as today unless the user provides a historical date.
5. Use tags such as `debugging`, `sdk`, `oci`, `streamlit`, `oracle-agent-memory`, `database`, or the product name.

Good note shape:

```text
Problem: ...
Context: ...
Root cause: ...
Fix: ...
Verification: ...
Useful commands/files: ...
```

## Commands

The bundled client script uses only the Python standard library.

### Add Knowledge

```bash
python Skills/hivemind-memory-skill/scripts/hivemind_client.py add \
  --category PRODUCT \
  --title "OCI GenAI embedding setup for Oracle Agent Memory" \
  --event-date 2026-05-01 \
  --product "Oracle Agent Memory" \
  --tags "debugging,oci,embedding" \
  --content "Problem: ... Fix: ... Verification: ..."
```

### Search Knowledge

```bash
python Skills/hivemind-memory-skill/scripts/hivemind_client.py search \
  --query "OCI GenAI embedding schema_policy" \
  --category PRODUCT \
  --limit 5
```

Use date filters for time-specific questions:

```bash
python Skills/hivemind-memory-skill/scripts/hivemind_client.py search \
  --query "Agent Memory" \
  --start-date 2026-05-01 \
  --end-date 2026-05-31
```

### List Memories

```bash
python Skills/hivemind-memory-skill/scripts/hivemind_client.py list --limit 20
```

### Delete Memory

```bash
python Skills/hivemind-memory-skill/scripts/hivemind_client.py delete --id "<memory-id>"
```

## Endpoint Reference

- `GET /health`
- `POST /memories`
- `GET /memories`
- `GET /search?q=<term>`
- `POST /memories/delete`

Payload fields for `POST /memories`:

- `category`: `ACCOUNT`, `ORG`, or `PRODUCT`
- `title`
- `event_date`: `YYYY-MM-DD`
- `account`
- `product`
- `source`
- `tags`: list of strings
- `content`: markdown/plain text

## Response Handling

When a save succeeds, report the memory id briefly. When a search returns useful context, cite the ids in your response so the user can update or delete them later.
