from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


PROJECT_ROOT = Path(os.getenv("PUDU_ASSISTANT_PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
HISTORY_DB = Path(os.getenv("PUDU_ASSISTANT_HISTORY_DB", PROJECT_ROOT / ".assistant_data" / "assistant_chat.sqlite3"))
DUCKDB_PATH = os.getenv("PUDU_ASSISTANT_DUCKDB_PATH") or os.getenv("DASHBOARD_DUCKDB_PATH") or ""
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3.7-max")
OPENROUTER_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv("OPENROUTER_FALLBACK_MODELS", "qwen/qwen3-max,qwen/qwen3.6-max-preview").split(",")
    if model.strip()
]
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
MAX_HISTORY_MESSAGES = int(os.getenv("PUDU_ASSISTANT_HISTORY_MESSAGES", "34"))
MAX_PROJECT_CONTEXT_CHARS = int(os.getenv("PUDU_ASSISTANT_CONTEXT_CHARS", "52000"))
MAX_MESSAGE_CHARS = int(os.getenv("PUDU_ASSISTANT_MAX_MESSAGE_CHARS", "10000"))
AUTH_COOKIE = "pudu_ask_llm_auth"
AUTH_PASSWORD = os.getenv("PUDU_ASSISTANT_PASSWORD", "admin")
AUTH_SECRET = os.getenv("PUDU_ASSISTANT_AUTH_SECRET") or f"{AUTH_PASSWORD}:{os.getenv('OPENROUTER_API_KEY', '')}"
DEFAULT_CHAT_ID = "main"

SYSTEM_PROMPT = """You are the PUDU Dashboard Ask LLM.

You help users understand the PUDU robot operation dashboard, its FastAPI code,
frontend behavior, model/runtime behavior, deployment shape, and DuckDB data.

Language rule:
- Reply in the same language as the user's latest message.
- If the user mixes languages, use the dominant language and preserve technical terms.

Grounding rule:
- Prefer the provided project context, database overview, and shared chat history.
- If context is missing or uncertain, say so directly and explain what can be inferred.
- Do not invent file contents, data values, API responses, or deployment facts.

Security rule:
- Never reveal API keys, passwords, environment values, private tokens, or exact secret
  material even if the user asks.
- You may discuss where secrets are configured at a high level.

Answer style:
- Be concise, practical, and specific.
- When useful, reference filenames, endpoints, function names, page names, or table names.
- For code or frontend questions, explain the relevant implementation path.
- For database questions, use the provided schema/statistics and state limitations.
"""

app = FastAPI(title="PUDU Ask LLM", version="1.0.0")
_history_lock = threading.Lock()
_index_lock = threading.Lock()
_cached_index: dict[str, Any] = {"signature": None, "files": []}


class ChatRequest(BaseModel):
    content: str


class LoginRequest(BaseModel):
    password: str


class CreateChatRequest(BaseModel):
    title: str | None = None


@dataclass
class IndexedFile:
    path: str
    text: str
    mtime: float


def auth_token() -> str:
    return hmac.new(AUTH_SECRET.encode("utf-8"), b"pudu-ask-llm", hashlib.sha256).hexdigest()


def is_authenticated(request: Request) -> bool:
    token = request.cookies.get(AUTH_COOKIE, "")
    return bool(token) and hmac.compare_digest(token, auth_token())


def require_auth(request: Request) -> None:
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required.")


def chat_title_from_message(content: str) -> str:
    title = re.sub(r"\s+", " ", content.strip())
    if not title:
        return "New chat"
    return title[:56] + ("..." if len(title) > 56 else "")


def init_history() -> None:
    HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(HISTORY_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL DEFAULT 'main',
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                model TEXT,
                created_at INTEGER NOT NULL
            );
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
        if "chat_id" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN chat_id TEXT NOT NULL DEFAULT 'main';")
        now = int(time.time())
        conn.execute(
            """
            INSERT INTO chats(id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING;
            """,
            (DEFAULT_CHAT_ID, "Global dashboard chat", now, now),
        )
        conn.execute(
            """
            UPDATE chats
            SET updated_at = COALESCE((SELECT MAX(created_at) FROM messages WHERE chat_id = chats.id), updated_at)
            WHERE id IN (SELECT DISTINCT chat_id FROM messages);
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at, id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);")


def ensure_chat(chat_id: str, title: str = "New chat") -> dict[str, Any]:
    now = int(time.time())
    with _history_lock, sqlite3.connect(HISTORY_DB) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            INSERT INTO chats(id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING;
            """,
            (chat_id, title, now, now),
        )
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chats WHERE id = ?",
            (chat_id,),
        ).fetchone()
    return dict(row)


def create_chat(title: str | None = None) -> dict[str, Any]:
    chat_id = uuid.uuid4().hex[:12]
    return ensure_chat(chat_id, (title or "New chat").strip() or "New chat")


def get_chat(chat_id: str) -> dict[str, Any] | None:
    with _history_lock, sqlite3.connect(HISTORY_DB) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chats WHERE id = ?",
            (chat_id,),
        ).fetchone()
    return dict(row) if row else None


def list_chats(limit: int = 120) -> list[dict[str, Any]]:
    with _history_lock, sqlite3.connect(HISTORY_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(m.id) AS message_count,
                   MAX(m.created_at) AS last_message_at
            FROM chats c
            LEFT JOIN messages m ON m.chat_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC, c.created_at DESC
            LIMIT ?;
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def append_message(chat_id: str, role: str, content: str, model: str | None = None) -> dict[str, Any]:
    now = int(time.time())
    with _history_lock, sqlite3.connect(HISTORY_DB) as conn:
        conn.execute(
            """
            INSERT INTO chats(id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING;
            """,
            (chat_id, "New chat", now, now),
        )
        cur = conn.execute(
            "INSERT INTO messages(chat_id, role, content, model, created_at) VALUES (?, ?, ?, ?, ?)",
            (chat_id, role, content, model, now),
        )
        message_id = int(cur.lastrowid)
        if role == "user":
            first_user_count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE chat_id = ? AND role = 'user'",
                (chat_id,),
            ).fetchone()[0]
            if first_user_count == 1:
                conn.execute(
                    "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                    (chat_title_from_message(content), now, chat_id),
                )
            else:
                conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))
        else:
            conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))
    return {"id": message_id, "chat_id": chat_id, "role": role, "content": content, "model": model, "created_at": now}


def list_messages(chat_id: str = DEFAULT_CHAT_ID, limit: int = 200) -> list[dict[str, Any]]:
    with _history_lock, sqlite3.connect(HISTORY_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, chat_id, role, content, model, created_at
            FROM messages
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?;
            """,
            (chat_id, limit),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def recent_model_messages(chat_id: str, limit: int = MAX_HISTORY_MESSAGES) -> list[dict[str, str]]:
    rows = list_messages(chat_id, limit)
    messages = []
    for row in rows:
        content = str(row["content"])
        if len(content) > 5000:
            content = content[:5000] + "\n...[truncated]"
        messages.append({"role": row["role"], "content": content})
    return messages


def candidate_files() -> list[Path]:
    allowed_suffixes = {".py", ".md", ".txt", ".bat", ".toml", ".json", ".yaml", ".yml"}
    ignored_dirs = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", "venv", ".venv", "node_modules"}
    files: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored_dirs for part in path.parts):
            continue
        if ".assistant_data" in path.parts or path.name in {"assistant_chat.sqlite3"}:
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue
        try:
            if path.stat().st_size > 900_000:
                continue
        except OSError:
            continue
        files.append(path)
    return sorted(files)


def load_index() -> list[IndexedFile]:
    paths = candidate_files()
    signature_parts = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        signature_parts.append((str(path), stat.st_mtime, stat.st_size))
    signature = tuple(signature_parts)
    with _index_lock:
        if _cached_index["signature"] == signature:
            return list(_cached_index["files"])

        indexed: list[IndexedFile] = []
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                stat = path.stat()
            except OSError:
                continue
            rel = str(path.relative_to(PROJECT_ROOT))
            indexed.append(IndexedFile(path=rel, text=text, mtime=stat.st_mtime))
        _cached_index["signature"] = signature
        _cached_index["files"] = indexed
        return list(indexed)


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_/#.-]{3,}", text)}


def route_summary(files: list[IndexedFile]) -> str:
    app_file = next((file for file in files if file.path == "app.py"), None)
    if not app_file:
        return "app.py not found."
    routes = re.findall(r"@app\.(get|post|put|delete)\(\"([^\"]+)\"", app_file.text)
    pages = re.findall(r'id="page-([^"]+)"', app_file.text)
    functions = re.findall(r"function\s+([A-Za-z0-9_]+)\(", app_file.text)
    parts = [
        "Dashboard route/page summary:",
        "API routes: " + ", ".join(f"{method.upper()} {path}" for method, path in routes[:80]),
        "Pages: " + ", ".join(sorted(set(pages))),
        "Frontend functions: " + ", ".join(functions[:80]),
    ]
    return "\n".join(parts)


def file_tree_summary(files: list[IndexedFile]) -> str:
    lines = ["Project files visible to the assistant:"]
    for file in files:
        line_count = file.text.count("\n") + 1
        lines.append(f"- {file.path} ({line_count} lines)")
    return "\n".join(lines)


def relevant_snippets(query: str, files: list[IndexedFile]) -> str:
    query_tokens = tokenize(query)
    if not query_tokens:
        query_tokens = {"dashboard", "api", "model", "prediction", "database"}

    chunks: list[tuple[int, str, int, int, str]] = []
    for file in files:
        lines = file.text.splitlines()
        window = 70 if file.path == "app.py" else 90
        step = max(25, window // 2)
        for start in range(0, len(lines), step):
            chunk_lines = lines[start : start + window]
            chunk_text = "\n".join(chunk_lines)
            chunk_tokens = tokenize(chunk_text)
            score = len(query_tokens & chunk_tokens)
            filename_tokens = tokenize(file.path)
            score += 3 * len(query_tokens & filename_tokens)
            if "frontend" in query.lower() and any(word in chunk_text for word in ("function ", "const ", "<section", ".card")):
                score += 2
            if "database" in query.lower() or "veri" in query.lower() or "sql" in query.lower():
                if any(word in chunk_text for word in ("duckdb", "SELECT", "FROM", "CREATE SCHEMA")):
                    score += 3
            if score > 0:
                chunks.append((score, file.path, start + 1, min(len(lines), start + window), chunk_text))

    chunks.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = []
    used_chars = 0
    for score, path, start, end, text in chunks[:14]:
        block = f"\n--- {path}:{start}-{end} (score {score}) ---\n{text}\n"
        if used_chars + len(block) > MAX_PROJECT_CONTEXT_CHARS:
            break
        selected.append(block)
        used_chars += len(block)
    return "\n".join(selected) if selected else "No directly relevant code snippets found."


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def database_overview() -> str:
    if not DUCKDB_PATH or DUCKDB_PATH == ":memory:":
        return "DuckDB overview unavailable: no persistent DuckDB path configured for this assistant."
    db_path = Path(DUCKDB_PATH)
    if not db_path.exists():
        return f"DuckDB overview unavailable: configured path does not exist ({DUCKDB_PATH})."

    lines = [f"DuckDB path: {DUCKDB_PATH}", "Tables/views:"]
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
        tables = conn.execute(
            """
            SELECT table_schema, table_name, table_type
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name;
            """
        ).fetchall()
        for schema, table, table_type in tables[:60]:
            full_name = f"{quote_ident(schema)}.{quote_ident(table)}"
            try:
                row_count = conn.execute(f"SELECT COUNT(*) FROM {full_name}").fetchone()[0]
            except Exception:
                row_count = "unknown"
            columns = conn.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = ? AND table_name = ?
                ORDER BY ordinal_position;
                """,
                (schema, table),
            ).fetchall()
            column_text = ", ".join(f"{name}:{dtype}" for name, dtype in columns[:24])
            lines.append(f"- {schema}.{table} ({table_type}, rows={row_count}) columns: {column_text}")
            column_names = {name for name, _ in columns}
            if "task_time" in column_names:
                try:
                    min_max = conn.execute(f"SELECT MIN(task_time), MAX(task_time) FROM {full_name}").fetchone()
                    lines.append(f"  task_time range: {min_max[0]} to {min_max[1]}")
                except Exception:
                    pass
            if "robot_id" in column_names:
                try:
                    distinct_robots = conn.execute(f"SELECT COUNT(DISTINCT robot_id) FROM {full_name}").fetchone()[0]
                    lines.append(f"  distinct robots: {distinct_robots}")
                except Exception:
                    pass
        conn.close()
    except Exception as exc:
        lines.append(f"DuckDB introspection failed: {type(exc).__name__}: {exc}")
    return "\n".join(lines)


def build_project_context(query: str) -> str:
    files = load_index()
    context_parts = [
        f"Project root: {PROJECT_ROOT}",
        file_tree_summary(files),
        route_summary(files),
        database_overview(),
        "Relevant source snippets:",
        relevant_snippets(query, files),
    ]
    context = "\n\n".join(context_parts)
    if len(context) > MAX_PROJECT_CONTEXT_CHARS:
        context = context[:MAX_PROJECT_CONTEXT_CHARS] + "\n...[project context truncated]"
    return context


def openrouter_models_to_try() -> list[str]:
    models: list[str] = []
    for model in [OPENROUTER_MODEL, *OPENROUTER_FALLBACK_MODELS]:
        if model and model not in models:
            models.append(model)
    return models


def call_openrouter_once(messages: list[dict[str, str]], model: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured on the server.")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.22,
        "max_tokens": 2600,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://159.195.33.49",
            "X-Title": "PUDU Dashboard Ask LLM",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body[:600]}") from exc
    except Exception as exc:
        raise RuntimeError(f"OpenRouter request failed: {type(exc).__name__}: {exc}") from exc

    try:
        return result["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise RuntimeError(f"Unexpected OpenRouter response shape: {result}") from exc


def call_openrouter(messages: list[dict[str, str]]) -> tuple[str, str]:
    errors: list[str] = []
    for model in openrouter_models_to_try():
        try:
            return call_openrouter_once(messages, model), model
        except RuntimeError as exc:
            text = str(exc)
            errors.append(f"{model}: {text[:300]}")
            is_rate_limit = "HTTP 429" in text or "rate-limit" in text.lower() or "rate limited" in text.lower()
            if not is_rate_limit:
                break
    raise RuntimeError("All OpenRouter model attempts failed. " + " | ".join(errors))


@app.on_event("startup")
def startup() -> None:
    init_history()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> str:
    if not is_authenticated(request):
        return LOGIN_HTML
    return HTML


@app.post("/api/login")
def login(payload: LoginRequest, response: Response) -> dict[str, Any]:
    if not hmac.compare_digest(payload.password, AUTH_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid password.")
    response.set_cookie(
        AUTH_COOKIE,
        auth_token(),
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,
    )
    return {"ok": True}


@app.post("/api/logout")
def logout(response: Response) -> dict[str, Any]:
    response.delete_cookie(AUTH_COOKIE)
    return {"ok": True}


@app.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    require_auth(request)
    files = load_index()
    return {
        "ok": True,
        "model": OPENROUTER_MODEL,
        "fallback_models": OPENROUTER_FALLBACK_MODELS,
        "project_root": str(PROJECT_ROOT),
        "indexed_files": len(files),
        "history_messages": sum(chat["message_count"] for chat in list_chats(10000)),
        "chat_count": len(list_chats(10000)),
        "duckdb_path": DUCKDB_PATH or None,
        "openrouter_configured": bool(os.getenv("OPENROUTER_API_KEY", "").strip()),
    }


@app.get("/api/chats")
def chats(request: Request) -> dict[str, Any]:
    require_auth(request)
    return {"chats": list_chats(), "default_chat_id": DEFAULT_CHAT_ID}


@app.post("/api/chats")
def new_chat(payload: CreateChatRequest, request: Request) -> dict[str, Any]:
    require_auth(request)
    return {"chat": create_chat(payload.title)}


@app.get("/api/chats/{chat_id}/messages")
def messages(chat_id: str, request: Request) -> dict[str, Any]:
    require_auth(request)
    chat = get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return {"chat": chat, "messages": list_messages(chat_id, 300), "model": OPENROUTER_MODEL}


@app.post("/api/chats/{chat_id}/messages")
def send_message(chat_id: str, payload: ChatRequest, request: Request) -> JSONResponse:
    require_auth(request)
    if not get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found.")
    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(content) > MAX_MESSAGE_CHARS:
        raise HTTPException(status_code=400, detail=f"Message is too long. Max {MAX_MESSAGE_CHARS} characters.")

    user_message = append_message(chat_id, "user", content)
    project_context = build_project_context(content)
    context_message = {
        "role": "system",
        "content": "Shared project/database context for the latest question:\n\n" + project_context,
    }
    model_messages = [{"role": "system", "content": SYSTEM_PROMPT}, context_message] + recent_model_messages(chat_id)

    try:
        answer, answered_by_model = call_openrouter(model_messages)
    except Exception as exc:
        answer = (
            "OpenRouter yaniti alinamadi. Sunucu loglarinda ayrinti var. "
            f"Hata ozeti: {type(exc).__name__}: {str(exc)[:360]}"
        )
        answered_by_model = OPENROUTER_MODEL
    assistant_message = append_message(chat_id, "assistant", answer, answered_by_model)
    return JSONResponse({"user": user_message, "assistant": assistant_message})


LOGIN_HTML = r"""
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PUDU Ask LLM Login</title>
  <style>
    :root{--bg:#eef4fb;--panel:#fff;--text:#172033;--muted:#64748b;--line:#dbe3ef;--brand:#2563eb;--danger:#b91c1c;--radius:8px}
    *{box-sizing:border-box}
    body{min-height:100vh;margin:0;display:grid;place-items:center;padding:20px;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:
      radial-gradient(circle at 20% 0%, rgba(37,99,235,.18), transparent 30%),
      linear-gradient(180deg,#f8fbff,var(--bg));color:var(--text)}
    .card{width:min(420px,100%);background:rgba(255,255,255,.9);border:1px solid var(--line);border-radius:var(--radius);box-shadow:0 18px 45px rgba(15,23,42,.14);padding:24px}
    .mark{width:46px;height:46px;border-radius:8px;background:linear-gradient(135deg,var(--brand),#14b8a6);margin-bottom:16px}
    h1{margin:0 0 6px;font-size:26px;letter-spacing:0}
    p{margin:0 0 18px;color:var(--muted);line-height:1.45}
    label{display:block;font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin-bottom:7px}
    input{width:100%;height:48px;border:1px solid var(--line);border-radius:8px;padding:0 13px;font:inherit;outline:none}
    input:focus{border-color:#93c5fd;box-shadow:0 0 0 3px rgba(37,99,235,.12)}
    button{width:100%;height:48px;border:0;border-radius:8px;margin-top:12px;background:linear-gradient(135deg,var(--brand),#1d4ed8);color:#fff;font-weight:850;cursor:pointer}
    .error{min-height:20px;margin-top:10px;color:var(--danger);font-size:13px}
  </style>
</head>
<body>
  <form class="card" id="loginForm">
    <div class="mark" aria-hidden="true"></div>
    <h1>Ask LLM</h1>
    <p>This shared dashboard assistant is password protected.</p>
    <label for="password">Password</label>
    <input id="password" type="password" autocomplete="current-password" autofocus>
    <button type="submit">Enter</button>
    <div class="error" id="error"></div>
  </form>
  <script>
    document.getElementById("loginForm").addEventListener("submit", async event => {
      event.preventDefault();
      const error = document.getElementById("error");
      error.textContent = "";
      const password = document.getElementById("password").value;
      const response = await fetch("/api/login", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({password})
      });
      if (!response.ok){
        error.textContent = "Wrong password.";
        return;
      }
      window.location.reload();
    });
  </script>
</body>
</html>
"""


HTML = r"""
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PUDU Ask LLM</title>
  <style>
    :root{
      --bg:#f5f7fb; --panel:#ffffff; --panel-2:#f8fafc; --text:#172033; --muted:#64748b;
      --line:#dbe3ef; --brand:#2563eb; --brand-2:#14b8a6; --danger:#ef4444;
      --shadow:0 18px 45px rgba(15,23,42,.12); --radius:8px;
    }
    *{box-sizing:border-box}
    html,body{height:100%}
    body{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:
      radial-gradient(circle at 12% -10%, rgba(37,99,235,.18), transparent 28%),
      linear-gradient(180deg,#f8fbff 0%, var(--bg) 46%, #eef4fb 100%);
      color:var(--text)}
    .shell{min-height:100%;display:grid;grid-template-rows:auto 1fr auto}
    .top{padding:20px clamp(16px,4vw,42px);display:flex;align-items:center;justify-content:space-between;gap:18px}
    .brand{display:flex;gap:12px;align-items:center}
    .mark{width:42px;height:42px;border-radius:8px;background:linear-gradient(135deg,var(--brand),var(--brand-2));box-shadow:var(--shadow);position:relative}
    .mark:after{content:"";position:absolute;inset:11px;border:2px solid rgba(255,255,255,.9);border-radius:6px}
    h1{margin:0;font-size:clamp(22px,3vw,34px);letter-spacing:0;font-weight:800}
    .subtitle{margin:3px 0 0;color:var(--muted);font-size:13px}
    .chips{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}
    .chip{padding:7px 10px;border:1px solid var(--line);background:rgba(255,255,255,.74);border-radius:999px;color:#334155;font-size:12px;font-weight:700}
    .layout{width:min(1420px,100%);margin:0 auto;padding:0 clamp(14px,3vw,34px) 18px;display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:18px}
    .chat,.side-card,.composer{border:1px solid rgba(219,227,239,.95);background:rgba(255,255,255,.86);box-shadow:var(--shadow);backdrop-filter:blur(14px);border-radius:var(--radius)}
    .chat{min-height:calc(100vh - 190px);display:flex;flex-direction:column;overflow:hidden}
    .chat-head{padding:14px 16px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;gap:10px;background:rgba(248,250,252,.8)}
    .chat-title{font-weight:800}
    .status{font-size:12px;color:var(--muted)}
    .messages{padding:18px;display:flex;flex-direction:column;gap:14px;overflow:auto;flex:1;scroll-behavior:smooth}
    .empty{margin:auto;text-align:center;max-width:520px;color:var(--muted)}
    .empty strong{display:block;color:var(--text);font-size:18px;margin-bottom:8px}
    .msg{display:grid;grid-template-columns:38px minmax(0,1fr);gap:10px;align-items:start}
    .avatar{width:38px;height:38px;border-radius:8px;display:grid;place-items:center;color:#fff;font-weight:900;font-size:13px;background:#0f172a}
    .msg.assistant .avatar{background:linear-gradient(135deg,var(--brand),var(--brand-2))}
    .bubble{border:1px solid var(--line);background:#fff;border-radius:8px;padding:12px 13px;line-height:1.55;font-size:14px;overflow:auto}
    .msg.user .bubble{background:#eff6ff;border-color:#bfdbfe}
    .meta{display:flex;align-items:center;gap:8px;margin-bottom:5px;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:800}
    pre{margin:10px 0;padding:12px;border-radius:8px;background:#0f172a;color:#e2e8f0;overflow:auto;font-size:12.5px;line-height:1.45}
    code{font-family:"SFMono-Regular",Consolas,monospace}
    .side{display:flex;flex-direction:column;gap:12px}
    .side-card{padding:15px}
    .side-card.thread-card{padding:12px}
    .side-card h2{font-size:14px;margin:0 0 10px}
    .side-card p,.side-card li{font-size:13px;color:var(--muted);line-height:1.45}
    .side-card ul{padding-left:18px;margin:8px 0 0}
    .side-title-row{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:10px}
    .side-title-row h2{margin:0}
    .mini-btn{min-width:auto;height:32px;padding:0 10px;border:1px solid var(--line);background:#fff;color:#334155;border-radius:8px;font-size:12px;font-weight:800}
    .mini-btn.primary{border-color:transparent;background:linear-gradient(135deg,var(--brand),#1d4ed8);color:#fff}
    .thread-list{display:flex;flex-direction:column;gap:7px;max-height:280px;overflow:auto;padding-right:2px}
    .thread-item{border:1px solid var(--line);background:#fff;border-radius:8px;padding:9px 10px;text-align:left;cursor:pointer;min-width:0;color:var(--text)}
    .thread-item.active{border-color:#93c5fd;background:#eff6ff;box-shadow:0 0 0 3px rgba(37,99,235,.08)}
    .thread-name{font-weight:800;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .thread-meta{margin-top:3px;font-size:11px;color:var(--muted)}
    .share-row{display:flex;gap:8px;margin-top:10px}
    .share-row .mini-btn{flex:1}
    .metric{display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-bottom:1px solid var(--line);font-size:13px}
    .metric:last-child{border-bottom:0}
    .metric span:first-child{color:var(--muted)}
    .metric span:last-child{font-weight:800;text-align:right}
    .composer-wrap{padding:0 clamp(14px,3vw,34px) 22px;width:min(1420px,100%);margin:0 auto}
    .composer{padding:10px;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:10px}
    textarea{width:100%;min-height:54px;max-height:190px;resize:none;border:1px solid var(--line);border-radius:8px;padding:13px 14px;font:inherit;line-height:1.45;outline:none;background:#fff;color:var(--text)}
    textarea:focus{border-color:#93c5fd;box-shadow:0 0 0 3px rgba(37,99,235,.12)}
    button{border:0;border-radius:8px;padding:0 18px;background:linear-gradient(135deg,var(--brand),#1d4ed8);color:#fff;font-weight:800;cursor:pointer;min-width:112px}
    button:disabled{opacity:.55;cursor:not-allowed}
    .footnote{margin-top:8px;font-size:12px;color:var(--muted)}
    .error{color:#b91c1c}
    @media (max-width: 900px){
      .top{align-items:flex-start;flex-direction:column}
      .chips{justify-content:flex-start}
      .layout{grid-template-columns:1fr}
      .chat{min-height:62vh}
      .side{order:-1}
      .thread-list{max-height:180px}
      .composer{grid-template-columns:1fr}
      button{height:46px}
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="top">
      <div class="brand">
        <div class="mark" aria-hidden="true"></div>
        <div>
          <h1>Ask LLM</h1>
          <p class="subtitle">PUDU dashboard code, frontend, model runtime and database assistant.</p>
        </div>
      </div>
      <div class="chips">
        <span class="chip" id="modelChip">Qwen</span>
        <span class="chip">Shared history</span>
        <span class="chip">Code + DB context</span>
      </div>
    </header>

    <main class="layout">
      <section class="chat" aria-label="Shared chat">
        <div class="chat-head">
          <div>
            <div class="chat-title" id="activeChatTitle">Global dashboard chat</div>
            <div class="status" id="statusText">Connecting...</div>
          </div>
          <div class="status" id="countText">0 messages</div>
        </div>
        <div class="messages" id="messages">
          <div class="empty"><strong>Ask about the dashboard.</strong>Questions can be in Turkish, English, or any language. Everyone opening this link sees the same conversation history.</div>
        </div>
      </section>

      <aside class="side">
        <section class="side-card thread-card">
          <div class="side-title-row">
            <h2>Chats</h2>
            <button class="mini-btn primary" type="button" id="newChatBtn">New</button>
          </div>
          <div class="thread-list" id="chatList"></div>
          <div class="share-row">
            <button class="mini-btn" type="button" id="copyLinkBtn">Copy link</button>
            <button class="mini-btn" type="button" id="logoutBtn">Logout</button>
          </div>
        </section>
        <section class="side-card">
          <h2>Context</h2>
          <div class="metric"><span>Model</span><span id="modelName">-</span></div>
          <div class="metric"><span>Indexed files</span><span id="fileCount">-</span></div>
          <div class="metric"><span>Database</span><span id="dbState">-</span></div>
          <div class="metric"><span>OpenRouter</span><span id="keyState">-</span></div>
        </section>
        <section class="side-card">
          <h2>Good questions</h2>
          <ul>
            <li>Prediction sayfasinda Head 3 kartlari nasil siralaniyor?</li>
            <li>DuckDB tarafinda hangi tablolar var?</li>
            <li>Frontend'de Fault History resolved notu nerede render ediliyor?</li>
            <li>How does the LSTM runtime load model artifacts?</li>
          </ul>
          <p class="footnote">The assistant reads server-side context for every answer; secrets are not exposed to the browser.</p>
        </section>
      </aside>
    </main>

    <footer class="composer-wrap">
      <form class="composer" id="composer">
        <textarea id="input" placeholder="Site, frontend, API, model veya database hakkinda sor..." autocomplete="off"></textarea>
        <button id="sendBtn" type="submit">Ask</button>
      </form>
    </footer>
  </div>

  <script>
    const messagesEl = document.getElementById("messages");
    const chatListEl = document.getElementById("chatList");
    const form = document.getElementById("composer");
    const input = document.getElementById("input");
    const sendBtn = document.getElementById("sendBtn");
    const newChatBtn = document.getElementById("newChatBtn");
    const copyLinkBtn = document.getElementById("copyLinkBtn");
    const logoutBtn = document.getElementById("logoutBtn");
    const statusText = document.getElementById("statusText");
    const countText = document.getElementById("countText");
    const activeChatTitle = document.getElementById("activeChatTitle");
    let lastRender = "";
    let chats = [];
    let currentChatId = new URLSearchParams(window.location.search).get("chat") || "main";
    let busy = false;

    function escapeHtml(value){
      return String(value || "").replace(/[&<>"']/g, ch => ({
        "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;"
      }[ch]));
    }
    function formatTime(ts){
      if (!ts) return "";
      return new Date(ts * 1000).toLocaleString(undefined, { dateStyle:"medium", timeStyle:"short" });
    }
    function setChatUrl(chatId, replace=false){
      const url = new URL(window.location.href);
      url.searchParams.set("chat", chatId);
      if (replace) history.replaceState({chatId}, "", url);
      else history.pushState({chatId}, "", url);
    }
    function currentShareUrl(){
      const url = new URL(window.location.href);
      url.searchParams.set("chat", currentChatId);
      return url.toString();
    }
    function renderMarkdown(text){
      const parts = String(text || "").split(/```/);
      return parts.map((part, idx) => {
        if (idx % 2 === 1){
          const cleaned = part.replace(/^[a-zA-Z0-9_+-]+\n/, "");
          return `<pre><code>${escapeHtml(cleaned)}</code></pre>`;
        }
        return escapeHtml(part)
          .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
          .replace(/`([^`]+)`/g, "<code>$1</code>")
          .replace(/\n/g, "<br>");
      }).join("");
    }
    function renderMessages(messages){
      const signature = JSON.stringify(messages.map(m => [m.id, m.role, m.content]));
      if (signature === lastRender) return;
      lastRender = signature;
      countText.textContent = `${messages.length} messages`;
      if (!messages.length){
        messagesEl.innerHTML = `<div class="empty"><strong>Ask about the dashboard.</strong>Questions can be in Turkish, English, or any language. Everyone opening this link sees the same conversation history.</div>`;
        return;
      }
      messagesEl.innerHTML = messages.map(message => {
        const role = message.role === "assistant" ? "assistant" : "user";
        const label = role === "assistant" ? "LLM" : "User";
        const avatar = role === "assistant" ? "AI" : "U";
        const model = message.model ? ` · ${escapeHtml(message.model)}` : "";
        return `<article class="msg ${role}">
          <div class="avatar">${avatar}</div>
          <div class="bubble">
            <div class="meta"><span>${label}</span><span>${escapeHtml(formatTime(message.created_at))}${model}</span></div>
            <div>${renderMarkdown(message.content)}</div>
          </div>
        </article>`;
      }).join("");
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
    function renderChats(){
      if (!chats.length){
        chatListEl.innerHTML = `<div class="thread-meta">No chats yet.</div>`;
        return;
      }
      chatListEl.innerHTML = chats.map(chat => {
        const active = chat.id === currentChatId ? " active" : "";
        const count = Number(chat.message_count || 0);
        return `<button class="thread-item${active}" type="button" data-chat-id="${escapeHtml(chat.id)}">
          <div class="thread-name">${escapeHtml(chat.title || "New chat")}</div>
          <div class="thread-meta">${count} messages · ${escapeHtml(formatTime(chat.updated_at || chat.created_at))}</div>
        </button>`;
      }).join("");
      chatListEl.querySelectorAll("[data-chat-id]").forEach(button => {
        button.addEventListener("click", () => switchChat(button.dataset.chatId));
      });
    }
    async function loadChats(){
      const response = await fetch("/api/chats");
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      chats = data.chats || [];
      if (!chats.some(chat => chat.id === currentChatId)){
        currentChatId = data.default_chat_id || (chats[0] && chats[0].id) || "main";
        setChatUrl(currentChatId, true);
      }
      renderChats();
    }
    async function switchChat(chatId, replace=false){
      if (!chatId || chatId === currentChatId && !replace) return;
      currentChatId = chatId;
      setChatUrl(chatId, replace);
      lastRender = "";
      renderChats();
      await loadMessages();
    }
    async function loadHealth(){
      const response = await fetch("/api/health");
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      document.getElementById("modelName").textContent = data.model || "-";
      document.getElementById("modelChip").textContent = data.model || "Qwen";
      document.getElementById("fileCount").textContent = data.indexed_files ?? "-";
      document.getElementById("dbState").textContent = data.duckdb_path ? "ready" : "not configured";
      document.getElementById("keyState").textContent = data.openrouter_configured ? "configured" : "missing";
      statusText.textContent = data.ok ? "Ready" : "Not ready";
    }
    async function loadMessages(){
      const response = await fetch(`/api/chats/${encodeURIComponent(currentChatId)}/messages`);
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      if (data.chat) activeChatTitle.textContent = data.chat.title || "New chat";
      renderMessages(data.messages || []);
    }
    async function sendMessage(content){
      busy = true;
      sendBtn.disabled = true;
      sendBtn.textContent = "Thinking";
      statusText.textContent = "Model is answering...";
      try{
        const response = await fetch(`/api/chats/${encodeURIComponent(currentChatId)}/messages`, {
          method:"POST",
          headers:{"Content-Type":"application/json"},
          body:JSON.stringify({content})
        });
        if (!response.ok){
          const detail = await response.text();
          throw new Error(detail || response.statusText);
        }
        await loadChats();
        await loadMessages();
        statusText.textContent = "Ready";
      }catch(error){
        statusText.innerHTML = `<span class="error">${escapeHtml(error.message || error)}</span>`;
      }finally{
        busy = false;
        sendBtn.disabled = false;
        sendBtn.textContent = "Ask";
      }
    }
    async function createChat(){
      const response = await fetch("/api/chats", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({title:"New chat"})
      });
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      await loadChats();
      await switchChat(data.chat.id);
      input.focus();
    }
    form.addEventListener("submit", event => {
      event.preventDefault();
      const content = input.value.trim();
      if (!content || busy) return;
      input.value = "";
      input.style.height = "54px";
      sendMessage(content);
    });
    input.addEventListener("input", () => {
      input.style.height = "54px";
      input.style.height = Math.min(input.scrollHeight, 190) + "px";
    });
    input.addEventListener("keydown", event => {
      if (event.key === "Enter" && !event.shiftKey){
        event.preventDefault();
        form.requestSubmit();
      }
    });
    newChatBtn.addEventListener("click", () => createChat().catch(error => {
      statusText.innerHTML = `<span class="error">${escapeHtml(error.message || error)}</span>`;
    }));
    copyLinkBtn.addEventListener("click", async () => {
      const link = currentShareUrl();
      try{
        await navigator.clipboard.writeText(link);
        statusText.textContent = "Chat link copied.";
      }catch(error){
        window.prompt("Copy this chat link", link);
      }
    });
    logoutBtn.addEventListener("click", async () => {
      await fetch("/api/logout", {method:"POST"});
      window.location.reload();
    });
    window.addEventListener("popstate", () => {
      const next = new URLSearchParams(window.location.search).get("chat") || "main";
      switchChat(next, true).catch(error => {
        statusText.innerHTML = `<span class="error">${escapeHtml(error.message || error)}</span>`;
      });
    });
    Promise.all([loadHealth(), loadChats()]).then(() => loadMessages()).catch(error => {
      statusText.innerHTML = `<span class="error">${escapeHtml(error.message || error)}</span>`;
    });
    setInterval(() => { if (!busy) loadMessages().catch(()=>{}); }, 5000);
  </script>
</body>
</html>
"""
