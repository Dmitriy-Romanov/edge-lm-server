import asyncio
import time
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
import mlx.core as mx
import uvicorn
from typing import Any, Dict, List

from edge_lm.models.load import load
from mlx_vlm import stream_generate

model = None
tokenizer = None
MODEL_SOURCE = os.environ.get(
    "EDGE_LM_MODEL_SOURCE",
    os.environ.get("EDGE_LM_MODEL", "TheStageAI/gemma-4-E4B-it-qat"),
)
MODEL_NAME = os.environ.get("EDGE_LM_MODEL_ID", MODEL_SOURCE)
MODEL_SIZE = os.environ.get("EDGE_LM_SIZE", "m")
MAX_CONTEXT_TOKENS = int(os.environ.get("EDGE_LM_CONTEXT_TOKENS", "128000"))
DEFAULT_MAX_TOKENS = 16000
HOST = os.environ.get("EDGE_LM_HOST", "127.0.0.1")
PORT = int(os.environ.get("EDGE_LM_PORT", "8000"))
STATS_PATH = Path(os.environ.get("EDGE_LM_STATS_PATH", "stats.json"))
FAVICON_PATH = Path(os.environ.get("EDGE_LM_FAVICON_PATH", "favicon.ico"))
SESSION_STARTED_AT = time.time()
SESSION_STATS = {
    "active_requests": 0,
    "started_requests": 0,
    "completed_requests": 0,
    "failed_requests": 0,
    "total_prompt_tokens": 0,
    "total_generated_tokens": 0,
    "total_generation_seconds": 0.0,
    "last_request": None,
}

TOOL_CALL_PATTERN = re.compile(
    r'<\|tool_call>call:([A-Za-z_][\w.-]*)\{(.*?)\}<tool_call\|>',
    re.DOTALL,
)
PARAM_PATTERN = re.compile(r'([A-Za-z_][\w.-]*):<\|"\|>(.*?)<\|"\|>', re.DOTALL)


def parse_gemma_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls = []
    for match in TOOL_CALL_PATTERN.finditer(text):
        name = match.group(1)
        params = {}
        for param_match in PARAM_PATTERN.finditer(match.group(2)):
            params[param_match.group(1)] = param_match.group(2)
        calls.append({"name": name, "arguments": params})
    return calls


def normalize_messages_for_template(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted = []
    tool_call_names = {}

    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            if tool_call.get("id") and function.get("name"):
                tool_call_names[tool_call["id"]] = function["name"]

    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg:
            continue

        role = msg["role"]
        if role == "assistant" and msg.get("tool_calls"):
            tool_calls = []
            for tool_call in msg.get("tool_calls"):
                function = dict(tool_call.get("function") or {})
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"arguments": arguments}
                function["arguments"] = arguments
                tool_calls.append({"type": "function", "function": function})
            formatted.append({"role": "assistant", "tool_calls": tool_calls})
            continue

        if role == "tool":
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = "".join(text_parts)
            item = {"role": "tool", "content": content}
            if msg.get("name"):
                item["name"] = msg["name"]
            elif msg.get("tool_call_id") in tool_call_names:
                item["name"] = tool_call_names[msg["tool_call_id"]]
            if msg.get("tool_call_id"):
                item["tool_call_id"] = msg["tool_call_id"]
            formatted.append(item)
            continue

        if "content" in msg:
            content = msg["content"]
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = "".join(text_parts)
            formatted.append({"role": role, "content": content})
    return formatted


def build_prompt_ids(
    messages: List[Dict[str, Any]],
    template_kwargs: Dict[str, Any],
    max_prompt_tokens: int,
) -> List[int]:
    keep_head = []
    rest = list(messages)
    while rest and rest[0].get("role") == "system":
        keep_head.append(rest.pop(0))

    while True:
        prompt_text = tokenizer.apply_chat_template(
            keep_head + rest,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        ids = [int(x) for x in tokenizer.encode(prompt_text)]
        if len(ids) <= max_prompt_tokens or len(rest) <= 1:
            return ids
        rest.pop(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print("Loading optimized Gemma-4 into unified memory...")
    model, tokenizer = load(MODEL_SOURCE, size=MODEL_SIZE)
    set_tokenizer_chat_template(tokenizer)
    print("Model loaded successfully and ready through MLX.")
    yield
    print("Unloading model from memory...")


class SuppressStatusJsonAccessLog(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/status.json" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(SuppressStatusJsonAccessLog())

app = FastAPI(title="TheStage AI Edge-LM OpenAI Server", lifespan=lifespan)


def set_tokenizer_chat_template(tokenizer):
    # Gemma chat template (based on Hugging Face Gemma tokenizer)
    template = "{% for message in messages %}{{'<bos>' if loop.first else ''}}{% if message['role'] == 'user' %}{{'<start_of_turn>user\n' + (message['content'] | default('')) + '<end_of_turn>\n'}}{% elif message['role'] == 'assistant' %}{{'<start_of_turn>model\n' + (message['content'] | default('')) + '<end_of_turn>\n'}}{% elif message['role'] == 'system' %}{{'<start_of_turn>system\n' + (message['content'] | default('')) + '<end_of_turn>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
    target = tokenizer
    if hasattr(tokenizer, "_tokenizer"):
        target = tokenizer._tokenizer
    if target.chat_template is None:
        target.chat_template = template


def default_persistent_stats() -> Dict[str, Any]:
    return {
        "server_starts": 0,
        "completed_requests": 0,
        "failed_requests": 0,
        "total_prompt_tokens": 0,
        "total_generated_tokens": 0,
        "total_generation_seconds": 0.0,
        "last_request": None,
    }


def load_persistent_stats() -> Dict[str, Any]:
    try:
        loaded = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default_persistent_stats()

    stats = default_persistent_stats()
    if isinstance(loaded, dict):
        for key in stats:
            if key in loaded:
                stats[key] = loaded[key]
    return stats


def save_persistent_stats() -> None:
    try:
        STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = STATS_PATH.with_name(f"{STATS_PATH.name}.tmp")
        temporary_path.write_text(
            json.dumps(PERSISTENT_STATS, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(STATS_PATH)
    except OSError as exc:
        print(f"[status] failed to write {STATS_PATH}: {exc}", flush=True)


PERSISTENT_STATS = load_persistent_stats()
PERSISTENT_STATS["server_starts"] += 1
save_persistent_stats()


def record_request_started(chat_id: str, prompt_tokens: int, max_tokens: int) -> None:
    SESSION_STATS["active_requests"] += 1
    SESSION_STATS["started_requests"] += 1
    SESSION_STATS["last_request"] = {
        "id": chat_id,
        "status": "running",
        "prompt_tokens": prompt_tokens,
        "generated_tokens": 0,
        "max_tokens": max_tokens,
        "generation_seconds": 0.0,
        "tokens_per_second": 0.0,
        "finished_at": None,
    }


def record_request_finished(
    chat_id: str,
    prompt_tokens: int,
    generated_tokens: int,
    generation_seconds: float,
    max_tokens: int,
    status: str,
) -> None:
    SESSION_STATS["active_requests"] = max(0, SESSION_STATS["active_requests"] - 1)
    if status == "completed":
        SESSION_STATS["completed_requests"] += 1
    else:
        SESSION_STATS["failed_requests"] += 1
    SESSION_STATS["total_prompt_tokens"] += prompt_tokens
    SESSION_STATS["total_generated_tokens"] += generated_tokens
    SESSION_STATS["total_generation_seconds"] += generation_seconds
    tokens_per_second = generated_tokens / generation_seconds if generation_seconds > 0 else 0.0
    last_request = {
        "id": chat_id,
        "status": status,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "max_tokens": max_tokens,
        "generation_seconds": generation_seconds,
        "tokens_per_second": tokens_per_second,
        "finished_at": time.time(),
    }
    SESSION_STATS["last_request"] = last_request

    if status == "completed":
        PERSISTENT_STATS["completed_requests"] += 1
    else:
        PERSISTENT_STATS["failed_requests"] += 1
    PERSISTENT_STATS["total_prompt_tokens"] += prompt_tokens
    PERSISTENT_STATS["total_generated_tokens"] += generated_tokens
    PERSISTENT_STATS["total_generation_seconds"] += generation_seconds
    PERSISTENT_STATS["last_request"] = last_request
    save_persistent_stats()


def totals_snapshot(stats: Dict[str, Any]) -> Dict[str, Any]:
    total_generation_seconds = float(stats["total_generation_seconds"])
    total_generated_tokens = int(stats["total_generated_tokens"])
    average_tokens_per_second = (
        total_generated_tokens / total_generation_seconds
        if total_generation_seconds > 0
        else 0.0
    )
    return {
        "promptTokens": stats["total_prompt_tokens"],
        "generatedTokens": total_generated_tokens,
        "generationSeconds": total_generation_seconds,
        "averageTokensPerSecond": average_tokens_per_second,
    }


def status_snapshot() -> Dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "modelSource": str(MODEL_SOURCE),
        "size": MODEL_SIZE,
        "uptimeSeconds": time.time() - SESSION_STARTED_AT,
        "limits": {
            "contextWindowTokens": MAX_CONTEXT_TOKENS,
            "defaultMaxOutputTokens": DEFAULT_MAX_TOKENS,
        },
        "requests": {
            "active": SESSION_STATS["active_requests"],
            "started": SESSION_STATS["started_requests"],
            "completed": SESSION_STATS["completed_requests"],
            "failed": SESSION_STATS["failed_requests"],
        },
        "totals": totals_snapshot(SESSION_STATS),
        "allTime": {
            "statsPath": str(STATS_PATH),
            "serverStarts": PERSISTENT_STATS["server_starts"],
            "requests": {
                "completed": PERSISTENT_STATS["completed_requests"],
                "failed": PERSISTENT_STATS["failed_requests"],
            },
            "totals": totals_snapshot(PERSISTENT_STATS),
            "lastRequest": PERSISTENT_STATS["last_request"],
        },
        "lastRequest": SESSION_STATS["last_request"],
    }


@app.get("/status.json")
async def status_json():
    return status_snapshot()


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    if not FAVICON_PATH.exists():
        raise HTTPException(status_code=404, detail="favicon.ico is missing")
    return FileResponse(FAVICON_PATH, media_type="image/x-icon")


@app.get("/status", response_class=HTMLResponse)
async def status_page():
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="/favicon.ico">
  <title>edge-lm-server status</title>
  <style>
    :root { color-scheme: light dark; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: Canvas; color: CanvasText; }
    main { max-width: 980px; margin: 0 auto; padding: 28px 20px 40px; }
    header { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 20px; }
    h1 { font-size: 24px; margin: 0; font-weight: 650; }
    #updated { color: color-mix(in srgb, CanvasText 62%, transparent); font-size: 13px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .card { border: 1px solid color-mix(in srgb, CanvasText 16%, transparent); border-radius: 8px; padding: 14px; background: color-mix(in srgb, Canvas 94%, CanvasText 6%); }
    .label { color: color-mix(in srgb, CanvasText 62%, transparent); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
    .value { font-size: 28px; font-weight: 700; margin-top: 6px; overflow-wrap: anywhere; }
    .hint { color: color-mix(in srgb, CanvasText 62%, transparent); font-size: 13px; margin-top: 6px; line-height: 1.35; }
    .span-2 { grid-column: span 2; }
    @media (max-width: 760px) {
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .span-2 { grid-column: 1 / -1; }
    }
    @media (max-width: 460px) {
      .grid { grid-template-columns: 1fr; }
      .span-2 { grid-column: 1; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>edge-lm-server status</h1>
      <div id="updated">loading...</div>
    </header>
    <section class="grid">
      <div class="card"><div class="label">Session generated</div><div class="value" id="session-generated">-</div><div class="hint">Completion tokens generated since this server process started.</div></div>
      <div class="card"><div class="label">Session time</div><div class="value" id="session-seconds">-</div><div class="hint">Time spent generating completions in this process.</div></div>
      <div class="card"><div class="label">Session speed</div><div class="value" id="session-speed">-</div><div class="hint">Session generated tokens divided by session generation time.</div></div>
      <div class="card"><div class="label">Requests</div><div class="value" id="requests">-</div><div class="hint">Completed / active in the current session.</div></div>
      <div class="card"><div class="label">Total generated</div><div class="value" id="total-generated">-</div><div class="hint">Completion tokens persisted across server restarts.</div></div>
      <div class="card"><div class="label">Total time</div><div class="value" id="total-seconds">-</div><div class="hint">Persisted generation time across server restarts.</div></div>
      <div class="card"><div class="label">Total speed</div><div class="value" id="total-speed">-</div><div class="hint">All-time generated tokens divided by all-time generation time.</div></div>
      <div class="card"><div class="label">Total requests</div><div class="value" id="total-requests">-</div><div class="hint">Completed / failed requests across all sessions.</div></div>
      <div class="card"><div class="label">Context window</div><div class="value" id="context">-</div><div class="hint">Maximum prompt context accepted by this server.</div></div>
      <div class="card"><div class="label">Default max output</div><div class="value" id="max-output">-</div><div class="hint">Default generation limit when the client sends a very small or missing max_tokens.</div></div>
      <div class="card span-2"><div class="label">Model</div><div class="value" id="model">-</div><div class="hint" id="model-source"></div></div>
    </section>
  </main>
  <script>
    const fmt = new Intl.NumberFormat();
    const sec = value => `${Number(value || 0).toFixed(2)} s`;
    const rate = value => `${Number(value || 0).toFixed(2)} tok/s`;
    async function refresh() {
      const response = await fetch('/status.json', { cache: 'no-store' });
      const data = await response.json();
      document.getElementById('context').textContent = fmt.format(data.limits.contextWindowTokens);
      document.getElementById('max-output').textContent = fmt.format(data.limits.defaultMaxOutputTokens);
      document.getElementById('session-generated').textContent = fmt.format(data.totals.generatedTokens);
      document.getElementById('total-generated').textContent = fmt.format(data.allTime.totals.generatedTokens);
      document.getElementById('session-seconds').textContent = sec(data.totals.generationSeconds);
      document.getElementById('total-seconds').textContent = sec(data.allTime.totals.generationSeconds);
      document.getElementById('session-speed').textContent = rate(data.totals.averageTokensPerSecond);
      document.getElementById('total-speed').textContent = rate(data.allTime.totals.averageTokensPerSecond);
      document.getElementById('requests').textContent = `${fmt.format(data.requests.completed)} / ${fmt.format(data.requests.active)}`;
      document.getElementById('total-requests').textContent = `${fmt.format(data.allTime.requests.completed)} / ${fmt.format(data.allTime.requests.failed)}`;
      document.getElementById('model').textContent = `${data.model} (${data.size})`;
      document.getElementById('model-source').textContent = data.modelSource;
      document.getElementById('updated').textContent = `updated ${new Date().toLocaleTimeString()}`;
    }
    refresh().catch(error => { document.getElementById('updated').textContent = error.message; });
    setInterval(() => refresh().catch(error => { document.getElementById('updated').textContent = error.message; }), 1000);
  </script>
</body>
</html>"""

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "thestageai",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(body: Dict[str, Any]):
    global model, tokenizer
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model is not loaded yet")

    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="'messages' is missing or empty")

    max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
    if max_tokens < 512:
        max_tokens = DEFAULT_MAX_TOKENS
    model_field = MODEL_NAME
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    formatted_messages = normalize_messages_for_template(messages)
    template_kwargs = {}
    if body.get("tools"):
        template_kwargs["tools"] = body["tools"]
        tool_names = [
            tool.get("function", {}).get("name", "<unknown>")
            for tool in body["tools"]
        ]
        print(f"[tools] received {len(tool_names)}: {', '.join(tool_names)}", flush=True)
    else:
        print("[tools] none received", flush=True)

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    max_prompt_tokens = max(1, MAX_CONTEXT_TOKENS - max_tokens)
    ids = build_prompt_ids(formatted_messages, template_kwargs, max_prompt_tokens)
    record_request_started(chat_id, len(ids), max_tokens)
    print(
        f"[{chat_id}] prompt={len(ids)} tokens, max_new={max_tokens}, "
        f"context_limit={MAX_CONTEXT_TOKENS}, include_usage={include_usage}",
        flush=True,
    )

    async def generate_chunks_async():
        initial_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_field,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            "usage": None,
        }
        yield f"data: {json.dumps(initial_chunk)}\n\n"

        text_parts = []
        generated_tokens = 0
        started_at = time.perf_counter()
        last_log_at = started_at
        stats_recorded = False
        try:
            for result in stream_generate(
                model,
                tokenizer,
                "",
                input_ids=mx.array([ids], dtype=mx.int32),
                max_tokens=max_tokens,
            ):
                text_parts.append(result.text)
                generated_tokens += 1
                now = time.perf_counter()
                if now - last_log_at >= 1.0:
                    elapsed = max(now - started_at, 1e-9)
                    print(
                        f"[{chat_id}] generated={generated_tokens} tokens, "
                        f"speed={generated_tokens / elapsed:.2f} tok/s",
                        flush=True,
                    )
                    last_log_at = now
                await asyncio.sleep(0)

            generated_text = "".join(text_parts).rstrip()
            # Remove Gemma special turn tokens from the end.
            suffixes = ["<end_of_turn>", "<turn|>"]
            for suf in suffixes:
                if generated_text.endswith(suf):
                    generated_text = generated_text[: -len(suf)].rstrip()
                    break
            elapsed = max(time.perf_counter() - started_at, 1e-9)
            record_request_finished(chat_id, len(ids), generated_tokens, elapsed, max_tokens, "completed")
            stats_recorded = True
            print(
                f"[{chat_id}] done generated={generated_tokens} tokens, "
                f"elapsed={elapsed:.2f}s, speed={generated_tokens / elapsed:.2f} tok/s",
                flush=True,
            )
            tool_calls = parse_gemma_tool_calls(generated_text)
            finish_reason = "stop"

            if tool_calls:
                finish_reason = "tool_calls"
                for index, tool_call in enumerate(tool_calls):
                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_field,
                        "usage": None,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": index,
                                            "id": f"call_{int(time.time() * 1000)}_{index}",
                                            "type": "function",
                                            "function": {
                                                "name": tool_call["name"],
                                                "arguments": json.dumps(
                                                    tool_call["arguments"],
                                                    ensure_ascii=False,
                                                ),
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0)
            elif generated_text:
                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_field,
                    "usage": None,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": generated_text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)

            final_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_field,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "usage": None,
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            if include_usage:
                usage_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_field,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": len(ids),
                        "completion_tokens": generated_tokens,
                        "total_tokens": len(ids) + generated_tokens,
                    },
                }
                yield f"data: {json.dumps(usage_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if not stats_recorded:
                elapsed = max(time.perf_counter() - started_at, 1e-9)
                record_request_finished(chat_id, len(ids), generated_tokens, elapsed, max_tokens, "interrupted")

    return StreamingResponse(generate_chunks_async(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
