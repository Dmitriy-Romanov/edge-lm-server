import asyncio
import time
import json
import os
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import mlx.core as mx
import uvicorn
from typing import Any, Dict, List

from edge_lm.models.load import load
from mlx_vlm import stream_generate

model = None
tokenizer = None
MODEL_SOURCE = os.environ.get(
    "EDGE_LM_MODEL_SOURCE",
    os.environ.get("EDGE_LM_MODEL", "TheStageAI/gemma-4-E4B-it"),
)
MODEL_NAME = os.environ.get("EDGE_LM_MODEL_ID", MODEL_SOURCE)
MODEL_SIZE = os.environ.get("EDGE_LM_SIZE", "m")
MAX_CONTEXT_TOKENS = int(os.environ.get("EDGE_LM_CONTEXT_TOKENS", "128000"))
HOST = os.environ.get("EDGE_LM_HOST", "127.0.0.1")
PORT = int(os.environ.get("EDGE_LM_PORT", "8000"))

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
            for tool_call in msg["tool_calls"]:
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
            item = {"role": "tool", "content": msg.get("content", "")}
            if msg.get("name"):
                item["name"] = msg["name"]
            elif msg.get("tool_call_id") in tool_call_names:
                item["name"] = tool_call_names[msg["tool_call_id"]]
            if msg.get("tool_call_id"):
                item["tool_call_id"] = msg["tool_call_id"]
            formatted.append(item)
            continue

        if "content" in msg:
            formatted.append({"role": role, "content": msg["content"]})
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
    print("Загрузка оптимизированной Gemma-4 в Unified Memory...")
    model, tokenizer = load(MODEL_SOURCE, size=MODEL_SIZE)
    print("Модель успешно загружена и готова к работе через MLX!")
    yield
    print("Выгрузка модели из памяти...")

app = FastAPI(title="TheStage AI Edge-LM OpenAI Server", lifespan=lifespan)

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "thestageai"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(body: Dict[str, Any]):
    global model, tokenizer
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Модель еще не загружена в память")
    
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="Массив 'messages' пуст или отсутствует")
    
    # Забираем то, что просит Pi, но если там пусто или мало — ставим 2048
    max_tokens = body.get("max_tokens", 16000)
    if max_tokens < 512: 
        max_tokens = 16000
    model_field = body.get("model", MODEL_NAME)
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
    
    chat_id = f"chatcmpl-{int(time.time())}"
    max_prompt_tokens = max(1, MAX_CONTEXT_TOKENS - max_tokens)
    ids = build_prompt_ids(formatted_messages, template_kwargs, max_prompt_tokens)
    print(
        f"[{chat_id}] prompt={len(ids)} tokens, max_new={max_tokens}, "
        f"context_limit={MAX_CONTEXT_TOKENS}, include_usage={include_usage}",
        flush=True,
    )

    # --- ИСПРАВЛЕННЫЙ АСИНХРОННЫЙ ГЕНЕРАТОР (Остается в потоке MLX) ---
    async def generate_chunks_async():
        # Отправляем начальный чанк с ролью
        initial_chunk = {
            "id": chat_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_field,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            "usage": None,
        }
        yield f"data: {json.dumps(initial_chunk)}\n\n"

        text_parts = []
        generated_tokens = 0
        started_at = time.perf_counter()
        last_log_at = started_at
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
            # Даем FastAPI передохнуть и отправить чанк, оставаясь в текущем потоке
            await asyncio.sleep(0)

        generated_text = "".join(text_parts)
        elapsed = max(time.perf_counter() - started_at, 1e-9)
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
        
        # Отправляем финальный чанк останова
        final_chunk = {
            "id": chat_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_field,
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

    # Передаем асинхронный генератор в StreamingResponse
    return StreamingResponse(generate_chunks_async(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
