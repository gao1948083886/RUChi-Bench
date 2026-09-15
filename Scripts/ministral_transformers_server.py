"""Small OpenAI-compatible text server for Ministral 3 FP8 checkpoints.

Xinference/vLLM 0.19.1 cannot currently load the Ministral-3 fine-grained FP8
checkpoint.  Transformers provides the model's supported loading path, so this
server is intentionally limited to the text-only chat API used by the pilot.
The parent process sets CUDA_VISIBLE_DEVICES to the requested physical GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from threading import Lock
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend


def make_app(model_path: str, model_name: str) -> FastAPI:
    print(f"LOADING_TOKENIZER {model_path}", flush=True)
    tokenizer = MistralCommonBackend.from_pretrained(model_path)
    print(f"LOADING_MODEL {model_path}", flush=True)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
    )
    model.eval()
    print("MODEL_READY", flush=True)

    app = FastAPI()
    generation_lock = Lock()

    def generate(messages: list[dict[str, Any]], max_new_tokens: int) -> tuple[str, int, int]:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            normalized.append({"role": message.get("role", "user"), "content": content})
        inputs = tokenizer.apply_chat_template(
            normalized,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = inputs["input_ids"]
        moved = {
            key: value.to("cuda") if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with generation_lock, torch.inference_mode():
            output = model.generate(
                **moved,
                max_new_tokens=max(1, min(int(max_new_tokens), 64)),
                do_sample=False,
            )
        prompt_tokens = int(input_ids.shape[-1])
        completion_tokens = int(output.shape[-1] - prompt_tokens)
        text = tokenizer.decode(output[0][prompt_tokens:])
        return text, prompt_tokens, completion_tokens

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": model_name}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> JSONResponse:
        payload = await request.json()
        try:
            messages = payload["messages"]
            text, prompt_tokens, completion_tokens = await asyncio.to_thread(
                generate, messages, payload.get("max_tokens", 8)
            )
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                status_code=500,
                content={"error": {"message": f"{type(exc).__name__}: {exc}"}},
            )
        return JSONResponse(
            {
                "id": f"ministral-{time.time_ns()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    uvicorn.run(make_app(args.model_path, args.model_name), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
