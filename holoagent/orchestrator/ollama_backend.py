"""Self-hosted local model backend via Ollama (TRD §6: "use an existing
runtime... do not write an inference engine").

Chosen over the TRD's own Phase 2 pick (DeepSeek-V4-Flash, 4-bit) and over
Kimi K3 for the actual hardware this runs on — an RTX 3050 Laptop GPU
(4GB VRAM), not the abstract "500 GB disk budget, don't worry about VRAM"
target the TRD's model landscape (§2) was really sized for:

  - Kimi K3: TRD §2 already rules this out (1.7TB storage, ~26s/token at
    the memory floor even on capable hardware). Worse here.
  - DeepSeek-V4-Flash: "only 13B active params" doesn't help when the GPU
    can't hold even one expert's working set in 4GB VRAM -- it would run
    almost entirely off CPU/disk, hitting the exact bottleneck the TRD
    itself names in §5 ("~80% of per-token time is disk I/O").

No new Python dependency: talks to Ollama's local REST API
(http://localhost:11434/api/chat) with the stdlib's urllib, since Ollama
is already the runtime (TRD §6's "do not write an inference engine" cuts
both ways -- don't add a client SDK for it either when a dozen lines of
stdlib HTTP does the job).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from typing import Any

from holoagent.orchestrator.backend import (
    AssistantToolCallTurn,
    ModelStep,
    ToolCall,
    ToolResultTurn,
    Turn,
    UserTurn,
)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
# qwen2.5-coder:7b (also on this machine) was tried first since it was
# already pulled and `ollama show` reports a "tools" capability flag for
# it -- but in practice it emits tool-call-shaped JSON as plain response
# text instead of populating Ollama's structured tool_calls field (verified
# directly against /api/chat, not assumed). It's code-completion-tuned,
# not instruction/agent-tuned, and the capability flag doesn't guarantee
# reliable triggering. qwen2.5:7b-instruct correctly uses tool_calls for
# the same request and is the default here instead.
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"


def ollama_server_running(host: str = DEFAULT_OLLAMA_HOST) -> bool:
    try:
        with urllib.request.urlopen(f"{host}/api/version", timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


class OllamaBackend:
    """Requires a running local Ollama server with a tool-calling-capable
    model already pulled — construction fails loudly, no silent fallback
    to some other backend, per the tool contract's no-silent-defaults
    principle (TRD §3.2).
    """

    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, host: str = DEFAULT_OLLAMA_HOST, timeout: float = 300.0):
        if not ollama_server_running(host):
            raise RuntimeError(
                f"OllamaBackend requires a running Ollama server at {host} — "
                "start it with `ollama serve` (or the Ollama app) first."
            )
        self._model = model
        self._host = host
        self._timeout = timeout

    def _to_ollama_messages(self, system_prompt: str, history: list[Turn]) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        for turn in history:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.text})
            elif isinstance(turn, AssistantToolCallTurn):
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": tc.name, "arguments": tc.arguments}} for tc in turn.tool_calls
                        ],
                    }
                )
            elif isinstance(turn, ToolResultTurn):
                for r in turn.results:
                    messages.append({"role": "tool", "content": r["text"]})
            else:
                raise TypeError(f"unknown turn type: {type(turn)}")
        return messages

    @staticmethod
    def _to_ollama_tools(tool_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["input_schema"],
                },
            }
            for spec in tool_specs
        ]

    def step(self, system_prompt: str, history: list[Turn], tool_specs: list[dict[str, Any]]) -> ModelStep:
        body = json.dumps(
            {
                "model": self._model,
                "messages": self._to_ollama_messages(system_prompt, history),
                "tools": self._to_ollama_tools(tool_specs),
                "stream": False,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self._host}/api/chat", data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Ollama /api/chat returned {exc.code}: {exc.read().decode('utf-8', 'replace')}") from exc

        message = payload["message"]
        raw_tool_calls = message.get("tool_calls") or []

        if raw_tool_calls:
            tool_calls = [
                # Prefer Ollama's own id when present; some models/versions
                # omit it, so fall back to a generated one rather than fail.
                ToolCall(
                    id=tc.get("id") or uuid.uuid4().hex,
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                )
                for tc in raw_tool_calls
            ]
            return ModelStep(tool_calls=tool_calls, final_text=None)

        return ModelStep(tool_calls=[], final_text=message.get("content", ""))
