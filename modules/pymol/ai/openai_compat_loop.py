"""OpenAI-compatible chat+tools agent loop for non-Anthropic providers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .openrouter_client import OpenRouterClient, OpenRouterClientError
from .providers import ProviderSpec, get_provider_spec, resolve_openai_compat_base_url


@dataclass
class OpenAICompatTurnResult:
    assistant_text: str = ""
    error: Optional[str] = None
    error_class: Optional[str] = None
    interrupted: bool = False
    num_turns: Optional[int] = None
    session_id: Optional[str] = None


def _pymol_tools_schema() -> List[Dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "run_pymol_command",
                "description": "Run one or more PyMOL commands in the current session.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "capture_viewer_snapshot",
                "description": "Capture current PyMOL viewport screenshot and compact viewer state summary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "purpose": {"type": "string"},
                    },
                },
            },
        },
    ]


class OpenAICompatLoop:
    def run_turn(
        self,
        *,
        prompt: str,
        model: str,
        api_key: str,
        provider_id: Optional[str] = None,
        system_prompt: str = "",
        max_turns: int = 8,
        messages: Optional[List[Dict[str, object]]] = None,
        on_text_chunk: Optional[Callable[[str], None]] = None,
        on_reasoning_chunk: Optional[Callable[[str], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        run_command_tool: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
        snapshot_tool: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
        **_ignored: Any,
    ) -> OpenAICompatTurnResult:
        spec = get_provider_spec(provider_id)
        if not api_key:
            return OpenAICompatTurnResult(
                error="API key is not set for provider %s." % (spec.id,),
                error_class="auth_error",
            )

        base_url = resolve_openai_compat_base_url(spec)
        client = OpenRouterClient(api_key=api_key, base_url=base_url)

        history: List[Dict[str, object]] = list(messages or [])
        if system_prompt:
            history.insert(0, {"role": "system", "content": system_prompt})
        history.append({"role": "user", "content": prompt})

        final_text = ""
        turns = 0
        try:
            for step in range(max(1, int(max_turns))):
                if should_cancel and should_cancel():
                    return OpenAICompatTurnResult(
                        assistant_text=final_text,
                        interrupted=True,
                        num_turns=turns,
                    )
                turns = step + 1
                result = client.stream_assistant_turn(
                    model=model,
                    messages=history,
                    tools=_pymol_tools_schema(),
                    on_text_chunk=on_text_chunk or (lambda _t: None),
                    on_reasoning_chunk=on_reasoning_chunk,
                    should_cancel=should_cancel,
                )
                assistant_text = str(result.get("assistant_text") or "")
                if assistant_text:
                    final_text = assistant_text
                tool_calls = list(result.get("tool_calls") or [])
                assistant_msg: Dict[str, object] = {"role": "assistant", "content": assistant_text}
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments_json
                                or json.dumps(tc.arguments or {}),
                            },
                        }
                        for tc in tool_calls
                    ]
                history.append(assistant_msg)

                if not tool_calls:
                    break

                for tc in tool_calls:
                    if should_cancel and should_cancel():
                        return OpenAICompatTurnResult(
                            assistant_text=final_text,
                            interrupted=True,
                            num_turns=turns,
                        )
                    payload: Dict[str, Any]
                    if tc.name == "capture_viewer_snapshot" and snapshot_tool:
                        payload = snapshot_tool(tc.tool_call_id, dict(tc.arguments or {}))
                        if isinstance(payload, dict) and "payload" in payload:
                            payload = dict(payload.get("payload") or {})
                    elif run_command_tool:
                        payload = run_command_tool(tc.tool_call_id, dict(tc.arguments or {}))
                    else:
                        payload = {"ok": False, "error": "tool callback unavailable"}
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.tool_call_id,
                            "content": json.dumps(payload, ensure_ascii=False),
                        }
                    )
        except OpenRouterClientError as exc:
            return OpenAICompatTurnResult(
                assistant_text=final_text,
                error=str(exc),
                error_class="provider_error",
                num_turns=turns,
            )
        except Exception as exc:  # noqa: BLE001
            return OpenAICompatTurnResult(
                assistant_text=final_text,
                error=str(exc),
                error_class="sdk_error",
                num_turns=turns,
            )

        return OpenAICompatTurnResult(assistant_text=final_text, num_turns=turns)
