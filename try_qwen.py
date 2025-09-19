#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A tiny wrapper around Qwen (DashScope compatible OpenAI API) chat completions.
- Exposes a class QwenChatClient for reusable calls
- Exposes a function call_qwen(message_list, ...) for one-off simple usage

Environment:
- DASHSCOPE_API_KEY must be set in environment variables unless an api_key is passed explicitly

Example:
    from try_qwen import call_qwen
    answer = call_qwen([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "用一句话介绍一下你自己"},
    ])
    print(answer)
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


class QwenChatClient:
    """
    Lightweight client for calling Qwen chat completions via DashScope-compatible OpenAI API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        default_model: str = DEFAULT_MODEL,
    ) -> None:
        key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError(
                "环境变量 DASHSCOPE_API_KEY 未设置，且未显式传入 api_key。"
            )
        self._client = OpenAI(api_key=key, base_url=base_url)
        self._default_model = default_model

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[str, Any]:
        """
        Send a chat completion request.

        Args:
            messages: List of {"role": str, "content": str}
            model: Override model name, defaults to self._default_model
            extra_body: Extra payload, e.g., {"enable_thinking": False}
            temperature: Optional sampling temperature
            **kwargs: Reserved for future extension

        Returns:
            (answer_text, raw_response)
        """
        payload: Dict[str, Any] = {
            "model": model or self._default_model,
            "messages": messages,
        }
        if extra_body is not None:
            payload["extra_body"] = extra_body
        if temperature is not None:
            payload["temperature"] = temperature

        completion = self._client.chat.completions.create(**payload)
        answer = completion.choices[0].message.content
        return answer, completion


def call_qwen(
    message_list: List[Dict[str, str]],
    model: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    """
    Convenience function to call Qwen with a message list.

    Args:
        message_list: List of {"role": str, "content": str}
        model: Optional model name override (default: qwen-plus)
        enable_thinking: If provided, sets extra_body = {"enable_thinking": <bool>}
        temperature: Optional sampling temperature
        api_key: Optional explicit API key; otherwise uses DASHSCOPE_API_KEY from env
        base_url: Optional base url override

    Returns:
        The assistant's answer text.
    """
    client = QwenChatClient(api_key=api_key, base_url=base_url, default_model=DEFAULT_MODEL)
    extra_body = None if enable_thinking is None else {"enable_thinking": enable_thinking}
    answer, _ = client.chat(
        messages=message_list,
        model=model,
        extra_body=extra_body,
        temperature=temperature,
    )
    return answer


def main() -> None:
    # Simple demo to show how to use the wrapper
    demo_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ]
    try:
        answer = call_qwen(demo_messages)
        print("=== Answer ===")
        print(answer)
    except Exception as e:
        print(f"[Error] 调用失败: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()