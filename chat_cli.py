import argparse
import json
import os
import sys
import re
from typing import List, Dict, Any, Optional, Union

import requests

# Default model name used when --model is not provided; override via OLLAMA_MODEL
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "banya-llama31-lora-merged")
# Ollama HTTP chat endpoint; override via OLLAMA_CHAT_URL
CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")

# Default generation options to encourage complete sentences and stable decoding
# Keys
# - num_predict: maximum tokens to generate
# - temperature/top_p/top_k: sampling controls (lower → more deterministic)
# - repeat_penalty/repeat_last_n: discourages repetition over last N tokens
# - stop: explicit stop sequences (empty → rely on guards/post-processing)
DEFAULT_OPTIONS: Dict[str, Any] = {
    "num_predict": 100,
    "temperature": 0.3,
    "top_p": 0.85,
    "top_k": 50,
    "repeat_penalty": 1.2,
    "repeat_last_n": 256,
    "stop": [],
}

# Built-in Korean system prompt describing safety, tone, and style
DEFAULT_SYSTEM_PROMPT_KO = (
    "너는 10대 발달장애인의 일상을 돕는 한국어 에이전트다. "
    "말은 간단하고 짧게 한다. 한 번에 한 단계씩 안내한다. "
    "위급한 상황이라고 판단될 경우 즉시 보호자나 119에 연락하도록 안내한다. 복잡한 요청은 다시 확인하고 "
    "필요한 정보를 먼저 묻는다. 일정 관리, 준비물 체크, 이동 안내, "
    "감정 조절 도움, 사회적 상황 대처 연습을 친절하게 돕는다. "
    "물결표와 이모티콘, 과도한 문장부호(!!!, .. 등)는 사용하지 않는다. 문장부호는 최대 1개만 사용한다."
)

# Built-in English system prompt, semantically equivalent to the Korean one
DEFAULT_SYSTEM_PROMPT_EN = (
    "You are an assistant that supports the daily life of teenagers with developmental disabilities. "
    "Use very simple and short sentences, and guide one step at a time. "
    "If the situation seems urgent or dangerous, immediately instruct to contact a guardian or emergency services (119). "
    "For complex requests, confirm first and ask for the necessary information. "
    "Kindly help with schedule management, preparation checklists, navigation, emotion regulation, and practicing social situations. "
    "Do not use tildes or emoticons, and avoid excessive punctuation (like !!! or ..). Use at most one punctuation mark at the end of a sentence."
)


def normalize_line(raw: Union[str, bytes]) -> Optional[str]:
    """Normalize one streaming line from Ollama.

    - Accepts bytes or str; decodes bytes to UTF-8 (ignore errors)
    - Trims whitespace and the optional SSE prefix "data: "
    - Returns a JSON string line or None if empty
    """
    if raw is None:
        return None
    if isinstance(raw, bytes):
        try:
            line = raw.decode("utf-8", errors="ignore")
        except Exception:
            return None
    else:
        line = raw
    line = line.strip()
    if not line:
        return None
    if line.startswith("data: "):
        line = line[len("data: ") :]
    return line


def parse_json_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from a single line; return None on failure."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def postprocess_completed_text(text: str) -> str:
    """Finalize assembled text after streaming ends.

    Applies project rules: replace tildes, collapse punctuation noise,
    ensure terminal punctuation.
    """
    # Replace tilde with period per user preference and ensure terminal punctuation
    cleaned = text.replace("~", ".")
    # Collapse repeated punctuation/emoticons
    cleaned = re.sub(r"!{2,}", "!", cleaned)
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    cleaned = re.sub(r"\?{2,}", "?", cleaned)
    cleaned = re.sub(r"\^{2,}", "", cleaned)
    cleaned = re.sub(r"ㅋ{2,}", "", cleaned)
    cleaned = cleaned.rstrip()
    if not cleaned:
        return cleaned
    if cleaned[-1] not in ".!?":
        cleaned = cleaned + "."
    return cleaned


def sanitize_delta(delta: str) -> str:
    """Clean up each incremental delta during streaming.

    Keeps output stable by collapsing punctuation and enforcing the
    tilde → period preference before printing.
    """
    # Streaming-time sanitization for stability
    delta = delta.replace("~", ".")
    # Aggressively collapse punctuation
    delta = re.sub(r"!{2,}", "!", delta)
    delta = re.sub(r"\.{2,}", ".", delta)
    delta = re.sub(r"\?{2,}", "?", delta)
    delta = re.sub(r"\^{2,}", "", delta)
    delta = re.sub(r"ㅋ{2,}", "", delta)
    return delta


def stream_chat(
    messages: List[Dict[str, str]],
    model: str,
    stream: bool = True,
    timeout: int = 600,
) -> str:
    """Call the Ollama chat API, optionally streaming partial deltas.

    Args:
        messages: conversation history (role/content pairs)
        model: Ollama model name
        stream: if True, print as we receive tokens
        timeout: HTTP timeout in seconds

    Returns:
        The assistant's full response text
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": DEFAULT_OPTIONS,
    }

    if stream:
        with requests.post(CHAT_URL, json=payload, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            full_text_parts: List[str] = []
            printed_chars = 0  # total characters printed so far
            punct_only_streak = 0  # count of consecutive punctuation-only deltas
            end_punct_emitted = False  # seen any sentence-ending punctuation
            PUNCT_ONLY_PATTERN = re.compile(r"^[\s\.!?]+$")  # delta is only punctuation/whitespace
            SENTENCE_END_PATTERN = re.compile(r"[\.!?]")  # to count sentences
            sentence_count = 0  # number of sentence ends seen
            MAX_SENTENCES = 2  # sentence limit guard
            MAX_OUTPUT_CHARS = 400  # hard character cap
            MAX_PUNCT_STREAK_AFTER_END = 2  # punctuation-only after end punct
            for raw in resp.iter_lines(decode_unicode=False):
                line = normalize_line(raw)
                if not line:
                    continue
                obj = parse_json_line(line)
                if not obj:
                    continue
                if "error" in obj:
                    raise RuntimeError(obj["error"])
                msg = obj.get("message")
                if isinstance(msg, dict):
                    delta = msg.get("content")
                    if delta:
                        delta = sanitize_delta(delta)
                        # Update guards
                        if PUNCT_ONLY_PATTERN.match(delta):
                            punct_only_streak += 1
                        else:
                            punct_only_streak = 0
                        if any(ch in delta for ch in ".!?"):
                            end_punct_emitted = True
                            sentence_count += len(SENTENCE_END_PATTERN.findall(delta))
                        printed_chars += len(delta)

                        # Stop if output is going off the rails
                        if sentence_count >= MAX_SENTENCES or (end_punct_emitted and punct_only_streak >= MAX_PUNCT_STREAK_AFTER_END) or printed_chars >= MAX_OUTPUT_CHARS:
                            # finalize and break
                            out = postprocess_completed_text("".join(full_text_parts))
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                            return out

                        if delta:
                            sys.stdout.write(delta)
                            sys.stdout.flush()
                            full_text_parts.append(delta)
                if obj.get("done") is True:
                    break
            sys.stdout.write("\n")
            sys.stdout.flush()
            return postprocess_completed_text("".join(full_text_parts))
    else:
        resp = requests.post(CHAT_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            if "message" in data and isinstance(data["message"], dict):
                return postprocess_completed_text(data["message"].get("content", ""))
            if "response" in data:
                return postprocess_completed_text(data["response"])
        return postprocess_completed_text(str(data))


def run_once(model: str, system_prompt: str, user_text: str, stream: bool) -> None:
    """Issue a single user prompt and print the assistant's reply."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    _ = stream_chat(messages, model=model, stream=stream)


def run_repl(model: str, system_prompt: str, stream: bool) -> None:
    """Interactive REPL that maintains conversation history and streams replies."""
    print(f"Model: {model}")
    print("명령: /exit 종료, /reset 초기화")
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text in {"/exit", ":q", "quit", "exit"}:
            break
        if user_text in {"/reset", ":r"}:
            messages = [{"role": "system", "content": system_prompt}]
            print("대화를 초기화했습니다.")
            continue
        messages.append({"role": "user", "content": user_text})
        print("Assistant:", end=" ")
        assistant_reply = stream_chat(messages, model=model, stream=stream)
        messages.append({"role": "assistant", "content": assistant_reply})


def main() -> None:
    """Parse CLI flags and run in one-shot or REPL mode.

    Flags:
      --model    : Ollama model name
      --system   : override system prompt text
      --lang     : choose built-in prompt (ko|en) when --system is not used
      --once     : send a single prompt and exit
      --no-stream: disable streaming and return response at once
    """
    parser = argparse.ArgumentParser(description="Interactive console chat with Ollama")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument(
        "--system", type=str, default=None, help="Override system prompt"
    )
    parser.add_argument(
        "--lang", type=str, choices=["ko", "en"], default="ko",
        help="Built-in system prompt language (ignored if --system is provided)"
    )
    parser.add_argument(
        "--once", type=str, default=None, help="Send a single prompt and exit"
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="Disable streaming responses"
    )
    args = parser.parse_args()

    system_prompt = args.system if args.system is not None else (
        DEFAULT_SYSTEM_PROMPT_EN if args.lang == "en" else DEFAULT_SYSTEM_PROMPT_KO
    )

    if args.once:
        run_once(model=args.model, system_prompt=system_prompt, user_text=args.once, stream=not args.no_stream)
    else:
        run_repl(model=args.model, system_prompt=system_prompt, stream=not args.no_stream)


if __name__ == "__main__":
    main()
