"""Confirms the vLLM server is reachable and returns reasoning_content."""

from openai import OpenAI

BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3.5-27B"

client = OpenAI(base_url=BASE_URL, api_key="EMPTY")


def run(label, user_text, extra_body):
    print(f"\n========== {label} ==========")
    print(f"user: {user_text!r}")
    print(f"extra_body: {extra_body!r}")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": user_text}],
        temperature=0.0,
        max_tokens=2048,
        extra_body=extra_body,
    )
    msg = resp.choices[0].message
    # vLLM with --reasoning-parser qwen3 returns the thinking trace as
    # `reasoning` (OpenAI-style name) on this server build; older builds use
    # `reasoning_content`. Check both.
    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
    if reasoning is None:
        raw = getattr(msg, "model_extra", None) or {}
        reasoning = raw.get("reasoning") or raw.get("reasoning_content")
    content = msg.content
    print("\n--- reasoning_content ---")
    print(reasoning if reasoning else "<EMPTY/MISSING>")
    print("\n--- content ---")
    print(content if content else "<EMPTY/MISSING>")
    print(f"\nreasoning_content present: {bool(reasoning)} | content present: {bool(content)}")
    return bool(reasoning)


print(f"Hitting {BASE_URL} with model={MODEL} ...")

base_q = "What is 17 * 23? Think carefully."

a = run("A: chat_template_kwargs.enable_thinking=True",
        base_q,
        {"chat_template_kwargs": {"enable_thinking": True}})

b = run("B: /think trigger in prompt, no extra_body",
        base_q + " /think",
        {})

c = run("C: /think trigger AND enable_thinking=True",
        base_q + " /think",
        {"chat_template_kwargs": {"enable_thinking": True}})

print("\n=== summary ===")
print(f"A (enable_thinking only)        reasoning_content? {a}")
print(f"B (/think trigger only)         reasoning_content? {b}")
print(f"C (both)                        reasoning_content? {c}")

if not (a or b or c):
    print(
        "\nWARNING: reasoning_content empty in all three configurations. "
        "Either --reasoning-parser qwen3 is not active, or this checkpoint "
        "is not a thinking-capable variant. Stop and investigate before "
        "running inference."
    )
