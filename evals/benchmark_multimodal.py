"""
Latency benchmark: three-way comparison
  1. qwen3:8b (GPU) + moondream (CPU)  — current pipeline
  2. qwen2.5vl:7b                      — single multimodal, 7B
  3. gemma4:e2b                        — single multimodal, MoE 2.3B effective

Measures per candidate:
  - Cold-start latency (first call, includes model load)
  - Warm steady-state: text-only turns
  - Warm steady-state: vision+text turns
  - VRAM usage

Run from project root:
  python evals/benchmark_multimodal.py
"""

import sys
import time
import base64
import statistics
import subprocess
from pathlib import Path
from io import BytesIO

import openai
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SYSTEM_PROMPT, OLLAMA_BASE_URL


# ── helpers ────────────────────────────────────────────────────────────────

def vram_used_mb() -> int:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(r.stdout.strip())


def image_to_b64(path: str) -> str:
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def unload_all():
    """Tell Ollama to unload all models (keep_alive=0)."""
    for model in ["qwen2.5vl:7b", "qwen3:8b", "moondream"]:
        requests.post("http://localhost:11434/api/generate",
                      json={"model": model, "keep_alive": 0}, timeout=5)
    time.sleep(2)


def fmt(ms_list: list[float]) -> str:
    if not ms_list:
        return "no data"
    return (f"min={min(ms_list):.0f}ms  "
            f"avg={statistics.mean(ms_list):.0f}ms  "
            f"max={max(ms_list):.0f}ms")


# ── benchmark qwen2.5vl:7b ────────────────────────────────────────────────

def bench_qwen25vl(image_b64: str, n_warm: int = 1, n_timed: int = 4):
    client = openai.OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    model  = "qwen2.5vl:7b"

    text_prompt  = "What can you tell me about my current environment and what I might be working on?"
    vision_prompt = "I'm showing you a live camera feed. Describe what you see about me and my environment."

    def call(prompt: str, with_image: bool) -> float:
        content = [{"type": "text", "text": prompt}]
        if with_image:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            })
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": content},
            ],
            temperature=0.7,
            max_tokens=256,
        )
        ms = (time.perf_counter() - t0) * 1000
        return ms, resp.choices[0].message.content

    print("\n" + "═" * 60)
    print("  qwen2.5vl:7b — COLD START")
    print("═" * 60)
    unload_all()
    vram_before = vram_used_mb()
    cold_ms, cold_resp = call(vision_prompt, with_image=True)
    vram_after = vram_used_mb()
    print(f"  Cold-start latency : {cold_ms:.0f} ms")
    print(f"  VRAM delta         : {vram_before} → {vram_after} MiB (+{vram_after - vram_before} MiB)")
    print(f"  Response preview   : {cold_resp[:120]}...")

    print(f"\n  Warm-up ({n_warm} call(s), discarded)...")
    for _ in range(n_warm):
        call(text_prompt, with_image=False)

    # Text-only turns
    print(f"\n  TEXT-ONLY turns ({n_timed} runs):")
    text_times = []
    for i in range(n_timed):
        ms, resp = call(text_prompt, with_image=False)
        text_times.append(ms)
        print(f"    [{i+1}] {ms:.0f} ms  →  {resp[:80]}...")

    # Vision+text turns
    print(f"\n  VISION + TEXT turns ({n_timed} runs):")
    vision_times = []
    for i in range(n_timed):
        ms, resp = call(vision_prompt, with_image=True)
        vision_times.append(ms)
        print(f"    [{i+1}] {ms:.0f} ms  →  {resp[:80]}...")

    return {
        "cold_ms":        cold_ms,
        "text_only":      text_times,
        "vision_text":    vision_times,
        "vram_loaded_mb": vram_after,
    }


# ── benchmark current pipeline (qwen3:8b + moondream) ────────────────────

def bench_current_pipeline(image_b64: str, n_warm: int = 1, n_timed: int = 4):
    llm_client = openai.OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    def call_llm(prompt: str) -> tuple[float, str]:
        t0 = time.perf_counter()
        resp = llm_client.chat.completions.create(
            model="qwen3:8b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.7,
            max_tokens=256,
            extra_body={"think": False},
        )
        ms = (time.perf_counter() - t0) * 1000
        return ms, resp.choices[0].message.content

    def call_vlm(prompt: str) -> tuple[float, str]:
        t0 = time.perf_counter()
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "moondream",
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {"num_gpu": 0},
            },
            timeout=60,
        )
        ms = (time.perf_counter() - t0) * 1000
        return ms, r.json().get("response", "").strip()

    text_prompt   = "What can you tell me about my current environment and what I might be working on?"
    vision_prompt = "Describe what you see about this person and their environment concisely."

    print("\n" + "═" * 60)
    print("  CURRENT PIPELINE — qwen3:8b (GPU) + moondream (CPU)")
    print("═" * 60)
    unload_all()
    vram_before = vram_used_mb()

    # Cold start — VLM + LLM sequential
    print("  Cold-start (moondream → qwen3, sequential):")
    vlm_cold_ms, vlm_desc = call_vlm(vision_prompt)
    llm_cold_ms, llm_resp = call_llm(f"[Scene: {vlm_desc}]\n{text_prompt}")
    vram_after = vram_used_mb()
    total_cold = vlm_cold_ms + llm_cold_ms
    print(f"    moondream : {vlm_cold_ms:.0f} ms")
    print(f"    qwen3     : {llm_cold_ms:.0f} ms")
    print(f"    total     : {total_cold:.0f} ms")
    print(f"  VRAM delta  : {vram_before} → {vram_after} MiB (+{vram_after - vram_before} MiB)")

    print(f"\n  Warm-up ({n_warm} call(s), discarded)...")
    for _ in range(n_warm):
        call_llm(text_prompt)
        call_vlm(vision_prompt)

    # Text-only (LLM only, no vision)
    print(f"\n  TEXT-ONLY turns ({n_timed} runs, LLM only):")
    text_times = []
    for i in range(n_timed):
        ms, resp = call_llm(text_prompt)
        text_times.append(ms)
        print(f"    [{i+1}] {ms:.0f} ms  →  {resp[:80]}...")

    # Vision+text (moondream → qwen3, sequential)
    print(f"\n  VISION + TEXT turns ({n_timed} runs, moondream → qwen3):")
    vision_times, vlm_times, llm_times = [], [], []
    for i in range(n_timed):
        vlm_ms, desc   = call_vlm(vision_prompt)
        llm_ms, resp   = call_llm(f"[Scene: {desc}]\n{text_prompt}")
        total_ms        = vlm_ms + llm_ms
        vision_times.append(total_ms)
        vlm_times.append(vlm_ms)
        llm_times.append(llm_ms)
        print(f"    [{i+1}] vlm={vlm_ms:.0f}ms + llm={llm_ms:.0f}ms = {total_ms:.0f}ms")

    return {
        "cold_ms":        total_cold,
        "text_only":      text_times,
        "vision_text":    vision_times,
        "vlm_only":       vlm_times,
        "llm_only":       llm_times,
        "vram_loaded_mb": vram_after,
    }


# ── benchmark gemma4:e2b ──────────────────────────────────────────────────

def bench_gemma4_e2b(image_b64: str, n_warm: int = 1, n_timed: int = 4):
    client = openai.OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    model  = "gemma4:e2b"

    text_prompt   = "What can you tell me about my current environment and what I might be working on?"
    vision_prompt = "I'm showing you a live camera feed. Describe what you see about me and my environment."

    def call(prompt: str, with_image: bool) -> tuple[float, str]:
        content = [{"type": "text", "text": prompt}]
        if with_image:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            })
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": content},
            ],
            temperature=0.7,
            max_tokens=256,
        )
        ms = (time.perf_counter() - t0) * 1000
        return ms, resp.choices[0].message.content

    print("\n" + "═" * 60)
    print("  gemma4:e2b — COLD START")
    print("═" * 60)
    unload_all()
    vram_before = vram_used_mb()
    cold_ms, cold_resp = call(vision_prompt, with_image=True)
    vram_after = vram_used_mb()
    print(f"  Cold-start latency : {cold_ms:.0f} ms")
    print(f"  VRAM delta         : {vram_before} → {vram_after} MiB (+{vram_after - vram_before} MiB)")
    print(f"  Response preview   : {cold_resp[:120]}...")

    print(f"\n  Warm-up ({n_warm} call(s), discarded)...")
    for _ in range(n_warm):
        call(text_prompt, with_image=False)

    print(f"\n  TEXT-ONLY turns ({n_timed} runs):")
    text_times = []
    for i in range(n_timed):
        ms, resp = call(text_prompt, with_image=False)
        text_times.append(ms)
        print(f"    [{i+1}] {ms:.0f} ms  →  {resp[:80]}...")

    print(f"\n  VISION + TEXT turns ({n_timed} runs):")
    vision_times = []
    for i in range(n_timed):
        ms, resp = call(vision_prompt, with_image=True)
        vision_times.append(ms)
        print(f"    [{i+1}] {ms:.0f} ms  →  {resp[:80]}...")

    return {
        "cold_ms":        cold_ms,
        "text_only":      text_times,
        "vision_text":    vision_times,
        "vram_loaded_mb": vram_after,
    }


# ── summary ────────────────────────────────────────────────────────────────

def print_summary(current: dict, vl7b: dict, gemma: dict):
    print("\n" + "═" * 72)
    print("  SUMMARY — three-way comparison")
    print("═" * 72)

    def ms(val):
        return f"{val:.0f} ms"

    rows = [
        ("Cold-start latency",
         ms(current["cold_ms"]),
         ms(vl7b["cold_ms"]),
         ms(gemma["cold_ms"])),
        ("VRAM loaded",
         f"{current['vram_loaded_mb']} MiB",
         f"{vl7b['vram_loaded_mb']} MiB",
         f"{gemma['vram_loaded_mb']} MiB"),
        ("Text-only avg",
         ms(statistics.mean(current["text_only"])),
         ms(statistics.mean(vl7b["text_only"])),
         ms(statistics.mean(gemma["text_only"]))),
        ("Vision+text avg",
         ms(statistics.mean(current["vision_text"])),
         ms(statistics.mean(vl7b["vision_text"])),
         ms(statistics.mean(gemma["vision_text"]))),
        ("Vision+text best",
         ms(min(current["vision_text"])),
         ms(min(vl7b["vision_text"])),
         ms(min(gemma["vision_text"]))),
    ]

    col = 22
    print(f"\n  {'Metric':<{col}}  {'qwen3+moondream':>18}  {'qwen2.5vl:7b':>14}  {'gemma4:e2b':>12}")
    print("  " + "─" * (col + 50))
    for row in rows:
        label, *vals = row
        print(f"  {label:<{col}}  {vals[0]:>18}  {vals[1]:>14}  {vals[2]:>12}")

    baseline = statistics.mean(current["vision_text"])
    for name, result in [("qwen2.5vl:7b", vl7b), ("gemma4:e2b", gemma)]:
        avg  = statistics.mean(result["vision_text"])
        pct  = ((avg - baseline) / baseline) * 100
        icon = "✅" if pct < 0 else "⚠️ "
        sign = "" if pct < 0 else "+"
        print(f"\n  {icon} {name}: {sign}{pct:.0f}% vs current pipeline on vision+text.")

    print()


# ── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-current",  action="store_true", help="Skip qwen3+moondream pipeline")
    parser.add_argument("--skip-qwen25vl", action="store_true", help="Skip qwen2.5vl:7b")
    parser.add_argument("--skip-gemma",    action="store_true", help="Skip gemma4:e2b")
    parser.add_argument("--n-warm",  type=int, default=1)
    parser.add_argument("--n-timed", type=int, default=4)
    args = parser.parse_args()

    test_image_path = Path(__file__).parent.parent / "test_frame.jpg"
    if not test_image_path.exists():
        img = Image.new("RGB", (320, 240), color=(100, 149, 237))
        test_image_path = Path("/tmp/bench_frame.jpg")
        img.save(test_image_path)
        print(f"⚠️  test_frame.jpg not found — using synthetic image")

    print(f"Using test image : {test_image_path}")
    print(f"Warm-up / timed  : {args.n_warm} / {args.n_timed} runs each\n")
    image_b64 = image_to_b64(str(test_image_path))

    current_results = bench_current_pipeline(image_b64, args.n_warm, args.n_timed) \
        if not args.skip_current else None
    vl7b_results    = bench_qwen25vl(image_b64, args.n_warm, args.n_timed) \
        if not args.skip_qwen25vl else None
    gemma_results   = bench_gemma4_e2b(image_b64, args.n_warm, args.n_timed) \
        if not args.skip_gemma else None

    if current_results and vl7b_results and gemma_results:
        print_summary(current_results, vl7b_results, gemma_results)
