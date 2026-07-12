# VISION

[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/Lech-iyoko/VISION)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A real-time voice and vision AI assistant — designed to be what JARVIS was to Tony Stark. VISION listens, sees, thinks, and speaks. The voice pipeline (ASR + TTS) runs fully local on consumer GPU hardware; reasoning runs on Gemini 2.5 Flash-Lite. It perceives both the physical world (webcam) and the digital workspace (screen), and streams responses to speech as they are generated.

---

## Architecture

```
Microphone ──→ Silero VAD ──→ faster-whisper (local ASR, GPU)
                                      │ transcript
                                      ▼
Screen ──→ pixel-diff gate ──→ Gemini Flash-Lite ──→ rolling workspace state
   (every 5s, background)       (only on change)          │
                                                          ▼
                                                    FusionEngine
                                                          │ grounded prompt
Webcam ──→ fresh frame at turn time ──────────────────────┤ (image attached)
                                                          ▼
                                        Gemini 2.5 Flash-Lite (streaming LLM)
                                                          │ token stream
                                                          ▼
                                                  Sentence buffer
                                                          │ first sentence ready
                                                          ▼
                                          Kokoro-82M (local TTS, 24kHz)
                                                          ▼
                                                      Speaker
```

Two key design decisions:

1. **Hybrid local/cloud split.** Everything latency-critical and always-on (VAD, ASR, TTS) runs locally on the GPU — no per-turn network round-trips for audio. The LLM runs in the cloud where a stronger model is affordable per token. Backends are switchable in `config.py` (`ASR_BACKEND`, `TTS_BACKEND`), so the old cloud voice stack (AssemblyAI + ElevenLabs) remains one flag away.

2. **Dual visual grounding.** The camera frame is attached raw to the prompt (physical world). The screen is tracked by a background thread — captured every 5 seconds, gated by a pixel diff so only meaningful changes trigger a Gemini description, accumulated into a rolling temporal buffer ("32s ago: …"). The system prompt teaches the model to keep the two sources separate: `[DISPLAY]` for screen questions, the attached image for physical ones.

**Streaming pipeline:** LLM tokens are buffered into sentences and sent to TTS as each sentence completes. One audio output stream stays open across all sentences, so the user hears the first sentence while the model is still generating the rest.

---

## Features

- **Fully local voice loop** — Silero VAD segmentation + faster-whisper (`small.en`) ASR, Kokoro-82M TTS
- **Screen awareness** — background tracker maintains a temporal description of the user's workspace; VISION can answer "what's on my screen?" and reference code, terminals, and errors
- **Live camera vision** — single fresh frame captured at turn time, attached directly to the LLM prompt
- **Barge-in** — Silero VAD detects speech during playback; user can interrupt mid-sentence
- **Acoustic Echo Cancellation** — prevents TTS audio from being picked up as mic input
- **Streaming LLM → TTS** — sentence-buffered pipeline minimizes time-to-first-audio
- **Evaluation logger** — per-turn JSONL logs capturing `time_to_first_audio_ms`, `llm_latency_ms`, `vision_fetch_ms`, and response quality

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | [Gemini 2.5 Flash-Lite](https://ai.google.dev/) (streaming, thinking budget configurable) |
| Speech-to-Text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `small.en` (local, GPU) |
| Text-to-Speech | [Kokoro-82M](https://github.com/hexgrad/kokoro) (local, 24kHz PCM) |
| Screen capture | [mss](https://github.com/BoboTiG/python-mss) + OpenCV pixel diff |
| Screen description | Gemini 2.5 Flash-Lite (only on meaningful change) |
| Voice Activity Detection | [Silero VAD](https://github.com/snakers4/silero-vad) (local) |
| Acoustic Echo Cancellation | Custom correlation-based AEC |
| Audio I/O | SoundDevice / PortAudio |
| Language | Python 3.10+ |

Alternate backends (flag-switchable): [AssemblyAI](https://www.assemblyai.com/) streaming ASR, [ElevenLabs](https://elevenlabs.io/) TTS, `gemma4:e2b` via [Ollama](https://ollama.com) for local LLM evaluation.

---

## Latency Benchmarking

The `evals/` directory contains per-session JSONL logs and a three-way benchmark (`benchmark_multimodal.py`) from the earlier local-LLM phase, comparing a cloud cascade (Groq + Llama4-Scout VLM), a local dual-model pipeline (qwen3:8b + moondream), and a local single multimodal model (gemma4:e2b):

| Metric | qwen3 + moondream | gemma4:e2b (streaming) |
|---|---|---|
| Vision fetch (warm) | ~16,000 ms (CPU inference) | ~100 ms (camera read) |
| Time to first audio | not measured | 4,200 ms avg (thinking ON) |
| E2E avg (incl. playback) | 24,700 ms | 4,100–13,000 ms |

Those results motivated the current architecture: the screen tracker pre-builds visual context in the background (zero cost at turn time), and `THINKING_ENABLED = False` disables the LLM's chain-of-thought for lower time-to-first-audio.

---

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU (8GB+ VRAM — runs faster-whisper and Kokoro)
- `espeak-ng` system package (Kokoro dependency): `sudo apt install espeak-ng`
- A [Gemini API key](https://aistudio.google.com/apikey)

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_key_here

# Optional — only needed if switching back to cloud voice backends
ASSEMBLYAI_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
```

### 3. Run

```bash
python orchestrator.py
```

Speak into your microphone. VISION listens, fuses the transcript with screen context and a fresh camera frame, generates a response, and streams it to your speakers. Press `Ctrl+C` to exit.

---

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `GEMINI_LLM_MODEL` | `gemini-2.5-flash-lite` | Active LLM |
| `ASR_BACKEND` | `local` | `local` (faster-whisper) or `assemblyai` |
| `TTS_BACKEND` | `kokoro` | `kokoro` or `elevenlabs` |
| `WHISPER_MODEL` | `small.en` | Local ASR model size |
| `KOKORO_VOICE` | `bm_george` | Local TTS voice |
| `THINKING_ENABLED` | `False` | LLM chain-of-thought (off = lower latency) |
| `SCREEN_CAPTURE_INTERVAL_S` | `5.0` | Seconds between screen capture attempts |
| `SCREEN_DIFF_THRESHOLD` | `0.02` | Pixel-change fraction that triggers a screen description |
| `SYSTEM_PROMPT` | JARVIS-style persona | Injected on every LLM call |

---

## Project Structure

```
orchestrator.py            # Main coordinator and conversation state machine
config.py                  # All settings, backend switches, and the system prompt
services/
  gemini_llm_client.py     # Gemini 2.5 Flash-Lite streaming client (active LLM)
  llm_client.py            # OllamaMultimodalClient (local LLM, eval use)
  vision/
    screen_state.py        # Background screen tracker → temporal workspace state
    screen_capture.py      # mss screen grab + pixel-diff change gate
    frame_capture.py       # Webcam capture with buffer flush
    fusion_engine.py       # Combines transcript + [DISPLAY] + [CAMERA] into one prompt
  voice/
    local_voice_streamer.py  # Silero VAD + faster-whisper ASR (active)
    kokoro_tts_client.py     # Kokoro-82M local TTS (active)
    voice_streamer.py        # AssemblyAI streaming ASR (alternate)
    tts_client.py            # ElevenLabs TTS client (alternate)
    audio_frontend.py        # AEC + VAD pipeline
    echo_guard.py            # Text-level echo detection
utils/
  eval_logger.py           # Per-turn JSONL evaluation logger
evals/
  benchmark_multimodal.py  # Three-way latency benchmark script
docs/
  code-review-map.md       # Guided reading order for the codebase
```

---

## Roadmap

- [x] Local voice stack (faster-whisper ASR + Kokoro TTS)
- [x] Screen awareness with temporal workspace context
- [ ] Conversation memory (sliding window history)
- [ ] Tool use / function calling
- [ ] Proactive assistance — screen tracker's trigger mechanism exists but is unwired
- [ ] Raspberry Pi 5 integration — offload voice pipeline to free GPU for robotics work
