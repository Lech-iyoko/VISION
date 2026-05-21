# VISION

[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/Lech-iyoko/VISION)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A real-time voice and vision AI assistant — designed to be what JARVIS was to Tony Stark. VISION listens, sees, thinks, and speaks. It runs a local multimodal LLM on consumer GPU hardware and streams responses to speech as they are generated, keeping perceived latency under 5 seconds.

---

## Architecture

```
Microphone ──→ AssemblyAI (streaming ASR)
                        │
                        ▼
              Webcam ──→ gemma4:e2b (Ollama)   ← single multimodal model
                        │                         sees frame + transcript
                        ▼ token stream
              Sentence buffer
                        │ first sentence ready (~500ms into generation)
                        ▼
              ElevenLabs (streaming TTS)
                        │
                        ▼
                    Speaker
```

The key design decision: a **single multimodal model** (gemma4:e2b) replaces what was previously a two-model pipeline (separate LLM + VLM). The raw camera frame is attached directly to the prompt — no lossy text-description bottleneck.

**Streaming pipeline:** LLM tokens are buffered into sentences and sent to TTS as each sentence completes. One audio output stream stays open across all sentences, so the user hears the first sentence while the model is still generating the rest. Audio starts within ~500ms of the first output token.

---

## Features

- **Real-time voice interaction** — AssemblyAI Universal Streaming with end-of-turn detection
- **Live vision** — single fresh frame captured at turn time, attached directly to the LLM prompt
- **Barge-in** — Silero VAD detects speech during playback; user can interrupt mid-sentence
- **Acoustic Echo Cancellation** — prevents TTS audio from being picked up as mic input
- **Streaming LLM → TTS** — sentence-buffered pipeline reduces time-to-first-audio vs. waiting for full response
- **Evaluation logger** — per-turn JSONL logs capturing `time_to_first_audio_ms`, `llm_latency_ms`, `vision_fetch_ms`, and response quality

---

## Tech Stack

| Component | Technology |
|---|---|
| Multimodal LLM | `gemma4:e2b` via [Ollama](https://ollama.com) (local, RTX 5060 8GB) |
| Speech-to-Text | [AssemblyAI](https://www.assemblyai.com/) Universal Streaming |
| Text-to-Speech | [ElevenLabs](https://elevenlabs.io/) streaming |
| Voice Activity Detection | [Silero VAD](https://github.com/snakers4/silero-vad) (local) |
| Acoustic Echo Cancellation | Custom correlation-based AEC |
| Audio I/O | SoundDevice / PortAudio |
| Language | Python 3.10+ |

---

## Latency Benchmarking

The `evals/` directory contains a three-way benchmark (`benchmark_multimodal.py`) comparing:

1. **Cloud cascade** — Groq (llama-3.3-70b) + Llama4-Scout VLM
2. **Local dual-model** — qwen3:8b (GPU) + moondream (CPU)
3. **Local single multimodal** — gemma4:e2b (current)

Key results from production session logs:

| Metric | qwen3 + moondream | gemma4:e2b (streaming) |
|---|---|---|
| Vision fetch (warm) | ~16,000 ms (CPU inference) | ~100 ms (camera read) |
| Time to first audio | not measured | 4,200 ms avg (thinking ON) |
| E2E avg (incl. playback) | 24,700 ms | 4,100–13,000 ms |

The `THINKING_ENABLED` flag in `config.py` controls gemma4's internal chain-of-thought. With thinking OFF, `time_to_first_audio_ms` is expected to drop to under 1 second.

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- NVIDIA GPU (8GB+ VRAM recommended for gemma4:e2b)
- API keys for AssemblyAI and ElevenLabs

### 1. Pull the model

```bash
ollama pull gemma4:e2b
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file in the project root:

```env
ASSEMBLYAI_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
```

### 4. Run

```bash
python orchestrator.py
```

Speak into your microphone. VISION listens, captures a frame when you finish speaking, generates a response, and streams it to your speakers. Press `Ctrl+C` to exit.

---

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `DEFAULT_LLM_MODEL` | `gemma4:e2b` | Ollama model name |
| `THINKING_ENABLED` | `True` | Enable/disable chain-of-thought reasoning |
| `DEFAULT_VOICE_ID` | ElevenLabs voice ID | TTS voice |
| `SYSTEM_PROMPT` | JARVIS-style persona | Injected on every LLM call |

---

## Project Structure

```
orchestrator.py          # Main coordinator and conversation state machine
config.py                # All settings and the system prompt
services/
  llm_client.py          # OllamaMultimodalClient — streaming + vision
  vision/
    frame_capture.py     # Webcam capture with buffer flush
  voice/
    voice_streamer.py    # AssemblyAI streaming ASR + Silero VAD
    tts_client.py        # ElevenLabs TTS client
    audio_frontend.py    # AEC + VAD pipeline
    echo_guard.py        # Text-level echo detection
utils/
  eval_logger.py         # Per-turn JSONL evaluation logger
evals/
  benchmark_multimodal.py  # Three-way latency benchmark script
```

---

## Roadmap

- [ ] Conversation memory (sliding window history)
- [ ] Tool use / function calling
- [ ] Native speech-to-speech via Gemini Live API (target: <1s latency)
- [ ] Raspberry Pi 5 integration — offload voice pipeline to free GPU for robotics work
