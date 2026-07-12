# VISION — Code Review Map

A reading guide for an architecture + implementation refresher. Ordered so each file builds on the previous one. Line counts are approximate as of 2026-07-08.

---

## 1. Start here — the spine

| File | Role | What to review |
|---|---|---|
| `orchestrator.py` (~320 lines) | Central coordinator. State machine (IDLE→LISTENING→THINKING→SPEAKING), turn handling, the streaming LLM→sentence→TTS pipeline, barge-in plumbing, eval logging. | `_handle_transcript` is the whole turn lifecycle. `_stream_response_to_speech` is the latency-critical path — sentence regex splitting, single persistent `RawOutputStream`, barge-in checks between TTS chunks. Note: `ScreenStateTracker` is constructed **without** the proactive callback — the trigger exists but is unwired. |
| `config.py` (~80 lines) | All settings + system prompt. Backend switches: `ASR_BACKEND` ("local"), `TTS_BACKEND` ("kokoro"), `GEMINI_LLM_MODEL` (active LLM). | The backend flags decide which client classes are live. The `SYSTEM_PROMPT` defines the camera-vs-screen disambiguation contract that `fusion_engine.py` depends on. |

## 2. Voice pipeline (active path)

| File | Role | What to review |
|---|---|---|
| `services/voice/local_voice_streamer.py` (~300 lines) | **Active ASR.** Mic → Silero VAD (segmentation + barge-in) → faster-whisper on GPU. Mute/cooldown, end-of-turn = 480ms silence. | The dual use of one VAD for segmentation *and* barge-in. Fixed silence threshold is the main latency bottleneck. Bug: unreachable code after `raise` at the end of `_setup_whisper` (lines ~114–116). |
| `services/voice/kokoro_tts_client.py` (~44 lines) | **Active TTS.** Kokoro-82M local, 24kHz int16 PCM. | Interface contract: `stream_sentence(text) -> Iterator[bytes]` + `sample_rate` + `audio_format` class attrs. This is the swap seam. |
| `services/voice/audio_frontend.py` (~260 lines) | AEC + noise suppression + VAD frame processing. `feed_tts_reference()` receives outgoing TTS audio for echo cancellation. | Whether the AEC reference path is actually effective — latest eval showed a barge-in firing before any audio played (`ttfa=-1`), suspect VAD reset path. |
| `services/voice/echo_guard.py` (~50 lines) | Text-level echo detection — safety net comparing ASR output to last TTS text. | Simple similarity threshold (0.6). Quick read. |

## 3. Vision pipeline (active path)

| File | Role | What to review |
|---|---|---|
| `services/vision/screen_state.py` (~230 lines) | Background thread: screen every 5s → pixel-diff gate → Gemini 2.5 Flash-Lite description → rolling 10-event temporal buffer. Keyword-based proactive trigger (unwired). | `get_workspace_state()` produces the temporal "Ns ago: …" context. The `_SIGNAL_KEYWORDS` proactive mechanism. Staleness: up to 5s+ old at turn time. |
| `services/vision/screen_capture.py` (~82 lines) | `mss` screen grab + OpenCV pixel diff (2% threshold) gating cloud calls. | Diff fragility (cursor blink vs real change). `mss.MSS()` created on main thread, used from background thread — thread-safety risk on some platforms. |
| `services/vision/frame_capture.py` (~59 lines) | Webcam capture at turn time. Buffer flush (`CAP_PROP_BUFFERSIZE=1` + grabs) so the frame is current. | Quick read — the buffer-flush trick is the whole point. |
| `services/vision/fusion_engine.py` (~41 lines) | Combines `[Conversation Memory]` / `[DISPLAY]` / `[CAMERA]` / `[User Said]` into one prompt. | The `memory` slot exists but is never populated by the orchestrator — this is where sliding-window memory plugs in. |

## 4. LLM client (active path)

| File | Role | What to review |
|---|---|---|
| `services/gemini_llm_client.py` (~59 lines) | **Active LLM.** Gemini 2.5 Flash-Lite streaming, thinking budget 0, PIL image attached natively. | Interface contract: `generate_streaming(prompt_text, image) -> Generator[str]`. Same seam pattern as TTS. |

## 5. Instrumentation

| File | Role | What to review |
|---|---|---|
| `utils/eval_logger.py` (~60 lines) | One JSONL record per turn: `time_to_first_audio_ms`, `llm_latency_ms`, `vision_fetch_ms`, `e2e_to_speech_ms`, barge-in flag. | Whether the metrics still match what the orchestrator measures. |
| `evals/benchmark_multimodal.py` | Three-way latency benchmark (cloud cascade vs local dual-model vs local single-model). | Historical — predates the Gemini migration. |
| `evals/session_*.jsonl` | 30+ real session logs, 2026-05-09 → 2026-07-01. | The ground truth on latency. Recent ttfa: ~2.2–6.7s. |

## 6. Legacy / inactive — skim only to understand history

| File | Status |
|---|---|
| `services/voice/voice_streamer.py` (~351 lines) | AssemblyAI ASR — inactive (`ASR_BACKEND="local"`), kept as fallback. |
| `services/voice/tts_client.py` (~65 lines) | ElevenLabs TTS — inactive (`TTS_BACKEND="kokoro"`), kept as fallback. Has uncommitted changes. |
| `services/llm_client.py` (~73 lines) | OllamaMultimodalClient (gemma4:e2b) — inactive, kept for local eval. |
| `services/vision/vision_manager.py` (~135 lines) | Old background VLM polling loop — superseded by `screen_state.py`. Candidate for deletion. |
| `services/vision/vision_model.py` (~67 lines) | Old moondream/Ollama VLM wrapper — superseded. Candidate for deletion. |
| `services/voice_streamer.py`, `services/tts_client.py` | **Empty (0 lines)** — dead leftovers from the directory restructure. Delete. |
| `models/vjepa_processor.py`, `utils/state_feedback.py` | Placeholders for v2 (V-JEPA, state feedback UI). No code. |

## 7. Docs — read last, with a critical eye

| File | Note |
|---|---|
| `README.md` | **Out of date** — still describes gemma4:e2b + AssemblyAI + ElevenLabs; actual stack is Gemini + faster-whisper + Kokoro. |
| `docs/architecture.md`, `docs/api.md`, `docs/setup.md` | Verify against current code; likely predate the Gemini migration. |
| `AGENTS.md` | Design principles + learning goals — still accurate, the architecture rules here are the review criteria. |
| `vision.service` | systemd unit — check it points at the right entrypoint/venv. |

---

## Suggested review passes

1. **Architecture pass** — `orchestrator.py` + `config.py` + `fusion_engine.py`: does the swappable-client discipline (per AGENTS.md) still hold everywhere?
2. **Latency pass** — `local_voice_streamer.py` → `gemini_llm_client.py` → `kokoro_tts_client.py`: trace one turn end-to-end and account for every serial millisecond against the eval logs.
3. **Concurrency pass** — three threads touch shared state (mic callback, screen tracker, whisper executor): check locks in `orchestrator.py`, `screen_state.py`, `local_voice_streamer.py`, and the mss thread-safety issue.
4. **Hygiene pass** — delete the empty/legacy files in §6, update README/docs.
