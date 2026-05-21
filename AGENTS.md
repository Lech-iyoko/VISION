# Copilot Agent Instructions for VISION

## Purpose of This Project
VISION is a multimodal, visually-grounded AI assistant. 
The goal is to understand the fundamentals of building intelligent systems from the ground up:
- orchestrator design
- ASR, TTS, LLMs, VLMs
- multimodal fusion
- conversation memory
- modular architecture
- clean data and logic flow

## How You Should Assist Me
- Provide code that is clean, modular, and easy to understand.
- Explain concepts when needed, especially around AI system design.
- Prioritize clarity over cleverness.
- When generating code, follow the existing architecture and folder structure.
- Use interfaces and abstraction layers so I can swap models (ASR, TTS, LLM, VLM).
- Help me understand the WHY behind design decisions.
- Avoid unnecessary complexity unless I explicitly ask for advanced implementations.

## Coding Style
- Python 3.10+
- Use type hints everywhere.
- Use async where appropriate.
- Keep functions small and single-purpose.
- Prefer composition over inheritance.
- Follow the existing naming conventions in the codebase.

## Architecture Principles
- Orchestrator is the central controller.
- Each modality (ASR, TTS, LLM, VLM) must be isolated behind a client class.
- No client should depend on another client directly.
- Orchestrator handles fusion, memory, and routing.
- Vision pipeline should be swappable (VLM now, V-JEPA later).
- Voice pipeline should be real-time and event-driven.

## What Not To Do
- Do not rewrite the entire project unless asked.
- Do not introduce unnecessary frameworks.
- Do not generate overly complex abstractions.
- Do not assume I want shortcuts — I want to learn the fundamentals.

## My Learning Goals
- Understand how intelligent systems are architected.
- Understand multimodal fusion.
- Understand how to design assistants that feel grounded and context-aware.
- Build a system I can extend with predictive models later (JEPA, PAN).
