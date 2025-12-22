"""Application settings and configuration"""

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"

# API Keys (load from environment variables)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Voice settings
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"

# Model settings
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
