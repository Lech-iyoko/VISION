# Architecture

## Project Structure
```
VISION/
├── src/vision/           # Main application package
│   ├── core/            # Core business logic
│   ├── services/        # External service clients
│   ├── models/          # ML models and processors
│   ├── utils/           # Utility functions
│   └── config/          # Configuration management
├── tests/               # Test suite
├── docs/                # Documentation
├── data/                # Data files
└── scripts/             # Utility scripts
```

## Component Overview

### Core Layer
- **Orchestrator**: Main coordination logic that manages the voice interaction loop

### Services Layer
- **Voice Streamer**: Real-time audio streaming and transcription
- **LLM Client**: Language model integration
- **TTS Client**: Text-to-speech synthesis

### Models Layer
- ML model processors and inference logic

### Utils Layer
- Shared utility functions
- State feedback mechanisms

## Data Flow
1. User speaks → Voice Streamer captures audio
2. Voice Streamer → Deepgram ASR → Text transcript
3. Transcript → LLM Client → Response generation
4. Response → TTS Client → Audio output
5. Loop continues...

## Configuration
- Environment-based configuration
- Centralized settings in `config/settings.py`
- API keys loaded from environment variables
