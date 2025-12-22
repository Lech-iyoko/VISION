# Setup Guide

## Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Lech-iyoko/VISION.git
cd VISION
```

### 2. Create and activate virtual environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
# Production dependencies
pip install -r requirements.txt

# Development dependencies (optional)
pip install -r requirements-dev.txt

# Or install as editable package
pip install -e .
```

### 4. Configure environment variables
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API keys
# - GROQ_API_KEY
# - ELEVENLABS_API_KEY
# - DEEPGRAM_API_KEY
```

### 5. Run the application
```bash
# As a module
python -m vision

# Or if installed
vision
```

## Development Setup

### Install development dependencies
```bash
pip install -r requirements-dev.txt
```

### Run tests
```bash
pytest
```

### Code formatting
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/
```

## Troubleshooting
- Ensure all API keys are properly set in `.env`
- Check Python version compatibility
- Verify audio device permissions for voice streaming
