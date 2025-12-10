"""Pytest configuration and shared fixtures"""

import pytest
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_audio_path():
    """Fixture to provide path to sample audio file"""
    return Path(__file__).parent.parent / "data" / "samples" / "test_audio.wav"
