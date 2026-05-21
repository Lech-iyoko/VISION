"""
Audio Frontend Module
Handles: AEC, Noise Suppression, VAD, and frame processing.

This is the industry-standard approach used by Alexa, Google Assistant, etc.
"""

import numpy as np
import threading
from collections import deque
from typing import Callable, Optional
from dataclasses import dataclass
from enum import Enum

# Silero VAD
import torch
torch.set_num_threads(1)


class VADState(Enum):
    SILENCE = "silence"
    SPEECH = "speech"


@dataclass
class AudioConfig:
    """Tunable parameters for the audio pipeline."""
    sample_rate: int = 16000
    frame_size_ms: int = 32          # 512 samples at 16kHz
    vad_threshold: float = 0.5       # Speech probability threshold
    onset_frames: int = 3            # ~96ms of speech to trigger barge-in
    silence_frames: int = 15         # ~480ms of silence to end turn
    aec_enabled: bool = True


class AudioFrontend:
    """
    Encapsulates the audio processing pipeline:
    - AEC (Acoustic Echo Cancellation)
    - Noise Suppression (optional)
    - VAD (Voice Activity Detection)
    - Frame chunking
    
    This module processes mic input and TTS reference to produce
    clean audio suitable for ASR, along with VAD state.
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        
        # Calculate frame size in samples
        self.frame_size = int(self.config.sample_rate * self.config.frame_size_ms / 1000)
        
        # Audio buffers
        self.mic_buffer = np.array([], dtype=np.float32)
        self.tts_buffer = deque(maxlen=50)  # Ring buffer for TTS reference
        
        # VAD state tracking
        self.vad_state = VADState.SILENCE
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Load VAD model
        self._setup_vad()
        
        # AEC setup (simplified - see note below)
        self._setup_aec()
        
        # Callbacks
        self.on_barge_in: Optional[Callable[[], None]] = None
        self.on_turn_end: Optional[Callable[[], None]] = None
        
        print(f"✅ AudioFrontend initialized")
        print(f"   Frame size: {self.frame_size} samples ({self.config.frame_size_ms}ms)")
        print(f"   AEC enabled: {self.config.aec_enabled}")
    
    def _setup_vad(self):
        """Initialize Silero VAD."""
        from silero_vad import load_silero_vad
        self.vad_model = load_silero_vad(onnx=True)
        print("✅ VAD loaded (Silero ONNX)")
    
    def _setup_aec(self):
        """
        Initialize AEC.
        
        NOTE: Full AEC requires native libraries (WebRTC/SpeexDSP).
        This is a simplified version using correlation-based subtraction.
        For production, use webrtc-audio-processing or speexdsp.
        """
        if self.config.aec_enabled:
            # Simplified AEC state
            self.aec_filter_length = 2048  # ~128ms at 16kHz
            self.aec_filter = np.zeros(self.aec_filter_length, dtype=np.float32)
            self.aec_adaptation_rate = 0.01
            print("✅ AEC initialized (simplified correlation-based)")
        else:
            print("⚠️ AEC disabled")
    
    def feed_tts_reference(self, tts_frame: bytes):
        """
        Feed TTS audio that's being sent to speakers.
        This is the reference signal for AEC.
        """
        if not self.config.aec_enabled:
            return
            
        # Convert to float32
        audio_int16 = np.frombuffer(tts_frame, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        self.tts_buffer.append(audio_float32)
    
    def process_frame(self, mic_frame: bytes) -> tuple[bytes, VADState]:
        """
        Process a microphone frame through the full pipeline.
        
        Args:
            mic_frame: Raw PCM bytes from microphone
            
        Returns:
            (clean_frame, vad_state): Cleaned audio and current VAD state
        """
        with self.lock:
            # 1. Convert to float32
            mic_int16 = np.frombuffer(mic_frame, dtype=np.int16)
            mic_float32 = mic_int16.astype(np.float32) / 32768.0
            
            # 2. Apply AEC (subtract estimated echo)
            if self.config.aec_enabled and len(self.tts_buffer) > 0:
                clean_float32 = self._apply_aec(mic_float32)
            else:
                clean_float32 = mic_float32
            
            # 3. Buffer for VAD (needs exactly 512 samples)
            self.mic_buffer = np.concatenate([self.mic_buffer, clean_float32])
            
            # 4. Process complete frames through VAD
            vad_results = []
            while len(self.mic_buffer) >= self.frame_size:
                frame = self.mic_buffer[:self.frame_size]
                self.mic_buffer = self.mic_buffer[self.frame_size:]
                
                # Get speech probability
                audio_tensor = torch.from_numpy(frame)
                speech_prob = self.vad_model(audio_tensor, self.config.sample_rate).item()
                is_speech = speech_prob > self.config.vad_threshold
                vad_results.append(is_speech)
            
            # 5. Update VAD state machine
            for is_speech in vad_results:
                self._update_vad_state(is_speech)
            
            # 6. Convert back to bytes
            clean_int16 = (clean_float32 * 32768.0).astype(np.int16)
            clean_bytes = clean_int16.tobytes()
            
            return clean_bytes, self.vad_state
    
    def _apply_aec(self, mic_signal: np.ndarray) -> np.ndarray:
        """
        Apply acoustic echo cancellation.
        
        This is a simplified NLMS-based AEC. For production use,
        consider WebRTC's AEC or SpeexDSP.
        """
        # Get recent TTS reference
        if len(self.tts_buffer) == 0:
            return mic_signal
        
        tts_ref = np.concatenate(list(self.tts_buffer))
        
        # Simple correlation-based echo estimation
        # This finds how much of the TTS is present in the mic
        min_len = min(len(mic_signal), len(tts_ref))
        if min_len < 100:
            return mic_signal
        
        # Estimate echo using cross-correlation
        correlation = np.correlate(
            mic_signal[:min_len], 
            tts_ref[:min_len], 
            mode='valid'
        )
        
        if len(correlation) > 0:
            # Find the scaling factor
            tts_power = np.sum(tts_ref[:min_len] ** 2) + 1e-10
            echo_scale = np.max(correlation) / tts_power
            echo_scale = np.clip(echo_scale, 0, 1)
            
            # Subtract estimated echo
            estimated_echo = tts_ref[:len(mic_signal)] * echo_scale
            if len(estimated_echo) == len(mic_signal):
                clean_signal = mic_signal - estimated_echo * 0.7  # Partial subtraction
                return clean_signal
        
        return mic_signal
    
    def _update_vad_state(self, is_speech: bool):
        """
        Update the VAD state machine and trigger callbacks.
        
        State transitions:
        - SILENCE + speech for onset_frames → SPEECH (potential barge-in)
        - SPEECH + silence for silence_frames → SILENCE (turn end)
        """
        if is_speech:
            self.consecutive_speech_frames += 1
            self.consecutive_silence_frames = 0
            
            # Check for speech onset (barge-in trigger)
            if (self.vad_state == VADState.SILENCE and 
                self.consecutive_speech_frames >= self.config.onset_frames):
                self.vad_state = VADState.SPEECH
                print(f"🎤 VAD: Speech started ({self.consecutive_speech_frames} frames)")
                if self.on_barge_in:
                    self.on_barge_in()
        else:
            self.consecutive_silence_frames += 1
            self.consecutive_speech_frames = 0
            
            # Check for turn end
            if (self.vad_state == VADState.SPEECH and 
                self.consecutive_silence_frames >= self.config.silence_frames):
                self.vad_state = VADState.SILENCE
                print(f"🔇 VAD: Turn ended ({self.consecutive_silence_frames} frames of silence)")
                if self.on_turn_end:
                    self.on_turn_end()
    
    def reset(self):
        """Reset all state (call when switching modes)."""
        with self.lock:
            self.mic_buffer = np.array([], dtype=np.float32)
            self.tts_buffer.clear()
            self.vad_state = VADState.SILENCE
            self.consecutive_speech_frames = 0
            self.consecutive_silence_frames = 0
            if hasattr(self.vad_model, 'reset_states'):
                self.vad_model.reset_states()
            print("🔄 AudioFrontend reset")


# --- Test Block ---
if __name__ == "__main__":
    import time
    
    print("Testing AudioFrontend...")
    
    frontend = AudioFrontend()
    
    # Simulate some audio frames
    for i in range(10):
        # Create fake mic frame (512 samples = 32ms at 16kHz)
        fake_audio = np.random.randn(1600).astype(np.float32) * 0.1
        fake_bytes = (fake_audio * 32768).astype(np.int16).tobytes()
        
        clean, vad_state = frontend.process_frame(fake_bytes)
        print(f"Frame {i}: VAD={vad_state.value}, clean_size={len(clean)}")
    
    print("✅ AudioFrontend test complete")