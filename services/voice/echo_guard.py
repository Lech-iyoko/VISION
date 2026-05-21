"""
Text-level echo detection.
Safety net to catch any residual echo that slips through AEC.
"""

from difflib import SequenceMatcher


def is_echo(asr_text: str, recent_tts_text: str, threshold: float = 0.6) -> bool:
    """
    Check if ASR output is likely an echo of recent TTS.
    
    Uses n-gram similarity to detect if the user's speech
    matches what the assistant just said.
    
    Args:
        asr_text: What ASR heard
        recent_tts_text: What TTS just said
        threshold: Similarity threshold (0.6 = 60% match)
    
    Returns:
        True if this is likely an echo, False if genuine user speech
    """
    if not asr_text or not recent_tts_text:
        return False
    
    # Normalize
    asr_lower = asr_text.lower().strip()
    tts_lower = recent_tts_text.lower().strip()
    
    # Calculate similarity
    similarity = SequenceMatcher(None, asr_lower, tts_lower).ratio()
    
    is_echo_detected = similarity >= threshold
    
    if is_echo_detected:
        print(f"🔇 Echo detected: {similarity:.0%} match")
        print(f"   ASR: '{asr_text[:50]}...'")
        print(f"   TTS: '{recent_tts_text[:50]}...'")
    
    return is_echo_detected


# --- Test ---
if __name__ == "__main__":
    # Test cases
    tts = "Hello, I'm Vision, your personal assistant."
    
    print(is_echo("Hello, I'm Vision, your personal assistant.", tts))  # True
    print(is_echo("Hello Vision", tts))  # Maybe
    print(is_echo("What's the weather today?", tts))  # False