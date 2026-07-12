class FusionEngine():
    """
    Combines transcript, screen workspace context, camera context, and memory
    into a single grounded prompt for the LLM.
    """

    def combine(
        self,
        transcript: str,
        workspace_state: str | None = None,
        visual_context: str | None = None,
        memory: str | None = None,
    ) -> str:
        parts = []

        if memory:
            parts.append(f"[Conversation Memory]:\n{memory}")

        if workspace_state:
            parts.append(f"[DISPLAY — computer screen content, text description]:\n{workspace_state}")

        if visual_context:
            parts.append(f"[CAMERA — physical room, see attached image]: {visual_context}")

        parts.append(f"[User Said]: {transcript}")

        return "\n\n".join(parts)


# --- Test Block ---
if __name__ == "__main__":
    engine = FusionEngine()
    
    result = engine.combine(
        transcript="What's on my desk?",
        visual_context="A laptop, coffee mug, and notebook on a wooden desk.",
        memory=None
    )
    
    print("=== Fused Prompt ===")
    print(result)