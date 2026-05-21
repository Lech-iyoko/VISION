class FusionEngine():
    """
    Resonsible for integrating visual context into the conversation flow.
    Combines trscript, visual context, and memory into a single prompt for the LLM.
    """

    def combine(self, transcript, visual_context=None, memory=None):
        parts = []

        if memory:
            parts.append(f"[Conversation Memory]:\n{memory}")

        if visual_context:
            parts.append(f"[Visual Context]: {visual_context}")

        parts.append(f"[User Said]:\n{transcript}")

        parts.append("\n[Instruction]: Respond helpfully. If relevant, reference what you see.")

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