"""Entry point for running VISION as a module: python -m vision"""

from vision.core import Orchestrator


def main():
    """Main entry point for the VISION application"""
    print("--- Starting VISION Voice Loop ---")
    try:
        orchestrator = Orchestrator()
        orchestrator.start()
    except KeyboardInterrupt:
        print("\nOrchestrator stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
