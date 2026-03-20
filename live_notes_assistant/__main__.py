import os

from .config import load_settings
from .coordinator import LiveNotesAssistant


def main():
    config_path = os.getenv("LIVE_NOTES_CONFIG", "config.toml")
    settings = load_settings(config_path)
    assistant = LiveNotesAssistant(settings=settings)
    assistant.run()


if __name__ == "__main__":
    main()
