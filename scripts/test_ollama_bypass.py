import argparse
import queue
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live_notes_assistant.config import load_settings
from live_notes_assistant.notes_processor import NotesProcessorWorker


def main():
    parser = argparse.ArgumentParser(
        description="Bypass audio/transcriber and test that Ollama processing appends into notes file"
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "config.toml"), help="Path to config.toml")

    transcript_group = parser.add_mutually_exclusive_group(required=True)
    transcript_group.add_argument(
        "--transcript",
        help="Manual transcript text sent directly to Ollama worker",
    )
    transcript_group.add_argument(
        "--transcript-file",
        default="",
        help="Path to txt/md file whose content will be used as transcript",
    )

    parser.add_argument("--notes-file", default="", help="Optional override for output notes file")
    parser.add_argument("--timeout", type=float, default=120.0, help="Max wait for worker finish in seconds")
    parser.add_argument(
        "--use-context",
        action="store_true",
        help="Use context from existing notes file (default bypass test uses zero context)",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    notes_file = Path(args.notes_file.strip() or settings.app.notes_file)

    if args.transcript_file:
        transcript_path = Path(args.transcript_file)
        transcript = transcript_path.read_text(encoding="utf-8")
        transcript_source = f"file:{transcript_path}"
    else:
        transcript = args.transcript
        transcript_source = "--transcript"

    context_lines = settings.processor.context_lines if args.use_context else 0
    heading_lines = settings.processor.heading_lines if args.use_context else 0

    transcript_queue = queue.Queue(maxsize=settings.queues.transcript_maxsize)
    stop_event = threading.Event()
    file_lock = threading.Lock()

    worker = NotesProcessorWorker(
        transcript_queue=transcript_queue,
        stop_event=stop_event,
        notes_file=notes_file,
        file_lock=file_lock,
        ollama_model=settings.app.ollama_model,
        context_lines=context_lines,
        heading_lines=heading_lines,
        ollama_timeout=settings.processor.ollama_timeout,
        ollama_retries=settings.processor.ollama_retries,
        retry_backoff_seconds=settings.processor.retry_backoff_seconds,
        feedback_preview_chars=settings.processor.feedback_preview_chars,
        max_context_chars=settings.processor.max_context_chars,
        min_bullet_points=settings.processor.min_bullet_points,
        max_section_lines=settings.processor.max_section_lines,
        prompt_mode=settings.processor.prompt_mode,
        two_phase_extraction=settings.processor.two_phase_extraction,
    )

    before_exists = notes_file.exists()
    before_size = notes_file.stat().st_size if before_exists else 0

    print(f"[bypass-test] using notes file: {notes_file}")
    print(f"[bypass-test] transcript source: {transcript_source}")
    print(f"[bypass-test] sending transcript chars={len(transcript)}")
    print(f"[bypass-test] context mode: {'enabled' if args.use_context else 'disabled (pure bypass)'}")

    worker.start()
    transcript_queue.put(transcript)
    transcript_queue.put(None)

    started = time.monotonic()
    worker.join(timeout=args.timeout)

    if worker.is_alive():
        stop_event.set()
        worker.join(timeout=3)
        raise TimeoutError("Ollama bypass test timed out")

    elapsed = time.monotonic() - started
    after_exists = notes_file.exists()
    after_size = notes_file.stat().st_size if after_exists else 0

    print(f"[bypass-test] finished in {elapsed:.2f}s")

    if after_size > before_size:
        print(f"[bypass-test] SUCCESS: file grew by {after_size - before_size} bytes")
        tail = notes_file.read_text(encoding="utf-8", errors="ignore")[-400:]
        print("[bypass-test] tail preview:")
        print(tail)
    else:
        print("[bypass-test] NO CHANGE: notes file was not modified")
        print("[bypass-test] check logs above for timeout, duplicate filter, or ollama errors")


if __name__ == "__main__":
    main()
