from dataclasses import dataclass
from pathlib import Path

import tomllib


@dataclass(frozen=True)
class AppSection:
    notes_file: str = "dejepis.md"
    model_size: str = "base"
    ollama_model: str = "llama3.2:3b"


@dataclass(frozen=True)
class AudioSection:
    chunk: int = 1024
    channels: int = 1
    rate: int = 16000
    device: str | int | None = None
    silence_threshold: int = 500
    silence_duration: float = 2.0
    max_chunk_seconds: float = 10.0


@dataclass(frozen=True)
class TranscriberSection:
    language: str = "cs"
    sample_width: int = 2
    compute_device: str = "auto"
    fp16_mode: str | bool = "auto"
    temperature: float = 0.0
    beam_size: int = 5
    best_of: int = 5
    condition_on_previous_text: bool = True
    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0


@dataclass(frozen=True)
class ProcessorSection:
    context_lines: int = 50
    heading_lines: int = 10
    ollama_timeout: int = 30
    ollama_retries: int = 2
    retry_backoff_seconds: float = 2.0
    feedback_preview_chars: int = 180
    max_context_chars: int = 6000
    min_bullet_points: int = 6
    max_section_lines: int = 60
    prompt_mode: str = "balanced"
    two_phase_extraction: bool = True
    max_required_facts: int = 12
    enforce_fact_coverage: bool = True
    preserve_note_style: bool = True
    adaptive_speed_mode: bool = True
    short_transcript_chars: int = 220
    coverage_repair_min_missing: int = 3
    coverage_repair_min_missing_ratio: float = 0.35
    quality_repair_min_chars: int = 140


@dataclass(frozen=True)
class QueueSection:
    audio_maxsize: int = 32
    transcript_maxsize: int = 64


@dataclass(frozen=True)
class MemorySection:
    enable_vector_memory: bool = True
    persist_dir: str = ".notes_vector_db"
    similarity_threshold: float = 0.05
    context_results: int = 3
    enable_wiki_links: bool = True


@dataclass(frozen=True)
class Settings:
    app: AppSection = AppSection()
    audio: AudioSection = AudioSection()
    transcriber: TranscriberSection = TranscriberSection()
    processor: ProcessorSection = ProcessorSection()
    queues: QueueSection = QueueSection()
    memory: MemorySection = MemorySection()


def _section(data, name):
    section = data.get(name, {})
    if isinstance(section, dict):
        return section
    return {}


def _parse_audio_device(value):
    if value is None:
        return None

    if isinstance(value, int):
        return value

    text = str(value).strip()
    if not text or text.lower() in {"default", "none"}:
        return None

    if text.isdigit():
        return int(text)

    return text


def _parse_bool_or_auto(value, default):
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in {"auto", "default"}:
        return "auto"
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False

    return default


def _parse_bool(value, default):
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False

    return default


def load_settings(config_path):
    path = Path(config_path)
    if not path.exists():
        print(f"[config] file not found, using defaults: {path}")
        return Settings()

    with path.open("rb") as handle:
        data = tomllib.load(handle)

    app_data = _section(data, "app")
    audio_data = _section(data, "audio")
    transcriber_data = _section(data, "transcriber")
    processor_data = _section(data, "processor")
    queues_data = _section(data, "queues")
    memory_data = _section(data, "memory")

    return Settings(
        app=AppSection(
            notes_file=str(app_data.get("notes_file", AppSection.notes_file)),
            model_size=str(app_data.get("model_size", AppSection.model_size)),
            ollama_model=str(app_data.get("ollama_model", AppSection.ollama_model)),
        ),
        audio=AudioSection(
            chunk=int(audio_data.get("chunk", AudioSection.chunk)),
            channels=int(audio_data.get("channels", AudioSection.channels)),
            rate=int(audio_data.get("rate", AudioSection.rate)),
            device=_parse_audio_device(audio_data.get("device", AudioSection.device)),
            silence_threshold=int(audio_data.get("silence_threshold", AudioSection.silence_threshold)),
            silence_duration=float(audio_data.get("silence_duration", AudioSection.silence_duration)),
            max_chunk_seconds=float(audio_data.get("max_chunk_seconds", AudioSection.max_chunk_seconds)),
        ),
        transcriber=TranscriberSection(
            language=str(transcriber_data.get("language", TranscriberSection.language)),
            sample_width=int(transcriber_data.get("sample_width", TranscriberSection.sample_width)),
            compute_device=str(transcriber_data.get("compute_device", TranscriberSection.compute_device)),
            fp16_mode=_parse_bool_or_auto(transcriber_data.get("fp16_mode", TranscriberSection.fp16_mode), TranscriberSection.fp16_mode),
            temperature=float(transcriber_data.get("temperature", TranscriberSection.temperature)),
            beam_size=int(transcriber_data.get("beam_size", TranscriberSection.beam_size)),
            best_of=int(transcriber_data.get("best_of", TranscriberSection.best_of)),
            condition_on_previous_text=_parse_bool(transcriber_data.get("condition_on_previous_text", TranscriberSection.condition_on_previous_text), TranscriberSection.condition_on_previous_text),
            no_speech_threshold=float(transcriber_data.get("no_speech_threshold", TranscriberSection.no_speech_threshold)),
            logprob_threshold=float(transcriber_data.get("logprob_threshold", TranscriberSection.logprob_threshold)),
        ),
        processor=ProcessorSection(
            context_lines=int(processor_data.get("context_lines", ProcessorSection.context_lines)),
            heading_lines=int(processor_data.get("heading_lines", ProcessorSection.heading_lines)),
            ollama_timeout=int(processor_data.get("ollama_timeout", ProcessorSection.ollama_timeout)),
            ollama_retries=int(processor_data.get("ollama_retries", ProcessorSection.ollama_retries)),
            retry_backoff_seconds=float(processor_data.get("retry_backoff_seconds", ProcessorSection.retry_backoff_seconds)),
            feedback_preview_chars=int(processor_data.get("feedback_preview_chars", ProcessorSection.feedback_preview_chars)),
            max_context_chars=int(processor_data.get("max_context_chars", ProcessorSection.max_context_chars)),
            min_bullet_points=int(processor_data.get("min_bullet_points", ProcessorSection.min_bullet_points)),
            max_section_lines=int(processor_data.get("max_section_lines", ProcessorSection.max_section_lines)),
            prompt_mode=str(processor_data.get("prompt_mode", ProcessorSection.prompt_mode)),
            two_phase_extraction=_parse_bool(
                processor_data.get("two_phase_extraction", ProcessorSection.two_phase_extraction),
                ProcessorSection.two_phase_extraction,
            ),
            max_required_facts=int(processor_data.get("max_required_facts", ProcessorSection.max_required_facts)),
            enforce_fact_coverage=_parse_bool(
                processor_data.get("enforce_fact_coverage", ProcessorSection.enforce_fact_coverage),
                ProcessorSection.enforce_fact_coverage,
            ),
            preserve_note_style=_parse_bool(
                processor_data.get("preserve_note_style", ProcessorSection.preserve_note_style),
                ProcessorSection.preserve_note_style,
            ),
            adaptive_speed_mode=_parse_bool(
                processor_data.get("adaptive_speed_mode", ProcessorSection.adaptive_speed_mode),
                ProcessorSection.adaptive_speed_mode,
            ),
            short_transcript_chars=int(
                processor_data.get("short_transcript_chars", ProcessorSection.short_transcript_chars)
            ),
            coverage_repair_min_missing=int(
                processor_data.get("coverage_repair_min_missing", ProcessorSection.coverage_repair_min_missing)
            ),
            coverage_repair_min_missing_ratio=float(
                processor_data.get(
                    "coverage_repair_min_missing_ratio",
                    ProcessorSection.coverage_repair_min_missing_ratio,
                )
            ),
            quality_repair_min_chars=int(
                processor_data.get("quality_repair_min_chars", ProcessorSection.quality_repair_min_chars)
            ),
        ),
        queues=QueueSection(
            audio_maxsize=int(queues_data.get("audio_maxsize", QueueSection.audio_maxsize)),
            transcript_maxsize=int(queues_data.get("transcript_maxsize", QueueSection.transcript_maxsize)),
        ),
        memory=MemorySection(
            enable_vector_memory=_parse_bool(
                memory_data.get("enable_vector_memory", MemorySection.enable_vector_memory),
                MemorySection.enable_vector_memory,
            ),
            persist_dir=str(memory_data.get("persist_dir", MemorySection.persist_dir)),
            similarity_threshold=float(
                memory_data.get("similarity_threshold", MemorySection.similarity_threshold)
            ),
            context_results=int(memory_data.get("context_results", MemorySection.context_results)),
            enable_wiki_links=_parse_bool(
                memory_data.get("enable_wiki_links", MemorySection.enable_wiki_links),
                MemorySection.enable_wiki_links,
            ),
        ),
    )
