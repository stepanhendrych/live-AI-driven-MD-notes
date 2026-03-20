import queue
import subprocess
import threading
import time

import sounddevice as sd

from .audio_capture import AudioCaptureWorker
from .config import Settings
from .notes_processor import NotesProcessorWorker
from .transcriber import TranscriberWorker


class LiveNotesAssistant:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.stop_event = threading.Event()
        self.file_lock = threading.Lock()
        self.audio_queue = queue.Queue(maxsize=self.settings.queues.audio_maxsize)
        self.transcript_queue = queue.Queue(maxsize=self.settings.queues.transcript_maxsize)

        self.capture_worker = AudioCaptureWorker(
            audio_queue=self.audio_queue,
            stop_event=self.stop_event,
            chunk=self.settings.audio.chunk,
            channels=self.settings.audio.channels,
            rate=self.settings.audio.rate,
            device=self.settings.audio.device,
            silence_threshold=self.settings.audio.silence_threshold,
            silence_duration=self.settings.audio.silence_duration,
            max_chunk_seconds=self.settings.audio.max_chunk_seconds,
        )
        self.transcriber_worker = TranscriberWorker(
            audio_queue=self.audio_queue,
            transcript_queue=self.transcript_queue,
            stop_event=self.stop_event,
            model_size=self.settings.app.model_size,
            language=self.settings.transcriber.language,
            channels=self.settings.audio.channels,
            rate=self.settings.audio.rate,
            sample_width=self.settings.transcriber.sample_width,
            compute_device=self.settings.transcriber.compute_device,
            fp16_mode=self.settings.transcriber.fp16_mode,
            temperature=self.settings.transcriber.temperature,
            beam_size=self.settings.transcriber.beam_size,
            best_of=self.settings.transcriber.best_of,
            condition_on_previous_text=self.settings.transcriber.condition_on_previous_text,
            no_speech_threshold=self.settings.transcriber.no_speech_threshold,
            logprob_threshold=self.settings.transcriber.logprob_threshold,
        )
        self.processor_worker = NotesProcessorWorker(
            transcript_queue=self.transcript_queue,
            stop_event=self.stop_event,
            notes_file=self.settings.app.notes_file,
            file_lock=self.file_lock,
            ollama_model=self.settings.app.ollama_model,
            context_lines=self.settings.processor.context_lines,
            heading_lines=self.settings.processor.heading_lines,
            ollama_timeout=self.settings.processor.ollama_timeout,
            ollama_retries=self.settings.processor.ollama_retries,
            retry_backoff_seconds=self.settings.processor.retry_backoff_seconds,
            feedback_preview_chars=self.settings.processor.feedback_preview_chars,
            max_context_chars=self.settings.processor.max_context_chars,
            min_bullet_points=self.settings.processor.min_bullet_points,
            max_section_lines=self.settings.processor.max_section_lines,
            prompt_mode=self.settings.processor.prompt_mode,
            two_phase_extraction=self.settings.processor.two_phase_extraction,
            max_required_facts=self.settings.processor.max_required_facts,
            enforce_fact_coverage=self.settings.processor.enforce_fact_coverage,
            preserve_note_style=self.settings.processor.preserve_note_style,
            adaptive_speed_mode=self.settings.processor.adaptive_speed_mode,
            short_transcript_chars=self.settings.processor.short_transcript_chars,
            coverage_repair_min_missing=self.settings.processor.coverage_repair_min_missing,
            coverage_repair_min_missing_ratio=self.settings.processor.coverage_repair_min_missing_ratio,
            quality_repair_min_chars=self.settings.processor.quality_repair_min_chars,
            enable_vector_memory=self.settings.memory.enable_vector_memory,
            vector_memory_persist_dir=self.settings.memory.persist_dir,
            vector_memory_similarity_threshold=self.settings.memory.similarity_threshold,
            vector_memory_context_results=self.settings.memory.context_results,
            enable_wiki_links=self.settings.memory.enable_wiki_links,
        )

    def _check_ollama_available(self):
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except Exception as exc:
            print(f"[app] ollama is not available: {exc}")
            return False

        if result.returncode != 0:
            err = (result.stderr or "").strip()
            print(f"[app] ollama check failed: {err[:200]}")
            return False
        return True

    def _workers(self):
        return [self.capture_worker, self.transcriber_worker, self.processor_worker]

    def _check_audio_input_device(self):
        try:
            info = sd.query_devices(self.settings.audio.device, kind="input")
            return True, info
        except Exception as exc:
            return False, str(exc)

    def _startup_self_check(self):
        print("[self-check] startup diagnostics")

        ok = True

        ollama_ok = self._check_ollama_available()
        if ollama_ok:
            print("[self-check] ollama: ok")
        else:
            print("[self-check] ollama: warning (note generation may fail)")

        try:
            model_name = self.settings.app.model_size
            compute_type = self.transcriber_worker.compute_type
            device = self.transcriber_worker.compute_device
            print(f"[self-check] faster-whisper: ok (model={model_name}, device={device}, compute_type={compute_type})")
        except Exception as exc:
            ok = False
            print(f"[self-check] whisper: failed ({exc})")

        audio_ok, audio_info = self._check_audio_input_device()
        if audio_ok:
            device_name = audio_info.get("name", "unknown")
            samplerate = int(audio_info.get("default_samplerate", self.settings.audio.rate))
            print(f"[self-check] audio input: ok ({device_name}, default_sr={samplerate})")
        else:
            ok = False
            print(f"[self-check] audio input: failed ({audio_info})")
            print("[self-check] tip: run `python -m sounddevice` and set `audio.device` in config.toml")

        return ok

    def stop(self):
        if self.stop_event.is_set():
            return

        self.stop_event.set()

        for q in (self.audio_queue, self.transcript_queue):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass

        for worker in self._workers():
            worker.join(timeout=5)

    def run(self):
        if not self._startup_self_check():
            print("[app] self-check failed, aborting startup")
            return

        for worker in self._workers():
            worker.start()

        print("[app] live notes assistant is running")

        try:
            while any(worker.is_alive() for worker in self._workers()):
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[app] shutdown requested")
        finally:
            self.stop()
            print("[app] stopped")
