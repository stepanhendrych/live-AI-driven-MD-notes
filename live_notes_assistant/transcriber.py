import queue
import threading
import time

import numpy as np
from faster_whisper import WhisperModel


class TranscriberWorker(threading.Thread):
    """Consumes audio chunks and emits text transcripts (backed by faster-whisper)."""

    def __init__(
        self,
        audio_queue,
        transcript_queue,
        stop_event,
        model_size="base",
        language="cs",
        channels=1,
        rate=16000,
        sample_width=2,
        compute_device="auto",
        fp16_mode="auto",
        temperature=0.0,
        beam_size=5,
        best_of=5,
        condition_on_previous_text=True,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
    ):
        super().__init__(daemon=True)
        self.audio_queue = audio_queue
        self.transcript_queue = transcript_queue
        self.stop_event = stop_event
        self.language = language
        self.channels = channels
        self.rate = rate
        self.sample_width = sample_width
        self.compute_device = self._resolve_compute_device(compute_device)
        self.compute_type = self._resolve_compute_type(fp16_mode)
        # fp16_enabled kept for backward-compatible self-check reporting
        self.fp16_enabled = self.compute_type in {"float16", "int8_float16"}
        self.temperature = temperature
        self.beam_size = max(1, int(beam_size))
        self.best_of = max(1, int(best_of))
        self.condition_on_previous_text = bool(condition_on_previous_text)
        self.no_speech_threshold = no_speech_threshold
        self.logprob_threshold = logprob_threshold
        self.whisper_model = WhisperModel(
            model_size,
            device=self.compute_device,
            compute_type=self.compute_type,
        )

    @staticmethod
    def _has_cuda():
        # Prefer ctranslate2 (faster-whisper dependency) for CUDA detection;
        # fall back to torch if available.
        try:
            import ctranslate2
            return len(ctranslate2.get_supported_compute_types("cuda")) > 0
        except Exception:
            pass
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _resolve_compute_device(self, requested_device):
        requested = str(requested_device).strip().lower()
        if requested in {"", "auto", "default"}:
            return "cuda" if self._has_cuda() else "cpu"

        if requested == "cuda" and not self._has_cuda():
            print("[transcriber] cuda requested but unavailable, falling back to cpu")
            return "cpu"

        if requested in {"cpu", "cuda"}:
            return requested

        print(f"[transcriber] unknown compute_device={requested_device}, falling back to auto")
        return "cuda" if self._has_cuda() else "cpu"

    def _resolve_compute_type(self, fp16_mode):
        """Map fp16_mode config to a faster-whisper compute_type string."""
        if isinstance(fp16_mode, bool):
            use_fp16 = fp16_mode
        else:
            mode = str(fp16_mode).strip().lower()
            # "auto" / "true" / "1" etc. -> use fp16 where sensible
            use_fp16 = mode not in {"0", "false", "no", "off"}

        if self.compute_device == "cuda":
            return "float16" if use_fp16 else "float32"
        # CPU: int8 gives the best speed/quality trade-off
        return "int8" if use_fp16 else "float32"

    def _bytes_to_mono_float32(self, audio_data):
        if self.sample_width != 2:
            raise ValueError(f"Unsupported sample_width={self.sample_width}; expected 2 bytes (int16)")

        samples = np.frombuffer(audio_data, dtype=np.int16)
        if samples.size == 0:
            return np.array([], dtype=np.float32)

        if self.channels > 1:
            remainder = samples.size % self.channels
            if remainder:
                samples = samples[: samples.size - remainder]
            if samples.size == 0:
                return np.array([], dtype=np.float32)
            samples = samples.reshape(-1, self.channels).mean(axis=1)

        audio = samples.astype(np.float32) / 32768.0
        return np.clip(audio, -1.0, 1.0)

    def run(self):
        print(
            f"[transcriber] faster-whisper ready (device={self.compute_device}, "
            f"compute_type={self.compute_type}, beam_size={self.beam_size}, "
            f"temperature={self.temperature})"
        )

        while not self.stop_event.is_set():
            try:
                audio_data = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if audio_data is None:
                break

            try:
                start = time.monotonic()
                audio_input = self._bytes_to_mono_float32(audio_data)
                if audio_input.size == 0:
                    continue

                transcribe_kwargs = dict(
                    language=self.language,
                    task="transcribe",
                    temperature=self.temperature,
                    beam_size=self.beam_size,
                    condition_on_previous_text=self.condition_on_previous_text,
                    no_speech_threshold=self.no_speech_threshold,
                    log_prob_threshold=self.logprob_threshold,
                )
                # best_of is only meaningful in sampling mode (temperature > 0)
                if self.temperature > 0:
                    transcribe_kwargs["best_of"] = self.best_of

                segments, _info = self.whisper_model.transcribe(audio_input, **transcribe_kwargs)
                # segments is a lazy generator; consume it to get the full text
                transcript = "".join(seg.text for seg in segments).strip()
            except Exception as exc:
                print(f"[transcriber] transcription failed: {exc}")
                continue

            if transcript:
                elapsed = time.monotonic() - start
                audio_seconds = len(audio_input) / float(self.rate)
                print(
                    f"[transcriber] ok ({audio_seconds:.2f}s audio -> {elapsed:.2f}s), text={transcript[:140]}"
                )
                try:
                    self.transcript_queue.put(transcript, timeout=0.5)
                except queue.Full:
                    print("[transcriber] transcript queue full, dropping transcript")

        try:
            self.transcript_queue.put(None, timeout=0.5)
        except queue.Full:
            pass
        print("[transcriber] stopped")
