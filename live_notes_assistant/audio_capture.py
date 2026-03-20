import queue
import threading
import time

import numpy as np
import sounddevice as sd


class AudioCaptureWorker(threading.Thread):
    """Captures microphone audio and emits speech chunks to a queue."""

    def __init__(
        self,
        audio_queue,
        stop_event,
        chunk=1024,
        channels=1,
        rate=16000,
        device=None,
        silence_threshold=500,
        silence_duration=2.0,
        max_chunk_seconds=10.0,
    ):
        super().__init__(daemon=True)
        self.audio_queue = audio_queue
        self.stop_event = stop_event
        self.chunk = chunk
        self.channels = channels
        self.rate = rate
        self.device = device
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.max_chunk_seconds = max_chunk_seconds
        self.audio_dtype = "int16"

    def _is_silent(self, data_chunk):
        audio_data = np.frombuffer(data_chunk, dtype=np.int16)
        return np.abs(audio_data).mean() < self.silence_threshold

    def _flush_frames(self, frames):
        if not frames:
            return []

        audio_data = b"".join(frames)
        try:
            self.audio_queue.put(audio_data, timeout=0.5)
        except queue.Full:
            print("[capture] audio queue full, dropping chunk")
        return []

    def run(self):
        stream = None
        frames = []
        silence_start = None
        chunk_start = time.time()

        try:
            stream = sd.RawInputStream(
                channels=self.channels,
                samplerate=self.rate,
                dtype=self.audio_dtype,
                blocksize=self.chunk,
                device=self.device,
            )
            stream.start()
            if self.device is None:
                print("[capture] recording started on default input (Ctrl+C to stop)")
            else:
                print(f"[capture] recording started on device={self.device} (Ctrl+C to stop)")

            while not self.stop_event.is_set():
                try:
                    data, overflowed = stream.read(self.chunk)
                except Exception as exc:
                    print(f"[capture] read error: {exc}")
                    time.sleep(0.05)
                    continue

                if overflowed:
                    print("[capture] input overflow detected")

                frames.append(bytes(data))
                now = time.time()

                if self._is_silent(frames[-1]):
                    if silence_start is None:
                        silence_start = now
                    elif now - silence_start >= self.silence_duration:
                        frames = self._flush_frames(frames)
                        silence_start = None
                        chunk_start = now
                else:
                    silence_start = None

                if now - chunk_start >= self.max_chunk_seconds:
                    frames = self._flush_frames(frames)
                    chunk_start = now

        finally:
            frames = self._flush_frames(frames)

            if stream is not None:
                stream.stop()
                stream.close()

            try:
                self.audio_queue.put(None, timeout=0.5)
            except queue.Full:
                pass
            print("[capture] stopped")
