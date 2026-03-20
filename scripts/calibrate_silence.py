import argparse
import re
import statistics
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

REPO_ROOT = Path(__file__).resolve().parents[1]


def sample_level(seconds, rate, channels, device, chunk):
    levels = []
    sample_count = int(seconds * rate)
    remaining = sample_count

    with sd.RawInputStream(
        samplerate=rate,
        channels=channels,
        dtype="int16",
        blocksize=chunk,
        device=device,
    ) as stream:
        while remaining > 0:
            frames_to_read = min(chunk, remaining)
            data, overflowed = stream.read(frames_to_read)
            if overflowed:
                print("[calibrate] input overflow detected")
            remaining -= frames_to_read

            audio_data = np.frombuffer(bytes(data), dtype=np.int16)
            if audio_data.size == 0:
                continue

            levels.append(float(np.abs(audio_data).mean()))

    return levels


def suggest_threshold(ambient_levels, speech_levels):
    ambient_p95 = float(np.percentile(ambient_levels, 95)) if ambient_levels else 0.0
    speech_p20 = float(np.percentile(speech_levels, 20)) if speech_levels else 0.0

    if speech_p20 <= ambient_p95:
        # Conservative fallback if separation is weak.
        return int(max(ambient_p95 * 2.0, ambient_p95 + 120.0))

    midpoint = (ambient_p95 + speech_p20) / 2.0
    return int(midpoint)


def replace_audio_key(config_text, key, value):
    pattern = re.compile(r"(^\[audio\]\n)(.*?)(^\[|\Z)", re.MULTILINE | re.DOTALL)
    match = pattern.search(config_text)
    if not match:
        raise ValueError("[audio] section not found in config.toml")

    audio_block = match.group(2)
    key_pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*.*$", re.MULTILINE)
    replacement_line = f"{key} = {value}"

    if key_pattern.search(audio_block):
        updated_audio_block = key_pattern.sub(replacement_line, audio_block)
    else:
        updated_audio_block = audio_block.rstrip() + "\n" + replacement_line + "\n"

    return config_text[: match.start(2)] + updated_audio_block + config_text[match.end(2) :]


def apply_to_config(config_path, threshold, silence_duration=None):
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")

    updated = replace_audio_key(text, "silence_threshold", str(int(threshold)))
    if silence_duration is not None:
        updated = replace_audio_key(updated, "silence_duration", str(float(silence_duration)))

    path.write_text(updated, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Calibrate silence threshold for microphone input")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.toml"), help="Path to config.toml")
    parser.add_argument("--device", default="", help="Input device index or name; empty means default")
    parser.add_argument("--rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--ambient-seconds", type=float, default=5.0)
    parser.add_argument("--speech-seconds", type=float, default=5.0)
    parser.add_argument("--silence-duration", type=float, default=None, help="Optional new silence_duration value")
    parser.add_argument("--apply", action="store_true", help="Write suggested values into config.toml")
    args = parser.parse_args()

    device = None
    if str(args.device).strip() not in {"", "default", "none"}:
        device = int(args.device) if str(args.device).isdigit() else args.device

    print("[calibrate] Step 1/2: stay silent...")
    time.sleep(0.5)
    ambient_levels = sample_level(args.ambient_seconds, args.rate, args.channels, device, args.chunk)

    print("[calibrate] Step 2/2: speak naturally...")
    time.sleep(0.5)
    speech_levels = sample_level(args.speech_seconds, args.rate, args.channels, device, args.chunk)

    if not ambient_levels or not speech_levels:
        raise RuntimeError("Not enough audio samples captured for calibration")

    threshold = suggest_threshold(ambient_levels, speech_levels)

    print("[calibrate] Results")
    print(f"- ambient mean: {statistics.mean(ambient_levels):.1f}")
    print(f"- ambient p95: {np.percentile(ambient_levels, 95):.1f}")
    print(f"- speech mean: {statistics.mean(speech_levels):.1f}")
    print(f"- speech p20: {np.percentile(speech_levels, 20):.1f}")
    print(f"- suggested silence_threshold: {threshold}")

    if args.silence_duration is not None:
        print(f"- suggested silence_duration: {args.silence_duration}")

    if args.apply:
        apply_to_config(args.config, threshold, args.silence_duration)
        print(f"[calibrate] Updated {args.config}")
    else:
        print("[calibrate] Run with --apply to write values into config.toml")


if __name__ == "__main__":
    main()
