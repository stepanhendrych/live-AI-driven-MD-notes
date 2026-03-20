# Live AI Driven MD Notes

Realtime asistent pro tvorbu studijních poznámek z mluveného výkladu.

Pipeline:
1. Zachytí audio z mikrofonu.
2. Whisper - transkripce
3. Ollama zpracování na strukturovaný markdown
4. Append do výstupního `.md` souboru

## Quick Start

### 1. Vytvoř a aktivuj `.venv`

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instaluj dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Nainstaluj model do Ollama
```powershell
ollama pull llama3.2:3b
```

Alternativy:
- `ollama pull mistral:7b`
- `ollama pull qwen2.5:7b`

### 4. Spusť aplikaci
```powershell
python -m live_notes_assistant
```

## Konfigurace

Hlavní konfigurace je v `config.toml`.

Nejčastěji měněné hodnoty:
- `[app].notes_file` - kam se ukládají poznámky
- `[app].model_size` - Whisper model (`tiny`, `base`, `small`, `medium`, `turbo`)
- `[app].ollama_model` - model pro generování poznámek
- `[audio]` - chunk/silence tuning a volba mikrofonu (`audio.device`)
- `[transcriber]` - kvalita přepisu + CPU/GPU přepínače
- `[processor]` - timeout, retry a feedback logy pro Ollama

Whisper model switch (`app.model_size`):
- `tiny` - nejrychlejsi, nejmene presny
- `base` - dobry default
- `small` - lepsi presnost za cenu vyssi latence
- `medium` - vysoka presnost, pomalejsi
- `turbo` - rychly model optimalizovany pro prepis

Dulezite switche pro transcribe:
- `transcriber.compute_device = "auto" | "cpu" | "cuda"`
- `transcriber.fp16_mode = "auto" | true | false`
- `transcriber.beam_size`, `transcriber.best_of`, `transcriber.temperature`

Dulezite switche pro Ollama stabilitu:
- `processor.ollama_timeout`
- `processor.ollama_retries`
- `processor.retry_backoff_seconds`
- `processor.feedback_preview_chars`
- `processor.max_context_chars`
- `processor.min_bullet_points`
- `processor.max_section_lines`
- `processor.prompt_mode = "fast" | "balanced" | "detailed"`
- `processor.two_phase_extraction = true | false`

Vyber mikrofonu v `config.toml`:
- `device = ""` pouzije system default
- `device = 1` pouzije konkretni index
- `device = "Microphone (USB Audio Device)"` pouzije konkretni nazev

Vypsani dostupnych audio zarizeni:
```powershell
python -m sounddevice
```

Custom cesta ke konfiguraci:
```powershell
$env:LIVE_NOTES_CONFIG = "config.toml"
python -m live_notes_assistant
```

### Kompletní popis `config.toml`

Sekce `[app]`:
- `notes_file`: cílový markdown soubor, do kterého se appendují nové poznámky.
- `model_size`: Whisper model (`tiny`, `base`, `small`, `medium`, `turbo`) pro transkripci.
- `ollama_model`: Ollama model použitý pro převod transcriptu na strukturované poznámky.

Sekce `[audio]`:
- `chunk`: velikost audio bloku ve vzorcích; menší hodnoty = častější zpracování, větší overhead.
- `channels`: počet vstupních kanálů (typicky `1` pro mikrofon).
- `rate`: vzorkovací frekvence vstupu (doporučeno `16000` pro Whisper).
- `device`: vstupní zařízení (`""` = default, číslo = index, text = název zařízení).
- `silence_threshold`: práh ticha pro detekci pauzy; nižší hodnota = citlivější na ticho.
- `silence_duration`: jak dlouho musí trvat ticho, aby se chunk flushnul do transkripce.
- `max_chunk_seconds`: maximální délka chunku i bez ticha (ochrana proti nekonečně dlouhému chunku).

Sekce `[transcriber]`:
- `language`: jazyk Whisper transkripce (`cs` pro češtinu).
- `sample_width`: šířka audio sample v bytech (`2` = int16).
- `compute_device`: zařízení pro Whisper (`auto`, `cpu`, `cuda`).
- `fp16_mode`: fp16 režim (`auto`, `true`, `false`), využije se hlavně na CUDA.
- `temperature`: dekódovací teplota; nižší = determinističtější přepis.
- `beam_size`: šířka beam search; vyšší může zlepšit přesnost za cenu latence.
- `best_of`: počet kandidátů při samplingu; vyšší může zlepšit kvalitu za cenu výkonu.
- `condition_on_previous_text`: zda Whisper navazuje na předchozí text při dekódování.
- `no_speech_threshold`: práh pro odfiltrování segmentů bez řeči.
- `logprob_threshold`: práh log-probability pro přijetí/odmítnutí segmentu.

Sekce `[processor]`:
- `context_lines`: kolik posledních řádků poznámek se posílá do promptu jako kontext.
- `heading_lines`: kolik posledních nadpisů se posílá do promptu jako outline.
- `ollama_timeout`: základní timeout jednoho volání Ollama (v sekundách).
- `ollama_retries`: kolikrát se Ollama po chybě/timeoutu zkusí znovu.
- `retry_backoff_seconds`: čekání mezi retry pokusy (lineárně roste podle pokusu).
- `feedback_preview_chars`: délka preview logu pro stdout/stderr/think text.
- `min_bullet_points`: minimální počet bullet pointů, aby výstup prošel quality checkem.
- `max_section_lines`: maximální počet řádků jedné sekce.
- `prompt_mode`: `"fast"` / `"balanced"` / `"detailed"` – úroveň detailu v poznámkách.
- `two_phase_extraction`: `true` = dvoufázový pipeline (nejprve extrakce faktů, pak renderování markdownu) pro lepší kvalitu výstupu; `false` = jednofázový prompt (rychlejší).

Sekce `[queues]`:
- `audio_maxsize`: maximální počet audio chunků ve frontě capture -> transcriber.
- `transcript_maxsize`: maximální počet transcriptů ve frontě transcriber -> processor.

## Troubleshooting

### Utility skripty

Kalibrace `silence_threshold` podle realneho mikrofonu:
```powershell
python scripts/calibrate_silence.py
```

Kalibrace a automaticky zapis do `config.toml`:
```powershell
python scripts/calibrate_silence.py --apply
```

Kalibrace s vlastnim zarizenim a prepisem `silence_duration`:
```powershell
python scripts/calibrate_silence.py --device 1 --silence-duration 1.6 --apply
```

Bypass audio/transcriber a test, ze Ollama opravdu zapisuje do souboru:
```powershell
python scripts/test_ollama_bypass.py --transcript "Humanismus a renesance v Anglii: Thomas More a Utopia."
```

Bypass s transcriptem ze souboru:
```powershell
python scripts/test_ollama_bypass.py --transcript-file "HumanismusARenesance-anglie-test.md"
```

Bypass s kontextem puvodnich poznamek (default bypass je bez kontextu):
```powershell
python scripts/test_ollama_bypass.py --transcript "Test" --use-context
```

Bypass test s vlastnim vystupnim souborem:
```powershell
python scripts/test_ollama_bypass.py --notes-file "HumanismusARenesance-anglie-test.md"
```

### Sounddevice install

Linux:
```bash
sudo apt-get install libportaudio2
```

macOS:
```bash
brew install portaudio
```

Windows:
- vetsinou staci `pip install sounddevice`
- pokud chybi PortAudio runtime, doinstaluj ovladace/audio runtime v systemu

### Ollama není dostupná

- ověř: `ollama --version`
- ověř, že běží služba a model je pullnutý

### Nic se nepřipisuje do souboru

- zkontroluj `app.notes_file` v `config.toml`
- zvyš `processor.ollama_timeout`
- sniž `audio.silence_threshold` (nebo uprav podle mikrofonu)
- zkus nastavit `audio.device` na konkretni index/nazev